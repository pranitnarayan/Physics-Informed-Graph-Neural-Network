from __future__ import annotations

import csv
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# Configuration

HERE = Path(__file__).resolve().parent
CSV_PATH = HERE / "cartpole.csv"

SEED = 42
EPOCHS = 600
HIDDEN = (32, 32)
BATCH_SIZE = 256
LEARNING_RATE = 1.0e-3
N_COLLOCATION = 4096
VAL_FRAC = 0.15
TEST_FRAC = 0.15

INPUT_COLUMNS = ["x", "x_dot", "theta", "theta_dot", "F"]
OUTPUT_COLUMNS = ["x_ddot", "theta_ddot"]

COLORS = {
    "actual": "#171717",
    "pinn": "#d1495b",
    "physics": "#00798c",
    "data": "#3066be",
    "validation": "#f28e2b",
    "total": "#6a4c93",
}

# Utilities


def load_csv(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        values = np.asarray([[float(v) for v in row] for row in reader], dtype=np.float64)
    if len(values) == 0:
        raise ValueError(f"No data rows found in {path}")
    return header, values


def column(data: np.ndarray, header: list[str], name: str) -> np.ndarray:
    return data[:, header.index(name)]


def r2_score(truth: np.ndarray, pred: np.ndarray) -> np.ndarray:
    residual = np.sum((truth - pred) ** 2, axis=0)
    total = np.sum((truth - truth.mean(axis=0)) ** 2, axis=0)
    return 1.0 - residual / total


def metrics(truth: np.ndarray, pred: np.ndarray) -> dict[str, np.ndarray]:
    error = pred - truth
    return {
        "rmse": np.sqrt(np.mean(error**2, axis=0)),
        "mae": np.mean(np.abs(error), axis=0),
        "max_abs": np.max(np.abs(error), axis=0),
        "r2": r2_score(truth, pred),
    }


def save_table(path: Path, header: list[str], rows: np.ndarray, fmt: str = "%.9e") -> None:
    np.savetxt(path, rows, delimiter=",", header=",".join(header), comments="", fmt=fmt)


# Data and exact cart-pole physics

header, raw = load_csv(CSV_PATH)
required = [
    "t",
    *INPUT_COLUMNS,
    *OUTPUT_COLUMNS,
    "M_true",
    "m_true",
    "l_true",
    "b_true",
    "c_true",
    "g",
]
missing = [name for name in required if name not in header]
if missing:
    raise ValueError(f"CSV is missing required columns: {missing}")

t = column(raw, header, "t")
X = np.column_stack([column(raw, header, name) for name in INPUT_COLUMNS])
Y = np.column_stack([column(raw, header, name) for name in OUTPUT_COLUMNS])

M_CART = float(column(raw, header, "M_true")[0])
M_BOB = float(column(raw, header, "m_true")[0])
L_ROD = float(column(raw, header, "l_true")[0])
B_JOINT = float(column(raw, header, "b_true")[0])
C_CART = float(column(raw, header, "c_true")[0])
GRAVITY = float(column(raw, header, "g")[0])


def physics_acceleration(states: np.ndarray) -> np.ndarray:
    """Return [x_ddot, theta_ddot] from the coupled nonlinear equations."""
    xd = states[:, 1]
    theta = states[:, 2]
    omega = states[:, 3]
    force = states[:, 4]

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    total_mass = M_CART + M_BOB
    determinant = M_BOB * L_ROD**2 * (total_mass - M_BOB * cos_theta**2)

    rhs_x = force - C_CART * xd + M_BOB * L_ROD * omega**2 * sin_theta
    rhs_theta = M_BOB * GRAVITY * L_ROD * sin_theta - B_JOINT * omega

    theta_ddot = (total_mass * rhs_theta - M_BOB * L_ROD * cos_theta * rhs_x) / determinant
    x_ddot = (M_BOB * L_ROD**2 * rhs_x - M_BOB * L_ROD * cos_theta * rhs_theta) / determinant
    return np.column_stack([x_ddot, theta_ddot])


# Check the CSV before training.
csv_physics_rmse = np.sqrt(np.mean((Y - physics_acceleration(X)) ** 2, axis=0))
if np.any(csv_physics_rmse > 1.0e-4):
    raise ValueError(
        "CSV accelerations do not match the declared cart-pole physics: " f"RMSE={csv_physics_rmse}"
    )

# Data split

rng = np.random.default_rng(SEED)
n = len(X)
order = rng.permutation(n)
n_test = int(TEST_FRAC * n)
n_val = int(VAL_FRAC * n)
idx_test = order[:n_test]
idx_val = order[n_test : n_test + n_val]
idx_train = order[n_test + n_val :]

X_train, Y_train = X[idx_train], Y[idx_train]
X_val, Y_val = X[idx_val], Y[idx_val]
X_test, Y_test = X[idx_test], Y[idx_test]

x_scaler = StandardScaler().fit(X_train)
y_scaler = StandardScaler().fit(Y_train)

X_train_z = x_scaler.transform(X_train)
Y_train_z = y_scaler.transform(Y_train)
X_val_z = x_scaler.transform(X_val)
Y_val_z = y_scaler.transform(Y_val)

domain_lo = X_train.min(axis=0)
domain_hi = X_train.max(axis=0)

# Use a fixed collocation set when recording the loss history.
X_phys_eval = rng.uniform(domain_lo, domain_hi, size=(N_COLLOCATION, X.shape[1]))
Y_phys_eval = physics_acceleration(X_phys_eval)
X_phys_eval_z = x_scaler.transform(X_phys_eval)
Y_phys_eval_z = y_scaler.transform(Y_phys_eval)

# PINN training

model = MLPRegressor(
    hidden_layer_sizes=HIDDEN,
    activation="tanh",
    solver="adam",
    alpha=0.0,
    batch_size=BATCH_SIZE,
    learning_rate_init=LEARNING_RATE,
    max_iter=1,
    shuffle=True,
    random_state=SEED,
    warm_start=True,
)

history = {
    "epoch": [],
    "total_loss": [],
    "data_loss": [],
    "physics_loss": [],
    "validation_loss": [],
    "validation_r2_x_ddot": [],
    "validation_r2_theta_ddot": [],
}

print(
    f"train={len(idx_train)} val={len(idx_val)} test={len(idx_test)} "
    f"collocation/epoch={N_COLLOCATION} epochs={EPOCHS}"
)
print(f"CSV/physics RMSE: x_ddot={csv_physics_rmse[0]:.3e}, theta_ddot={csv_physics_rmse[1]:.3e}")

for epoch in range(1, EPOCHS + 1):
    # Draw new physics points on every epoch.
    X_collocation = rng.uniform(domain_lo, domain_hi, size=(N_COLLOCATION, X.shape[1]))
    Y_collocation = physics_acceleration(X_collocation)

    X_epoch_z = np.vstack([X_train_z, x_scaler.transform(X_collocation)])
    Y_epoch_z = np.vstack([Y_train_z, y_scaler.transform(Y_collocation)])
    model.partial_fit(X_epoch_z, Y_epoch_z)

    train_pred_z = model.predict(X_train_z)
    phys_pred_z = model.predict(X_phys_eval_z)
    val_pred_z = model.predict(X_val_z)

    data_loss = float(np.mean((train_pred_z - Y_train_z) ** 2))
    physics_loss = float(np.mean((phys_pred_z - Y_phys_eval_z) ** 2))
    validation_loss = float(np.mean((val_pred_z - Y_val_z) ** 2))
    total_loss = data_loss + physics_loss
    val_r2 = r2_score(Y_val_z, val_pred_z)

    history["epoch"].append(epoch)
    history["total_loss"].append(total_loss)
    history["data_loss"].append(data_loss)
    history["physics_loss"].append(physics_loss)
    history["validation_loss"].append(validation_loss)
    history["validation_r2_x_ddot"].append(val_r2[0])
    history["validation_r2_theta_ddot"].append(val_r2[1])

    if epoch == 1 or epoch % 25 == 0:
        print(
            f"epoch {epoch:3d} total={total_loss:.3e} data={data_loss:.3e} "
            f"physics={physics_loss:.3e} val={validation_loss:.3e}"
        )

# Evaluation and numerical exports

Y_pred_all = y_scaler.inverse_transform(model.predict(x_scaler.transform(X)))
Y_pred_test = Y_pred_all[idx_test]
test_stats = metrics(Y_test, Y_pred_test)

physics_test = physics_acceleration(X_test)
physics_residual = Y_pred_test - physics_test
physics_rmse = np.sqrt(np.mean(physics_residual**2, axis=0))
physics_max = np.max(np.abs(physics_residual), axis=0)

split = np.full(n, "train", dtype=object)
split[idx_val] = "validation"
split[idx_test] = "test"

with (HERE / "cartpole_pinn_predictions.csv").open("w", newline="") as handle:
    writer = csv.writer(handle, lineterminator="\n")
    writer.writerow(
        [
            "t",
            "split",
            "x_ddot_actual",
            "x_ddot_pinn",
            "x_ddot_error",
            "theta_ddot_actual",
            "theta_ddot_pinn",
            "theta_ddot_error",
        ]
    )
    for row in zip(
        t,
        split,
        Y[:, 0],
        Y_pred_all[:, 0],
        Y_pred_all[:, 0] - Y[:, 0],
        Y[:, 1],
        Y_pred_all[:, 1],
        Y_pred_all[:, 1] - Y[:, 1],
    ):
        writer.writerow([f"{row[0]:.9e}", row[1], *[f"{value:.9e}" for value in row[2:]]])

history_array = np.column_stack([history[name] for name in history])
save_table(HERE / "cartpole_pinn_training_history.csv", list(history), history_array)

with (HERE / "cartpole_pinn_metrics.csv").open("w", newline="") as handle:
    writer = csv.writer(handle, lineterminator="\n")
    writer.writerow(
        ["output", "rmse", "mae", "max_abs_error", "r2", "physics_rmse", "physics_max_abs"]
    )
    for i, name in enumerate(OUTPUT_COLUMNS):
        values = [
            test_stats["rmse"][i],
            test_stats["mae"][i],
            test_stats["max_abs"][i],
            test_stats["r2"][i],
            physics_rmse[i],
            physics_max[i],
        ]
        writer.writerow([name, *[f"{value:.9e}" for value in values]])

# Store the model and both scalers in one file.
model_bundle = pickle.dumps({"model": model, "x_scaler": x_scaler, "y_scaler": y_scaler})
np.savez_compressed(
    HERE / "cartpole_pinn_model.npz",
    model_bundle=np.frombuffer(model_bundle, dtype=np.uint8),
    parameters=np.array([M_CART, M_BOB, L_ROD, B_JOINT, C_CART, GRAVITY]),
)

print("\nTest metrics")
for i, name in enumerate(OUTPUT_COLUMNS):
    print(
        f"{name:12s} RMSE={test_stats['rmse'][i]:.5f} "
        f"MAE={test_stats['mae'][i]:.5f} R2={test_stats['r2'][i]:.6f} "
        f"physics_RMSE={physics_rmse[i]:.5f}"
    )

# Scientific plots

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "axes.titleweight": "normal",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)

epochs = np.asarray(history["epoch"])


def style_axis(ax: plt.Axes) -> None:
    ax.grid(True, which="both", alpha=0.22, linewidth=0.7)


def plot_losses(ax: plt.Axes) -> None:
    ax.semilogy(epochs, history["total_loss"], color=COLORS["total"], lw=1.8, label="Total loss")
    ax.semilogy(
        epochs, history["data_loss"], color=COLORS["data"], ls="--", lw=1.3, label="Data loss"
    )
    ax.semilogy(
        epochs,
        history["physics_loss"],
        color=COLORS["physics"],
        ls=":",
        lw=1.7,
        label="Physics loss",
    )
    ax.semilogy(
        epochs,
        history["validation_loss"],
        color=COLORS["validation"],
        lw=1.3,
        label="Validation loss",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean squared loss")
    ax.set_title("Training and Validation Loss")
    ax.legend(frameon=False)
    style_axis(ax)


def plot_parity(ax: plt.Axes, output: int, title: str, units: str) -> None:
    truth = Y_test[:, output]
    pred = Y_pred_test[:, output]
    lo = min(truth.min(), pred.min())
    hi = max(truth.max(), pred.max())
    pad = 0.04 * (hi - lo)
    ax.plot(
        [lo - pad, hi + pad],
        [lo - pad, hi + pad],
        color="black",
        ls=":",
        lw=1.6,
        label="Ideal prediction",
    )
    ax.scatter(
        truth, pred, s=11, alpha=0.42, color=COLORS["pinn"], edgecolors="none", label="Test data"
    )
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"Actual [{units}]")
    ax.set_ylabel(f"PINN [{units}]")
    ax.set_title(f"{title}  R² = {test_stats['r2'][output]:.5f}")
    ax.legend(frameon=False, loc="upper left")
    style_axis(ax)


def plot_time(
    ax: plt.Axes, output: int, title: str, units: str, mask: np.ndarray | None = None
) -> None:
    use = np.ones(n, dtype=bool) if mask is None else mask
    # Plot every second point to keep the figure readable.
    ids = np.flatnonzero(use)[::2]
    ax.plot(t[ids], Y[ids, output], color=COLORS["actual"], lw=1.5, label="Simulation data")
    ax.plot(
        t[ids],
        Y_pred_all[ids, output],
        color=COLORS["pinn"],
        lw=1.35,
        ls=":",
        label="PINN prediction",
    )
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(units)
    ax.set_title(title)
    ax.legend(frameon=False, ncol=2)
    style_axis(ax)


def plot_validation_r2(ax: plt.Axes) -> None:
    ax.plot(
        epochs,
        history["validation_r2_x_ddot"],
        color=COLORS["data"],
        lw=1.6,
        label="Cart acceleration",
    )
    ax.plot(
        epochs,
        history["validation_r2_theta_ddot"],
        color=COLORS["pinn"],
        ls=":",
        lw=1.8,
        label="Angular acceleration",
    )
    ax.axhline(1.0, color="black", ls="--", lw=1.0, label="Ideal value")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation R²")
    ax.set_ylim(0.0, 1.025)
    ax.set_title("Validation Accuracy")
    ax.legend(frameon=False, loc="lower right")
    style_axis(ax)


# Training history
fig, ax = plt.subplots(figsize=(10.5, 6.2))
plot_losses(ax)
fig.suptitle("Cart Pole PINN Training History", fontsize=15)
fig.tight_layout()
fig.savefig(HERE / "cartpole_pinn_training_iterations.png", dpi=220, bbox_inches="tight")
plt.close(fig)

# Time histories
fig, axes = plt.subplots(2, 2, figsize=(14, 8.2), sharex="col")
plot_time(axes[0, 0], 0, "Cart Acceleration over the Full Simulation", "m/s²")
plot_time(axes[1, 0], 1, "Angular Acceleration over the Full Simulation", "rad/s²")
zoom = t <= 6.0
plot_time(axes[0, 1], 0, "Cart Acceleration during Swing Up (0 to 6 s)", "m/s²", zoom)
plot_time(axes[1, 1], 1, "Angular Acceleration during Swing Up (0 to 6 s)", "rad/s²", zoom)
fig.suptitle("Simulation Data and PINN Predictions", fontsize=15)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(HERE / "cartpole_pinn_actual_vs_predicted.png", dpi=220, bbox_inches="tight")
plt.close(fig)

# Test errors
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
plot_parity(axes[0, 0], 0, "Cart Acceleration", "m/s²")
plot_parity(axes[0, 1], 1, "Angular Acceleration", "rad/s²")
for output, (title, units) in enumerate(
    [("Cart Acceleration Error", "m/s²"), ("Angular Acceleration Error", "rad/s²")]
):
    err = Y_pred_test[:, output] - Y_test[:, output]
    axes[1, output].hist(err, bins=45, color=COLORS["physics"], alpha=0.82, edgecolor="white")
    axes[1, output].axvline(0, color="black", ls=":", lw=1.5)
    axes[1, output].set_xlabel(f"PINN − actual [{units}]")
    axes[1, output].set_ylabel("Number of test samples")
    axes[1, output].set_title(f"{title}  RMSE = {test_stats['rmse'][output]:.4f} {units}")
    style_axis(axes[1, output])
fig.suptitle("Cart Pole PINN Prediction Diagnostics", fontsize=15)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(HERE / "cartpole_pinn_diagnostics.png", dpi=220, bbox_inches="tight")
plt.close(fig)

# Combined results
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
plot_losses(axes[0, 0])
plot_parity(axes[0, 1], 0, "Cart Acceleration", "m/s²")
plot_parity(axes[0, 2], 1, "Angular Acceleration", "rad/s²")
plot_time(axes[1, 0], 0, "Cart Acceleration Prediction", "m/s²")
plot_time(axes[1, 1], 1, "Angular Acceleration Prediction", "rad/s²")
plot_validation_r2(axes[1, 2])
fig.suptitle("Physics Informed Neural Network Results for the Cart Pole", fontsize=16)
fig.tight_layout(rect=(0, 0, 1, 0.965))
fig.savefig(HERE / "cartpole_pinn_all_results.png", dpi=220, bbox_inches="tight")
plt.close(fig)

print("\nSaved PINN model, CSV values, and four PNG figures in:")
print(HERE)
