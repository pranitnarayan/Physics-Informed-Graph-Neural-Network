"""
=============================================================================
SIMPLE PINN FOR THE UNFORCED DAMPED PENDULUM
=============================================================================

Learns the dynamics map

        f : (L, theta, theta_dot, b)  ->  theta_ddot

for the equation of motion in Pendulum_Physics.py with tau = 0:

        m*L^2 * theta_ddot + b * theta_dot + m*g*L*sin(theta) = 0
   =>   theta_ddot = -(b/(m*L^2)) * theta_dot - (g/L) * sin(theta)

The dataset (dataset_pendolo_m1_g981.csv) is 10 000 INDEPENDENT state samples,
not a time series, so there is no time integration here. Each row is one point
in the 4-D input domain with its exact angular acceleration.

TOTAL LOSS
----------
        L_total = L_data  +  lambda_phys * L_physics

    L_data     supervised MSE on the labelled rows.
    L_physics  ODE residual evaluated at COLLOCATION points sampled uniformly
               from the input domain. These carry no labels -- the physics
               alone tells the network what the answer must be there. This is
               what makes it a PINN rather than a plain regressor: it
               regularises the network across the whole domain, including
               regions the 10 000 samples cover sparsely.

NOTE ON THIS PARTICULAR ODE
---------------------------
The residual is ALGEBRAIC in the network output (theta_ddot appears directly,
not as d^2/dt^2 of a predicted theta(t)). So no autograd derivatives are
needed, and the physics term is an unusually strong signal -- strong enough
that the network can be trained with N_DATA = 0. That is deliberately left as
a knob below so you can see the effect.
=============================================================================
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ------------------------------------------------------------------ config
CSV_PATH   = "dataset_pendolo_m1_g981.csv"
M          = 1.0        # kg   -- from the filename: _m1_
G          = 9.81       # m/s2 -- from the filename: _g981

SEED         = 42
HIDDEN       = [64, 64, 64]
EPOCHS       = 600
BATCH_SIZE   = 256      # set to None for full-batch training
LR           = 1e-3
LAMBDA_PHYS  = 1.0      # weight on the physics residual
N_COLLOC     = 2048     # collocation points resampled every epoch
LOSS_FN      = "mse"    # "mse" or "huber"
VAL_FRAC     = 0.15
TEST_FRAC    = 0.15
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(SEED)
np.random.seed(SEED)


# ------------------------------------------------------------------ physics
def physics_theta_ddot(L, theta, theta_dot, b):
    """
    Right-hand side of the equation of motion with tau = 0.
    Works on torch tensors. This is exactly Pendulum.accel(theta, theta_dot, 0)
    from Pendulum_Physics.py, with I = m*L^2.
    """
    return (-b * theta_dot - M * G * L * torch.sin(theta)) / (M * L ** 2)


# ------------------------------------------------------------------ data
df = pd.read_csv(CSV_PATH)
X_all = df[["L_m", "theta_rad", "theta_dot_rad_s", "damping_b"]].to_numpy(np.float32)
y_all = df[["theta_ddot_rad_s2"]].to_numpy(np.float32)

n = len(df)
perm = np.random.permutation(n)
n_test = int(TEST_FRAC * n)
n_val = int(VAL_FRAC * n)
idx_test, idx_val, idx_train = perm[:n_test], perm[n_test:n_test + n_val], perm[n_test + n_val:]

X_train, y_train = X_all[idx_train], y_all[idx_train]
X_val,   y_val   = X_all[idx_val],   y_all[idx_val]
X_test,  y_test  = X_all[idx_test],  y_all[idx_test]

# Standardisation statistics come from the TRAINING SPLIT ONLY (no leakage).
x_mu, x_sd = X_train.mean(0), X_train.std(0)
y_mu, y_sd = y_train.mean(0), y_train.std(0)

# Collocation domain = bounding box of the training inputs.
lo, hi = X_train.min(0), X_train.max(0)

to_t = lambda a: torch.tensor(a, dtype=torch.float32, device=DEVICE)
X_train_t, y_train_t = to_t(X_train), to_t(y_train)
X_val_t,   y_val_t   = to_t(X_val),   to_t(y_val)
X_test_t,  y_test_t  = to_t(X_test),  to_t(y_test)
x_mu_t, x_sd_t = to_t(x_mu), to_t(x_sd)
y_mu_t, y_sd_t = to_t(y_mu), to_t(y_sd)
lo_t, hi_t = to_t(lo), to_t(hi)


# ------------------------------------------------------------------ model
class PINN(nn.Module):
    """
    MLP with tanh activations. Inputs and outputs are standardised internally,
    so the network always sees O(1) numbers while the caller works in SI units.
    theta_ddot spans roughly +/-100 rad/s2 here; feeding that raw to an MSE
    loss would make the gradients scale-dominated by the large-|theta_ddot|
    samples.
    """

    def __init__(self, n_in=4, n_out=1, hidden=HIDDEN):
        super().__init__()
        layers, prev = [], n_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.Tanh()]
            prev = h
        layers += [nn.Linear(prev, n_out)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """x in SI units -> theta_ddot in SI units."""
        z = (x - x_mu_t) / x_sd_t
        return self.net(z) * y_sd_t + y_mu_t


model = PINN().to(DEVICE)
optimiser = torch.optim.Adam(model.parameters(), lr=LR)
# Cosine decay rather than a constant LR: Adam at fixed lr converges to a noise
# ball whose radius scales with the step size, which is what makes the loss
# curve rattle. Annealing to LR/100 keeps the fast early progress and shrinks
# that ball at the end.
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimiser, T_max=EPOCHS, eta_min=LR / 100)

_mse = nn.MSELoss()
_huber = nn.HuberLoss(delta=1.0)
criterion = _mse if LOSS_FN == "mse" else _huber


def data_loss(x, y):
    """Supervised loss, computed in standardised units."""
    return criterion((model(x) - y_mu_t) / y_sd_t, (y - y_mu_t) / y_sd_t)


def physics_loss(n_points):
    """
    ODE residual at fresh uniform collocation points. No labels used.
    Divided by y_sd so it sits on the same numerical scale as data_loss.
    """
    xc = lo_t + (hi_t - lo_t) * torch.rand(n_points, 4, device=DEVICE)
    L, theta, theta_dot, b = xc[:, 0:1], xc[:, 1:2], xc[:, 2:3], xc[:, 3:4]
    residual = model(xc) - physics_theta_ddot(L, theta, theta_dot, b)
    return criterion(residual / y_sd_t, torch.zeros_like(residual))


# ------------------------------------------------------------------ training
history = {"total": [], "data": [], "phys": [], "val": [], "lr": []}
n_train = len(X_train_t)
batch = n_train if BATCH_SIZE is None else BATCH_SIZE

print(f"device={DEVICE}  train={n_train}  val={len(X_val_t)}  test={len(X_test_t)}  "
      f"batch={batch}  lambda_phys={LAMBDA_PHYS}")

for epoch in range(1, EPOCHS + 1):
    model.train()
    order = torch.randperm(n_train, device=DEVICE)
    ep_tot = ep_dat = ep_phy = 0.0
    n_batches = 0

    for start in range(0, n_train, batch):
        sel = order[start:start + batch]
        xb, yb = X_train_t[sel], y_train_t[sel]

        # Collocation points are shared per epoch in count but resampled per
        # batch, so the physics term never overfits a fixed point cloud.
        n_col = max(1, round(N_COLLOC * len(sel) / n_train))
        l_dat = data_loss(xb, yb)
        l_phy = physics_loss(n_col)
        loss = l_dat + LAMBDA_PHYS * l_phy

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        ep_tot += loss.item(); ep_dat += l_dat.item(); ep_phy += l_phy.item()
        n_batches += 1

    model.eval()
    with torch.no_grad():
        l_val = data_loss(X_val_t, y_val_t).item()

    history["total"].append(ep_tot / n_batches)
    history["data"].append(ep_dat / n_batches)
    history["phys"].append(ep_phy / n_batches)
    history["val"].append(l_val)
    history["lr"].append(optimiser.param_groups[0]["lr"])
    scheduler.step()   # cosine steps on epoch count, not on a metric

    if epoch % 20 == 0 or epoch == 1:
        print(f"epoch {epoch:4d}  total={history['total'][-1]:.3e}  "
              f"data={history['data'][-1]:.3e}  phys={history['phys'][-1]:.3e}  "
              f"val={l_val:.3e}  lr={history['lr'][-1]:.1e}")


# ------------------------------------------------------------------ evaluation
model.eval()
with torch.no_grad():
    pred = model(X_test_t)
    err = (pred - y_test_t).cpu().numpy().ravel()
    truth = y_test_t.cpu().numpy().ravel()

    rmse = float(np.sqrt((err ** 2).mean()))
    mae = float(np.abs(err).mean())
    r2 = 1.0 - (err ** 2).sum() / ((truth - truth.mean()) ** 2).sum()

    # Physics residual on the held-out test inputs: how well the learned map
    # obeys the equation of motion, independent of the labels.
    L, th, om, b = (X_test_t[:, i:i + 1] for i in range(4))
    phys_res = (pred - physics_theta_ddot(L, th, om, b)).cpu().numpy().ravel()

print("\n--- test set ---")
print(f"RMSE            = {rmse:.4f} rad/s^2")
print(f"MAE             = {mae:.4f} rad/s^2")
print(f"R^2             = {r2:.6f}")
print(f"max |error|     = {np.abs(err).max():.4f} rad/s^2")
print(f"physics residual: RMSE = {np.sqrt((phys_res ** 2).mean()):.4f}, "
      f"max = {np.abs(phys_res).max():.4f} rad/s^2")


# ------------------------------------------------------------------ plots
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))

ep = np.arange(1, EPOCHS + 1)
ax[0].semilogy(ep, history["total"], label="total (train)", lw=1.6)
ax[0].semilogy(ep, history["data"],  label="data (train)",  lw=1.2, ls="--")
ax[0].semilogy(ep, history["phys"],  label="physics",       lw=1.2, ls="--")
ax[0].semilogy(ep, history["val"],   label="data (val)",    lw=1.6)
ax[0].set_xlabel("epoch")
ax[0].set_ylabel(f"{LOSS_FN.upper()} loss (standardised units)")
ax[0].set_title("PINN training loss")
ax[0].legend()
ax[0].grid(alpha=0.3, which="both")

lim = [truth.min(), truth.max()]
ax[1].plot(lim, lim, "k--", lw=1, label="ideal")
ax[1].scatter(truth, pred.cpu().numpy().ravel(), s=4, alpha=0.3)
ax[1].set_xlabel(r"true $\ddot{\theta}$  [rad/s$^2$]")
ax[1].set_ylabel(r"predicted $\ddot{\theta}$  [rad/s$^2$]")
ax[1].set_title(f"Test parity   $R^2$ = {r2:.5f}")
ax[1].legend()
ax[1].grid(alpha=0.3)

fig.tight_layout()
fig.savefig("pinn_loss_curve.png", dpi=150)
print("\nsaved pinn_loss_curve.png")
plt.show()
