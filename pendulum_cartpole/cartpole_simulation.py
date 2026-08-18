from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.patches import Circle, Rectangle, FancyArrow

# Output files

SAVE_DIR = Path(__file__).resolve().parent
SAVE_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = SAVE_DIR / "cartpole.csv"
VIDEO_PATH = SAVE_DIR / "cartpole.mp4"

# Model and controller settings

G = 9.81

M_CART = 1.0  # cart mass, kg
M_BOB = 0.30  # pendulum bob mass, kg
L_ROD = 0.62  # rod length, m

B_JOINT = 0.010  # viscous damping at hinge, N*m*s
C_CART = 0.10  # rolling friction on cart, N*s/m

F_MAX = 22.0
F_KEY = 7.0

X_LIM = 2.3

# Balance controller gains
KP = 130.0
KD = 24.0
KX = 5.0
KV = 8.5

# Swing-up controller
K_SWING = 16.0
F_SWING = 8.0

# Switch from swing-up to balancing
CATCH = 0.45

# Simulation
DT = 0.001
SUBSTEPS = 16

# Simulation model


class CartPole:

    def __init__(self):
        self.reset()

    def reset(self):

        self.t = 0.0
        # State:
        #
        # x
        # x_dot
        # theta
        # theta_dot
        #
        # theta = 0 means upright
        self.s = np.array([0.0, 0.0, np.pi - 0.12, 0.0])
        self.push = 0.0
        self.log = []

    def target(self):
        """Return the moving cart-position reference."""
        return 1.35 * np.sin(2 * np.pi * self.t / 13.0)

    def force(self, s):

        x, xd, th, om = s
        m = M_BOB
        l = L_ROD
        # Wrap angle to [-pi, pi]
        wrapped = (th + np.pi) % (2 * np.pi) - np.pi
        f = self.push
        # Swing-up control
        if abs(wrapped) > CATCH:
            # Pendulum energy
            E = 0.5 * m * l * l * om * om + m * G * l * np.cos(th)
            # Energy pumping
            f += K_SWING * (E - m * G * l) * np.sign(om * np.cos(th))
            # Limit swing-up force
            f = float(np.clip(f, -F_SWING, F_SWING))
            # Prevent the cart running away
            f -= 0.9 * xd + 1.4 * x
        # Balance control
        else:
            f += KP * np.sin(wrapped) + KD * om + KX * (x - self.target()) + KV * xd
        return float(np.clip(f, -F_MAX, F_MAX))

    def deriv(self, s):

        x, xd, th, om = s
        m = M_BOB
        M = M_CART
        l = L_ROD
        F = self.force(s)
        c = np.cos(th)
        st = np.sin(th)
        D = M + m
        det = m * l * l * (D - m * c * c)
        rhs_x = F - C_CART * xd + m * l * om * om * st
        rhs_t = m * G * l * st - B_JOINT * om
        thdd = (D * rhs_t - m * l * c * rhs_x) / det
        xdd = (m * l * l * rhs_x - m * l * c * rhs_t) / det
        return (np.array([xd, xdd, om, thdd]), F, xdd, thdd)

    def step(self, dt):

        # RK4 integration
        k1, F, xdd, thdd = self.deriv(self.s)
        k2, *_ = self.deriv(self.s + dt / 2 * k1)
        k3, *_ = self.deriv(self.s + dt / 2 * k2)
        k4, *_ = self.deriv(self.s + dt * k3)
        self.s = self.s + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        self.t += dt
        # Track limits
        if abs(self.s[0]) > X_LIM:
            self.s[0] = float(np.clip(self.s[0], -X_LIM, X_LIM))
            self.s[1] *= -0.3
        # Log the updated state
        _, F_new, xdd_new, thdd_new = self.deriv(self.s)
        x, xd, th, om = self.s
        self.log.append([self.t, x, xd, xdd_new, th, om, thdd_new, F_new])

    def energy(self):

        x, xd, th, om = self.s
        m = M_BOB
        M = M_CART
        l = L_ROD
        vx = xd + l * om * np.cos(th)
        vy = -l * om * np.sin(th)
        return 0.5 * M * xd**2 + 0.5 * m * (vx**2 + vy**2) + m * G * l * np.cos(th)

    def save_csv(self, path=CSV_PATH, rate=200.0):
        """Save logged samples at the requested rate."""
        if len(self.log) == 0:
            print("No simulation data available to save.")
            return
        a = np.array(self.log)
        stride = max(1, int(round(1.0 / (rate * DT))))
        a = a[::stride]
        truth = np.tile([M_CART, M_BOB, L_ROD, B_JOINT, C_CART, G], (len(a), 1))
        output = np.hstack([a, truth])
        np.savetxt(
            path,
            output,
            delimiter=",",
            header=(
                "t,"
                "x,"
                "x_dot,"
                "x_ddot,"
                "theta,"
                "theta_dot,"
                "theta_ddot,"
                "F,"
                "M_true,"
                "m_true,"
                "l_true,"
                "b_true,"
                "c_true,"
                "g"
            ),
            comments="",
            fmt="%.6e",
        )
        print(f"\nCSV saved successfully:\n" f"{path}\n" f"{len(a)} rows @ {rate:g} Hz\n")


# Simulation state

sim = CartPole()

# Figure setup

BODY = "#a8a8a8"
EDGE = "#000000"
BOB = "#0a0a0a"
WHEEL = "#9d9d9d"

PUSH_R = "#c1121f"
PUSH_L = "#0353a4"

fig, (ax, axg) = plt.subplots(
    2, 1, figsize=(10, 8), facecolor="white", height_ratios=[2.4, 1], gridspec_kw=dict(hspace=0.22)
)

fig.canvas.manager.set_window_title("Cart-pole")

# Cart and pendulum view

ax.set_facecolor("white")

ax.set_xlim(-2.75, 2.75)

ax.set_ylim(-0.30, 1.52)

ax.set_aspect("equal")

ax.axis("off")

GROUND = 0.0

WHEEL_R = 0.235

DECK_H = 0.24

DECK_W = 1.15

DECK_Y = GROUND + 2 * WHEEL_R - 0.02

PIVOT_Y = DECK_Y + DECK_H

# Ground

ax.plot([-2.75, 2.75], [GROUND, GROUND], color=EDGE, lw=1.6, zorder=1)

for hx in np.arange(-2.75, 2.78, 0.115):

    ax.plot([hx, hx - 0.085], [GROUND, GROUND - 0.095], color=EDGE, lw=1.1, zorder=1)

# Cart

wheel_l = Circle((0, GROUND + WHEEL_R), WHEEL_R, fc=WHEEL, ec=EDGE, lw=1.6, zorder=2)

wheel_r = Circle((0, GROUND + WHEEL_R), WHEEL_R, fc=WHEEL, ec=EDGE, lw=1.6, zorder=2)

ax.add_patch(wheel_l)
ax.add_patch(wheel_r)

deck = Rectangle((0, DECK_Y), DECK_W, DECK_H, fc=BODY, ec=EDGE, lw=1.8, zorder=3)

ax.add_patch(deck)

# Pendulum

(rod,) = ax.plot([], [], color=BODY, lw=11, solid_capstyle="round", zorder=4)

(rod_edge,) = ax.plot([], [], color=EDGE, lw=13.5, solid_capstyle="round", zorder=3.5)

bob = Circle((0, 0), 0.125, fc=BOB, ec=EDGE, lw=1.5, zorder=6)

ax.add_patch(bob)

hinge = Circle((0, PIVOT_Y), 0.055, fc=BODY, ec=EDGE, lw=1.5, zorder=5)

ax.add_patch(hinge)

(vline,) = ax.plot([], [], color="#b9b9b9", lw=1.0, zorder=2)

(arc,) = ax.plot([], [], color=EDGE, lw=1.4, zorder=5)

arrow_F = [None]

# Live plot

WIN = 14.0

axg.set_facecolor("white")

axg.set_xlim(0, WIN)

axg.set_ylim(-185, 185)

axg.axhline(0, color="#cccccc", lw=0.9, zorder=1)

axg.set_yticks([-180, -90, 0, 90, 180])

axg.tick_params(labelsize=8, colors="#555555", length=3)

for side in ("top", "right"):

    axg.spines[side].set_visible(False)

for side in ("left", "bottom"):

    axg.spines[side].set_color("#999999")

    axg.spines[side].set_linewidth(0.9)

# Angle curve

(tr_th,) = axg.plot([], [], color="#0a0a0a", lw=1.6, zorder=4)

# Force axis

axf = axg.twinx()

axf.set_xlim(0, WIN)

axf.set_ylim(-F_MAX * 1.08, F_MAX * 1.08)

axf.tick_params(labelsize=8, colors="#555555", length=3)

for side in ("top", "left"):

    axf.spines[side].set_visible(False)

axf.spines["right"].set_color("#999999")

axf.spines["bottom"].set_color("#999999")

fill_pos = [None]
fill_neg = [None]

(tr_F,) = axf.plot([], [], color="#777777", lw=0.8, alpha=0.5, zorder=2)

running = True

# Drawing


def render():

    x, xd, th, om = sim.s

    F = sim.force(sim.s)

    # Cart

    wheel_l.center = (x - 0.33, GROUND + WHEEL_R)

    wheel_r.center = (x + 0.33, GROUND + WHEEL_R)

    deck.set_x(x - DECK_W / 2)

    hinge.center = (x, PIVOT_Y)

    # Pendulum

    bx = x + L_ROD * np.sin(th)

    by = PIVOT_Y + L_ROD * np.cos(th)

    rod.set_data([x, bx], [PIVOT_Y, by])

    rod_edge.set_data([x, bx], [PIVOT_Y, by])

    bob.center = (bx, by)

    vline.set_data([], [])

    # Angle arc

    r = 0.42 * L_ROD

    a = np.linspace(0, th, 40)

    arc.set_data([], [])

    # Force arrow

    if arrow_F[0] is not None:

        arrow_F[0].remove()
        arrow_F[0] = None

    if abs(F) > 0.15:

        mag = min(abs(F) / F_MAX, 1.0)
        span = 0.16 + 0.62 * mag
        d = 1.0 if F > 0 else -1.0
        tail = x + d * (DECK_W / 2 + 0.05)
        arrow_F[0] = FancyArrow(
            tail,
            DECK_Y + DECK_H / 2,
            d * span,
            0,
            width=(0.020 + 0.026 * mag),
            head_width=(0.085 + 0.075 * mag),
            head_length=(0.11 + 0.07 * mag),
            color=(PUSH_R if F > 0 else PUSH_L),
            alpha=(0.45 + 0.55 * mag),
            zorder=6,
            length_includes_head=True,
        )
        ax.add_patch(arrow_F[0])

    # Live data

    if sim.log:

        data = np.array(sim.log)
        t0 = max(0.0, sim.t - WIN)
        keep = data[:, 0] >= t0
        tt = data[keep, 0]
        th_deg = np.degrees((data[keep, 4] + np.pi) % (2 * np.pi) - np.pi)
        ff = data[keep, 7]
        # Prevent line across ±180 degree wrap
        th_plot = th_deg.copy()
        jump = np.abs(np.diff(th_plot)) > 180.0
        th_plot[1:][jump] = np.nan
        tr_th.set_data(tt, th_plot)
        tr_F.set_data(tt, ff)
        # Remove previous fills
        for fill in (fill_pos, fill_neg):
            if fill[0] is not None:
                fill[0].remove()
                fill[0] = None
        # Force fill
        if len(tt) > 1:
            fill_pos[0] = axf.fill_between(
                tt, 0, ff, where=(ff > 0), color=PUSH_R, alpha=0.30, lw=0, zorder=1
            )
            fill_neg[0] = axf.fill_between(
                tt, 0, ff, where=(ff < 0), color=PUSH_L, alpha=0.30, lw=0, zorder=1
            )
        axg.set_xlim(t0, t0 + WIN)
        axf.set_xlim(t0, t0 + WIN)
        # Angle-axis scaling
        yt = max(12.0, 1.25 * float(np.nanmax(np.abs(th_deg))))
        axg.set_ylim(-yt, yt)
        if yt > 120:
            step = 90
        elif yt > 60:
            step = 45
        elif yt > 25:
            step = 10
        else:
            step = 5
        axg.set_yticks(np.arange(-np.floor(yt / step) * step, np.floor(yt / step) * step + 1, step))
        # Force-axis scaling
        yf = max(2.5, 1.25 * float(np.abs(ff).max()))
        axf.set_ylim(-yf, yf)

    return ()


# Video export


def record(path=VIDEO_PATH, seconds=30.0, fps=30):
    """Record a fresh simulation and save its CSV data."""

    global running

    was_running = running

    running = False

    sim.reset()

    n_frames = int(seconds * fps)

    steps = int(round((1.0 / fps) / DT))

    # Video writer

    try:

        writer = FFMpegWriter(fps=fps, bitrate=2400)
        out = Path(path)

    except Exception:
        writer = PillowWriter(fps=fps)
        out = Path(path).with_suffix(".gif")

    print()
    print(f"Recording {seconds:.0f} seconds")

    print(f"Video destination:\n{out}")

    # Record frames

    with writer.saving(fig, str(out), dpi=110):

        for i in range(n_frames):
            for _ in range(steps):
                sim.step(DT)
            render()
            writer.grab_frame()
            if (i + 1) % (fps * 5) == 0:
                print(f"{(i + 1) / fps:.0f} s recorded")

    print(f"\nVideo saved successfully:\n{out}")

    # Save matching CSV data

    sim.save_csv(CSV_PATH)
    print()
    print("Recording complete.")

    print(f"VIDEO:\n{out}")

    print(f"\nCSV:\n{CSV_PATH}")

    running = was_running


# Keyboard input


def on_press(e):

    global running

    # SPACE
    if e.key == " ":

        running = not running

    # RESET
    elif e.key == "r":

        sim.reset()
        for fill in (fill_pos, fill_neg):
            if fill[0] is not None:
                fill[0].remove()
                fill[0] = None

    # SAVE CSV
    elif e.key == "s":

        sim.save_csv(CSV_PATH)

    # RECORD VIDEO + CSV
    elif e.key == "v":

        record(VIDEO_PATH)

    # PUSH LEFT
    elif e.key == "left":

        sim.push = -F_KEY

    # PUSH RIGHT
    elif e.key == "right":

        sim.push = F_KEY


# Key release


def on_release(e):

    if e.key in ("left", "right"):

        sim.push = 0.0


# Event handlers

fig.canvas.mpl_connect("key_press_event", on_press)

fig.canvas.mpl_connect("key_release_event", on_release)

# Animation


def draw(_):

    if running:

        for _ in range(SUBSTEPS):
            sim.step(DT)

    return render()


anim = FuncAnimation(fig, draw, interval=16, blit=False, cache_frame_data=False)

# Program entry point

if __name__ == "__main__":

    print("\nCart-Pole Simulation")

    print("SPACE  : pause/resume")

    print("LEFT   : push left")

    print("RIGHT  : push right")

    print("R      : reset")

    print("S      : save CSV")

    print("V      : record 30 s video + CSV")

    print("\nFiles will be saved in:")

    print(SAVE_DIR)

    print()

    plt.show()
