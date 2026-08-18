"""
=============================================================================
CONTROLLED DAMPED PENDULUM  --  shared reference implementation
=============================================================================
 
Single rigid link, point mass at the tip, revolute joint, motor at the joint.
 
CONVENTIONS  (please do not silently change these)
-------------------------------------------------
   theta = 0        -> hanging straight DOWN  (stable equilibrium)
   theta = pi       -> straight UP            (unstable equilibrium)
   theta increases  -> counter-clockwise
   Cartesian tip:   x =  L*sin(theta)
                    y = -L*cos(theta)
   tau > 0          -> counter-clockwise motor torque, units N*m
 
EQUATION OF MOTION
------------------
   Lagrangian for a point mass m at distance L:
        I = m*L^2                (moment of inertia about the pivot)
 
        m*L^2 * theta_ddot  +  b * theta_dot  +  m*g*L*sin(theta)  =  tau
 
   Divide through by m*L^2 to get the form used in the project statement:
 
        theta_ddot + (b/(m*L^2)) * theta_dot + (g/L) * sin(theta) = tau/(m*L^2)
 
   NOTE the forcing term is tau/(m*L^2), NOT tau/m or tau/L.
   With tau = 0 this reduces exactly to the warm-up equation.
 
   b has units N*m*s/rad  (viscous joint damping).
 
STATE / TASK
------------
   state   s = [theta, theta_dot]
   goal    theta_d  (desired angle; theta_dot_d = 0 for a rest-to-rest swing)
   network tau = f(theta, theta_dot, theta_d ; params)   -> scalar torque
 
WHY THE TORQUE LIMIT MATTERS
----------------------------
   If tau_max >= m*g*L the network can lift the mass quasi-statically and the
   task is nearly trivial. Set tau_max < m*g*L to force an energy-pumping
   swing-up, which is where the problem gets interesting. Check the ratio
   tau_max / (m*g*L) before running anything.
=============================================================================
"""
 
import numpy as np
from scipy.integrate import solve_ivp
 
# ---------------------------------------------------------------- parameters
G = 9.81
 
 
class Pendulum:
    """Controlled damped pendulum. All SI units."""
 
    def __init__(self, m=0.5, L=0.6, b=0.10, g=G, tau_max=1.5):
        self.m, self.L, self.b, self.g = m, L, b, g
        self.tau_max = tau_max
 
    # ---- useful scalars -------------------------------------------------
    @property
    def I(self):
        """Moment of inertia about the pivot."""
        return self.m * self.L**2
 
    @property
    def tau_gravity_max(self):
        """Peak gravitational torque, at theta = pi/2. Compare to tau_max."""
        return self.m * self.g * self.L
 
    @property
    def omega_n(self):
        """Small-angle natural frequency, rad/s."""
        return np.sqrt(self.g / self.L)
 
    @property
    def zeta(self):
        """Small-angle damping ratio. <1 underdamped, >1 overdamped."""
        return self.b / (2.0 * self.I * self.omega_n)
 
    # ---- dynamics -------------------------------------------------------
    def accel(self, theta, theta_dot, tau):
        """theta_ddot from the equation of motion."""
        return (tau - self.b * theta_dot
                - self.m * self.g * self.L * np.sin(theta)) / self.I
 
    def rhs(self, t, s, controller):
        theta, theta_dot = s
        tau = np.clip(controller(t, theta, theta_dot), -self.tau_max, self.tau_max)
        return [theta_dot, self.accel(theta, theta_dot, tau)]
 
    def simulate(self, controller, s0=(0.0, 0.0), t_end=5.0, dt=0.02):
        """
        controller: (t, theta, theta_dot) -> tau   [Nm, clipped internally]
        returns dict of arrays; tau is recomputed post-hoc for logging.
        """
        t = np.arange(0.0, t_end + 1e-12, dt)
        sol = solve_ivp(self.rhs, (t[0], t[-1]), list(s0), t_eval=t,
                        args=(controller,), method="DOP853",
                        rtol=1e-10, atol=1e-12)
        if not sol.success:
            raise RuntimeError(sol.message)
        th, om = sol.y
        tau = np.clip([controller(ti, thi, omi) for ti, thi, omi in zip(t, th, om)],
                      -self.tau_max, self.tau_max)
        acc = self.accel(th, om, tau)
        return dict(t=t, theta=th, theta_dot=om, theta_ddot=acc, tau=tau,
                    x=self.L * np.sin(th), y=-self.L * np.cos(th))
 
    # ---- baselines the network must beat --------------------------------
    def pd_controller(self, theta_d, kp=4.0, kd=0.8):
        """Plain PD. Will NOT reach theta_d > pi/2 if tau_max < m*g*L."""
        return lambda t, th, om: kp * (theta_d - th) + kd * (0.0 - om)
 
    def gravity_compensated_pd(self, theta_d, kp=4.0, kd=0.8):
        """
        PD + feedforward cancellation of gravity and damping.
        This is the ANALYTICAL inverse-dynamics solution -- it is the honest
        baseline. Any learned controller should be compared against this,
        not against plain PD.
        """
        def ctrl(t, th, om):
            return (kp * (theta_d - th) + kd * (0.0 - om)
                    + self.m * self.g * self.L * np.sin(th) + self.b * om)
        return ctrl
 
 
# ---------------------------------------------------------------- inverse dyn
def inverse_dynamics(p: Pendulum, theta, theta_dot, theta_ddot):
    """
    Exact torque required to realise a given (theta, theta_dot, theta_ddot).
    Supervised target if you train the network on reference trajectories.
    """
    return (p.I * np.asarray(theta_ddot)
            + p.b * np.asarray(theta_dot)
            + p.m * p.g * p.L * np.sin(np.asarray(theta)))
 
 
# ---------------------------------------------------------------- demo
if __name__ == "__main__":
    p = Pendulum(m=0.5, L=0.6, b=0.10, tau_max=1.5)
    print(f"I = {p.I:.4f} kg m^2   omega_n = {p.omega_n:.3f} rad/s   "
          f"zeta = {p.zeta:.4f}")
    print(f"tau_gravity_max = {p.tau_gravity_max:.3f} Nm   "
          f"tau_max = {p.tau_max:.3f} Nm   "
          f"ratio = {p.tau_max / p.tau_gravity_max:.2f}")
 
    theta_d = np.deg2rad(60.0)
 
    for name, ctrl in [("PD", p.pd_controller(theta_d)),
                       ("PD + gravity comp", p.gravity_compensated_pd(theta_d))]:
        r = p.simulate(ctrl, s0=(0.0, 0.0), t_end=5.0)
        err = np.rad2deg(theta_d - r["theta"][-1])
        print(f"{name:20s} final theta = {np.rad2deg(r['theta'][-1]):7.2f} deg  "
              f"(target {np.rad2deg(theta_d):.1f})  err = {err:6.2f} deg  "
              f"peak |tau| = {np.abs(r['tau']).max():.3f} Nm")
 
    # consistency check: inverse dynamics must reproduce the applied torque
    r = p.simulate(p.gravity_compensated_pd(theta_d), t_end=5.0)
    tau_id = inverse_dynamics(p, r["theta"], r["theta_dot"], r["theta_ddot"])
    print(f"inverse-dynamics round-trip max abs error = "
          f"{np.abs(tau_id - r['tau']).max():.2e} Nm")