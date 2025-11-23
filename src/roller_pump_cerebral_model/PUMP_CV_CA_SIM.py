"""
roller_pump.py

Copyright (c) 2025 MGLvanVeen
Licensed under the MIT License (see LICENSE file in the project root).
https://github.com/MGLvanVeen/RollerPumpCerebroVascular

Geometry-based roller pump model coupled to a lumped-parameter
cerebrovascular model with (optional) cerebral autoregulation (CA).

OVERVIEW
--------
This module implements the model described in the thesis section
"Model Development":

1) Roller pump compartment
   - Converts a prescribed shaft speed RPM(t) into an outlet flow
     waveform Q_out(t) based purely on pump geometry, assuming:
       * incompressible fluid
       * time-invariant material properties
   - Outlet flow is modelled as:
       Q_out(t) = Q_nom(ω(t)) + ω(t) * Σ_k p(φ(t) - φ_k)
     where:
       - Q_nom(ω) is the nominal conveyor flow (non-occluded tube),
       - p(ψ) is the pulsatile displacement shape function [mL/rad],
       - φ(t) is the shaft angle [rad],
       - φ_k = 2π k / N_r are roller offsets.

2) Aortic compartment (Windkessel)
   - Proximal compliance C_a1, distal compliance C_a2,
     aortic resistance R_a, inertance L_a.
   - Governing equations match Equations (2)–(4).

3) Systemic compartment
   - Represented by a single resistance R_s.
   - No explicit systemic compliance/inertance (lumped into vascular tree).
   - Parallel arrangement with cerebral branch between Va_2 and V_v.

4) Cerebral compartment
   - Proximal resistance R_c1, distal resistance R_c2(t),
     compliance C_c, inertance L_c.
   - i_c through R_c1 ~ middle cerebral artery (relates to TCD CBFV).
   - Governing equations match (5)–(6).

5) Venous compartment
   - Compliance C_v with drainage resistance R_drain.
   - Buffer for cerebral + systemic return, output V_v ~ CVP.
   - Governing equation matches (7).

6) Cerebral autoregulation (CA)
   - Feedforward myogenic-like controller acting on R_c2(t).
   - Based on a running mean of distal aortic pressure Va_2_mean(t)
     with time constant tau_mean:
         d(Va_2_mean)/dt = (Va_2 - Va_2_mean) / tau_mean
   - Target R_c2 is:
         R_target(t) = Rc_2_set * (1 + S * (Va_2_mean - Va_2mean_set))
   - Dynamic state R_c2_dyn(t) relaxes towards R_target(t) with
     time constant tau_Rc_2, subject to bounds
     [Rc_2_set * fact_Rc_2_min, Rc_2_set * fact_Rc_2_max].

The roller-pump component is conceptually based on the MIT licenced MATLAB model by
M.P. McIntyre (peristaltic pump modelling and Simulink implementation, 2020),
translated and adapted to Python and extended for use inside an ODE
framework with time-varying RPM. 
https://github.com/michaelmcintyre/roller-pump-model-and-simulation

The cerebral model parameters and CA formulation were developed in the
context of neonatal CPB modelling (see accompanying thesis).
"""

from dataclasses import dataclass

import numpy as np
from scipy.integrate import odeint


# ======================================================================
# 1. Configuration data containers
# ======================================================================


@dataclass
class PumpGeometry:
    """
    Geometric configuration of the peristaltic roller pump.

    All dimensions are in millimetres [mm].

    Attributes
    ----------
    Nr : int
        Number of rollers on the pump head (N_r in the text).
    r_b : float
        Radius from pump centre to backplate (r_b).
    r_r : float
        Radius of a single roller (r_r).
    r_offset : float
        Orbit radius: distance from pump centre to roller centre
        (r_offset).
    d_o : float
        Outer diameter of the compliant pump tube (d_o).
    d_i : float
        Inner diameter (lumen) of the pump tube (d_i).
    """
    Nr: int = 2
    r_b: float = 75.0
    r_r: float = 15.25
    r_offset: float = 57.0
    d_o: float = 9.525
    d_i: float = 6.35


@dataclass
class Hemodynamics:
    """
    Lumped-parameter hemodynamic properties of the systemic and cerebral
    circulation.

    Units
    -----
    Resistances : [mmHg·s/mL]
    Compliances : [mL/mmHg]
    Inertances  : [mmHg·s²/mL]

    Parameters here correspond to:
      - Aortic Windkessel (Ra, Ca_1, Ca_2, La)
      - Systemic branch (Rs)
      - Cerebral branch (Rc_1, Rc_2, Cc, Lc)
      - Venous reservoir (Cv, R_drain)
    """
    Ra: float = 0.5      # aortic characteristic resistance
    Rs: float = 2.6735   # systemic peripheral resistance
    Rc_1: float = 144.0  # cerebral proximal conduit resistance (MCA)
    Rc_2: float = 60.5   # cerebral distal (pial) resistance
    R_drain: float = 0.626  # venous drainage resistance to CVP

    Ca_1: float = 0.06     # aortic compliance at Va_1
    Ca_2: float = 0.0008   # distal aortic compliance at Va_2
    Cc: float = 0.009      # cerebral arterial compliance
    Cv: float = 20.0       # venous reservoir compliance

    La: float = 0.0010     # aortic inertance Va_1–Va_2
    Lc: float = 0.50       # cerebral inertance Vc–Vv


@dataclass
class CARParams:
    """
    Parameters controlling cerebral autoregulation (CA) via dynamic
    adjustment of Rc_2(t).

    The controller maintains a running mean of distal aortic pressure
    Va_2_mean(t) and drives Rc_2_dyn(t) towards a target value:

        Rc_2_target(t) = Rc_2_set * (1 + S * (Va_2_mean - Va_2mean_set))

    with first-order dynamics:

        dRc_2_dyn/dt = (Rc_2_target - Rc_2_dyn) / tau_Rc_2

    Rc_2_dyn is bounded between:
        Rc_2_set * fact_Rc_2_min  and  Rc_2_set * fact_Rc_2_max

    This realises a myogenic-type autoregulation: for mean pressure
    above the setpoint, Rc_2 increases (vasoconstriction); below the
    setpoint, Rc_2 decreases (vasodilatation).
    """
    enable: bool = True       # master switch: enable/disable CA
    SRc_2: float = 0.15       # sensitivity S of Rc_2 to Va_2_mean error
    Rc_2_set: float = 60.0    # baseline (setpoint) Rc_2, overwritten on init
    Va_2mean_set: float = 36.5  # target mean distal aortic pressure [mmHg]
    tau_mean: float = 2.0       # time constant for Va_2_mean [s]
    tau_Rc_2: float = 2.0       # time constant for Rc_2 adjustment [s]
    fact_Rc_2_max: float = 2.0  # max factor change from Rc_2_set (upper bound)
    fact_Rc_2_min: float = 0.5  # min factor change from Rc_2_set (lower bound)


# ======================================================================
# 2. Roller pump component
# ======================================================================


class RollerPump:
    """
    Geometry-based peristaltic roller pump with precomputed pulse shape.

    This class implements the relationship between shaft angle φ(t),
    angular speed ω(t) and pump outlet flow Q_out(t).

    The outlet flow is the sum of:
      - a nominal conveyor flow Q_nom(ω), based on non-occluded tube
        volume transported over the effective radius, and
      - a pulsatile displacement term ω * Σ_k p(φ - φ_k), where p(ψ)
        is the volume displacement per radian of shaft angle during the
        engagement of a roller with the tube.

    All geometry is provided via a PumpGeometry instance.
    """

    def __init__(self, geom: PumpGeometry, n_occ: float = 1.0) -> None:
        """
        Parameters
        ----------
        geom : PumpGeometry
            Pump geometry (tube and roller dimensions, radii, etc.).
        n_occ : float, optional
            Occlusion factor in [0, 1]. This lumps together:
              - fluid compression,
              - leakage past fully occluded segments,
              - load-dependent effects.
        """
        self.g = geom
        self.n_occ = float(n_occ)

        # Precompute the engagement span θ_e and a lookup table for the
        # pulsatile displacement shape p(ψ) over [0, 2π).
        self.theta_e = self._engagement_span_rb(
            self.g.r_b,
            self.g.d_o,
            self.g.r_r,
            self.g.r_offset,
        )
        self.psi_grid, self.s_vals = self._s_out_lookup(
            self.theta_e,
            self.g.r_b,
            self.g.d_o,
            self.g.d_i,
            self.g.r_r,
            self.g.r_offset,
        )
        # Angular offsets of roller k: φ_k = 2π k / N_r
        self.phi_offsets = 2.0 * np.pi * np.arange(self.g.Nr) / self.g.Nr

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def set_occlusion(self, n_occ: float) -> None:
        """
        Set the occlusion factor in [0, 1].

        Parameters
        ----------
        n_occ : float
            New occlusion factor. Values outside [0, 1] are not clipped
            here; they are passed through literally.
        """
        self.n_occ = float(n_occ)

    def Q_out(self, phi: float, omega: float) -> float:
        """
        Compute pump outlet flow Q_out [mL/s] for given shaft angle and
        angular speed.

        Parameters
        ----------
        phi : float
            Current shaft angle φ(t) [rad].
        omega : float
            Current angular speed ω(t) [rad/s].

        Returns
        -------
        float
            Outlet flow rate Q_out(t) [mL/s].
        """
        # Relative angles for each roller k
        psi_k = phi - self.phi_offsets

        # Sum of volume-displacement per radian over all rollers
        # (s_vals is in mL/rad)
        pulse_sum = np.sum(self._s_periodic(psi_k, self.psi_grid, self.s_vals))

        # Total flow: nominal conveyor flow + pulsatile contribution
        return self._Q_nominal_cont(omega) + omega * pulse_sum

    # ------------------------------------------------------------------
    # Nominal (conveyor) flow using geometry + occlusion
    # ------------------------------------------------------------------

    def _Q_nominal_cont(self, omega: float) -> float:
        """
        Compute the nominal (non-pulsatile) conveyor flow Q_nom [mL/s].

        This is the baseline flow given by the tube lumen area, effective
        radius and angular speed ω(t), scaled by the occlusion factor.

        Q_nom(t) = π (d_i / 2)^2 * ω(t) * (r_b - d_o / 2) * n_occ / 1000

        All geometry is taken from the PumpGeometry instance; 1000
        converts [mm^3/s] to [mL/s].
        """
        # Tube cross-sectional area [mm^2]
        a = np.pi * (self.g.d_i / 2.0) ** 2
        # Effective swept distance at radius (r_b - d_o/2); multiplied
        # by occlusion factor n_occ and converted to mL/s.
        return (
            omega
            * (self.g.r_b - self.g.d_o / 2.0)
            * a
            * self.n_occ
            / 1000.0
        )

    # ------------------------------------------------------------------
    # Geometry / pulse shape helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _engagement_span_rb(
        r_b: float,
        d_o: float,
        r_r: float,
        r_offset: float,
    ) -> float:
        """
        Compute engagement span θ_e based on pump geometry.

        Engagement span is the range of shaft angle over which the roller
        compresses/releases the tube, estimated from the distance
        between backplate, roller and tube.

        Formula:
            θ_e = π/2 - arcsin((r_b - d_o - r_r) / r_offset)
        """
        a = r_b - d_o - r_r
        b = r_offset
        alpha = np.arcsin(np.clip(a / b, -1.0, 1.0))
        return np.pi / 2.0 - alpha

    @staticmethod
    def _L_contact(
        psi: float,
        r_b: float,
        d_o: float,
        r_r: float,
        r_offset: float,
    ) -> float:
        """
        Compute the length of the contact arc between roller and tube.

        This is used to determine the extent of indentation over which
        the tube lumen is compressed by the roller.
        """
        # delta: distance between casing radius and (roller + orbit)
        delta = r_b - (r_r + r_offset * np.cos(psi))
        d1 = d_o - delta

        # Compute angle beta describing the intersection geometry
        term = np.clip((r_r - d1) / r_r, -1.0, 1.0)
        beta = np.arccos(term)
        return r_r * np.sin(beta)

    @staticmethod
    def _c_indent(x: np.ndarray, r_r: float) -> np.ndarray:
        """
        Roller indentation profile c(x) for distance x along the contact
        arc. The roller is approximated as a circle of radius r_r.

        Parameters
        ----------
        x : array-like
            Positions along the contact path [mm].
        r_r : float
            Roller radius [mm].

        Returns
        -------
        np.ndarray
            Indentation depth [mm] at each position x.
        """
        return r_r - np.sqrt(np.maximum(r_r * r_r - x * x, 0.0))

    @staticmethod
    def _lumen_area_slice(h: np.ndarray, d_i: float) -> np.ndarray:
        """
        Approximate the remaining lumen area of a tube slice given a
        local wall indentation height h.

        The cross-section is approximated as an ellipse deformation of
        an initially circular lumen.

        Parameters
        ----------
        h : array-like
            Wall indentation heights [mm] (clipped between 0 and d_i).
        d_i : float
            Tube inner diameter [mm].

        Returns
        -------
        np.ndarray
            Lumen area for each indentation value [mm^2].
        """
        r_ID = d_i / 2.0
        circumference = 2.0 * np.pi * r_ID

        # "Short" radius shrinks with indentation
        r_short = r_ID - h / 2.0
        # "Long" radius from approximate perimeter conservation
        r_long = np.sqrt(
            np.maximum(
                2.0 * (circumference / (2.0 * np.pi)) ** 2 - r_short ** 2,
                0.0,
            )
        )

        # Lumen area = full circle - compressed ellipse
        return np.pi * r_ID ** 2 - np.pi * r_short * r_long

    def _V_of_psi(
        self,
        psi: float,
        r_b: float,
        d_o: float,
        d_i: float,
        r_r: float,
        r_offset: float,
        nx: int = 200,
    ) -> float:
        """
        Compute displaced volume V(ψ) [mL] under a single roller as a
        function of engagement angle ψ.

        The roller contact length is discretised into nx slices, tube
        indentation is computed via `_c_indent`, and the lumen area is
        integrated numerically.

        Parameters
        ----------
        psi : float
            Engagement angle (relative roller angle) [rad].
        r_b, d_o, d_i, r_r, r_offset :
            Pump geometry parameters [mm].
        nx : int, optional
            Number of discretisation points along the contact length.

        Returns
        -------
        float
            Displaced volume [mL] due to compression at angle ψ.
        """
        L = self._L_contact(psi, r_b, d_o, r_r, r_offset)
        if L <= 0.0:
            return 0.0

        # Discretise contact length [mm]
        x = np.linspace(0.0, L, nx)

        # Distance between roller and tube outer surface
        delta = r_b - (r_r + r_offset * np.cos(psi))
        d1 = d_o - delta

        # Indentation height, clipped to [0, d_i]
        h = np.clip(d1 - self._c_indent(x, r_r), 0.0, d_i)

        # Lumen area profile [mm^2]
        A = self._lumen_area_slice(h, d_i)

        # Volume under the roller: integrate area symmetrically
        V_mm3 = 2.0 * np.trapz(A, x)

        # Convert [mm^3] → [mL]
        return V_mm3 / 1000.0

    def _dV_dpsi_eng(
        self,
        psi: np.ndarray,
        r_b: float,
        d_o: float,
        d_i: float,
        r_r: float,
        r_offset: float,
        dpsi: float = 1e-4,
    ) -> np.ndarray:
        """
        Numerical derivative dV/dψ within the engagement span.

        Central difference approximation is used on a small step dpsi.

        Parameters
        ----------
        psi : array-like
            Engagement angles [rad].
        r_b, d_o, d_i, r_r, r_offset :
            Pump geometry parameters [mm].
        dpsi : float, optional
            Step size [rad] for finite differencing.

        Returns
        -------
        np.ndarray
            dV/dψ [mL/rad] evaluated at each psi.
        """
        pp = np.minimum(psi + dpsi, np.pi)
        pm = np.maximum(psi - dpsi, 0.0)

        Vp = np.array(
            [
                self._V_of_psi(x, r_b, d_o, d_i, r_r, r_offset)
                for x in np.atleast_1d(pp)
            ]
        )
        Vm = np.array(
            [
                self._V_of_psi(x, r_b, d_o, d_i, r_r, r_offset)
                for x in np.atleast_1d(pm)
            ]
        )
        return (Vp - Vm) / (2.0 * dpsi)

    def _s_out_lookup(
        self,
        theta_e: float,
        r_b: float,
        d_o: float,
        d_i: float,
        r_r: float,
        r_offset: float,
        n_psi: int = 2048,
    ):
        """
        Precompute the pulsatile displacement derivative s(ψ) = dV/dψ
        over [0, 2π) for lookup during simulation.

        Only [0, θ_e] is non-zero (roller engaged); outside that range
        the displacement is zero.

        Parameters
        ----------
        theta_e : float
            Engagement span [rad].
        r_b, d_o, d_i, r_r, r_offset :
            Pump geometry parameters [mm].
        n_psi : int, optional
            Number of grid points over [0, 2π).

        Returns
        -------
        psi_grid : np.ndarray
            Uniform grid over [0, 2π) [rad].
        s_vals : np.ndarray
            Values of dV/dψ [mL/rad] on this grid.
        """
        psi_grid = np.linspace(0.0, 2.0 * np.pi, n_psi, endpoint=False)
        s_vals = np.zeros_like(psi_grid)

        # Only angles ψ ≤ θ_e contribute; the rest remain zero
        mask = psi_grid <= theta_e
        psi_rev = theta_e - psi_grid[mask]

        s_vals[mask] = self._dV_dpsi_eng(
            psi_rev,
            r_b,
            d_o,
            d_i,
            r_r,
            r_offset,
        )
        return psi_grid, s_vals

    @staticmethod
    def _s_periodic(
        psi: np.ndarray,
        psi_grid: np.ndarray,
        s_vals: np.ndarray,
    ) -> np.ndarray:
        """
        Evaluate the periodic pulse shape s(ψ) at arbitrary ψ using
        linear interpolation on [0, 2π).

        Parameters
        ----------
        psi : array-like
            Angles at which s(ψ) should be evaluated [rad].
        psi_grid : np.ndarray
            Grid used for precomputation [rad].
        s_vals : np.ndarray
            dV/dψ values on `psi_grid` [mL/rad].

        Returns
        -------
        np.ndarray
            Interpolated s(ψ) [mL/rad] at the given angles.
        """
        n = len(psi_grid)
        L = 2.0 * np.pi

        # Map ψ into [0, 2π), then into grid indices
        x = (np.mod(psi, L)) * (n / L)
        i0 = np.floor(x).astype(int) % n
        i1 = (i0 + 1) % n
        w = x - np.floor(x)

        return (1.0 - w) * s_vals[i0] + w * s_vals[i1]


# ======================================================================
# 3. Utility: RPM→ω(t) interpolant
# ======================================================================


def _make_omega_of_t(t_cmd, rpm_cmd, mode: str = "linear"):
    """
    Build a function omega_of_t(t) [rad/s] from piecewise RPM data.

    Parameters
    ----------
    t_cmd : array-like
        Strictly increasing time points [s].
    rpm_cmd : array-like
        RPM values [revolutions per minute] at each time in t_cmd.
    mode : {"linear", "zoh"}, optional
        Interpolation mode:
        - "linear": piecewise linear interpolation between knots.
        - "zoh":    zero-order hold (step-wise).

    Returns
    -------
    omega_of_t : callable
        Function t -> ω(t) in [rad/s].

    Raises
    ------
    ValueError
        If t_cmd is not strictly increasing or mode is unsupported.
    """
    t_cmd = np.asarray(t_cmd, dtype=float)
    rpm_cmd = np.asarray(rpm_cmd, dtype=float)

    if np.any(np.isnan(t_cmd)) or np.any(np.isnan(rpm_cmd)):
        raise ValueError("NaNs in RPM input.")

    if np.any(np.diff(t_cmd) <= 0.0):
        raise ValueError("t_cmd must be strictly increasing.")

    # Disallow negative RPM (no reverse rotation by default)
    rpm_cmd = np.maximum(rpm_cmd, 0.0)

    if mode == "linear":
        def rpm_of_t_scalar(t):
            return float(np.interp(t, t_cmd, rpm_cmd))
    elif mode == "zoh":
        def rpm_of_t_scalar(t):
            idx = np.searchsorted(t_cmd, t, side="right") - 1
            idx = int(np.clip(idx, 0, len(t_cmd) - 1))
            return float(rpm_cmd[idx])
    else:
        raise ValueError("mode must be 'linear' or 'zoh'.")

    # Convert RPM → rad/s via ω = 2π/60 * RPM
    return lambda t: rpm_of_t_scalar(t) * 2.0 * np.pi / 60.0


def _quantize_down(t, dt: float):
    """
    Quantise time t down to the previous multiple of dt.

    This is used to evaluate schedules (RPM, Rs, occlusion) at
    consistent grid points, avoiding stage order artifacts in ODE
    integrators.

    Parameters
    ----------
    t : float or array-like
        Time(s) to quantise.
    dt : float
        Time step [s].

    Returns
    -------
    float or np.ndarray
        Quantised time(s).
    """
    tt = np.asarray(t, dtype=float)
    return np.floor(tt / dt) * dt


# ======================================================================
# 4. Top-level coupled model (pump + circulation + CA)
# ======================================================================


class RollerPumpCerebralModel:
    """
    Full coupled model: geometry-based roller pump + Windkessel
    cerebrovascular model + cerebral autoregulation (optional).

    State vector x has length 9:

        x = [
            Va_1,      # proximal aortic pressure [mmHg]
            Va_2,      # distal aortic pressure [mmHg]
            Vc,        # cerebral arterial pressure [mmHg]
            Vv,        # venous pressure (CVP) [mmHg]
            iLa,       # aortic inertance flow [mL/s]
            iLc,       # cerebral inertance flow [mL/s]
            phi,       # pump shaft angle [rad]
            Va_2_mean, # running mean of Va_2 [mmHg] (for CA)
            Rc_2_dyn,  # dynamic distal cerebrovascular resistance [mmHg·s/mL]
        ]

    Usage
    -----
    - Instantiate with a time step dt:
        model = RollerPumpCerebralModel(dt=1e-3)

    - Configure geometry & hemodynamics via:
        model.geom, model.hemo, model.car, or via `load_preset(...)`.

    - Define pump speed via:
        model.set_rpm_schedule(t_cmd, rpm_cmd, mode="linear" or "zoh")

    - Optionally make Rs or occlusion time-varying via:
        Rs_of_t(t), n_occ_of_t(t) passed into `simulate(..., Rs_of_t=...)`.

    - Run simulations with:
        t, sol, out = model.simulate(T=..., y0=..., car_enable=True/False)

    Outputs
    -------
    t   : time vector [s]
    sol : state trajectories (N x 9)
    out : dict with derived outputs:
          - "Q_out"    : pump outlet flow [mL/s]
          - "ic"       : cerebral inflow [mL/s]
          - "Rc_2_eff" : dynamic R_c2 (reported state) [mmHg·s/mL]
          - "Rs"       : effective systemic resistance series [mmHg·s/mL]
    """

    def __init__(
        self,
        dt: float = 1e-3,
        use_rpm_data: bool = False,
        t_cmd=None,
        rpm_cmd=None,
        rpm_mode: str = "linear",
    ) -> None:
        """
        Parameters
        ----------
        dt : float, optional
            Fixed integration time step [s] used to build the time grid
            in `simulate`.
        use_rpm_data : bool, optional
            If True and (t_cmd, rpm_cmd) are provided, build omega_of_t
            from those arrays; otherwise use the built-in analytic
            example schedule.
        t_cmd : array-like or None
            Time knots [s] for RPM data (if use_rpm_data=True).
        rpm_cmd : array-like or None
            RPM values at t_cmd (if use_rpm_data=True).
        rpm_mode : {"linear", "zoh"}, optional
            Interpolation mode for RPM data.
        """
        self.dt = float(dt)

        # If True, schedules (RPM, Rs, occlusion) are evaluated on a
        # dt-quantised time grid to avoid stage order issues.
        self.stage_safe_sampling = True

        # Configuration blocks
        self.geom = PumpGeometry()
        self.hemo = Hemodynamics()
        # Initialise CA with Rc_2_set equal to current hemo.Rc_2
        self.car = CARParams(Rc_2_set=self.hemo.Rc_2)

        # Pump component
        self.pump = RollerPump(self.geom, n_occ=1.0)

        # RPM schedule
        if use_rpm_data and (t_cmd is not None and rpm_cmd is not None):
            self.omega_of_t = _make_omega_of_t(t_cmd, rpm_cmd, mode=rpm_mode)
        else:
            # Analytic fallback: simple step/ramp example
            self.omega_of_t = (
                lambda t: self._rpm_of_t_analytic(t) * 2.0 * np.pi / 60.0
            )

    # ------------------------------------------------------------------
    # Public setters
    # ------------------------------------------------------------------

    def set_occlusion_factor(self, n_occ: float) -> None:
        """
        Set the occlusion factor for the pump.

        Parameters
        ----------
        n_occ : float
            Occlusion factor in [0, 1]. Passed directly to the RollerPump.
        """
        self.pump.set_occlusion(n_occ)

    def set_Rs(self, Rs_value: float) -> None:
        """
        Set the (constant) systemic resistance R_s.

        Parameters
        ----------
        Rs_value : float
            Systemic resistance [mmHg·s/mL].
        """
        self.hemo.Rs = float(Rs_value)

    def set_rpm_schedule(self, t_cmd, rpm_cmd, mode: str = "linear") -> None:
        """
        Define a time-varying RPM schedule for the pump.

        Parameters
        ----------
        t_cmd : array-like
            Strictly increasing time knots [s].
        rpm_cmd : array-like
            RPM values at each time in t_cmd.
        mode : {"linear", "zoh"}, optional
            Interpolation mode for RPM data.
        """
        self.omega_of_t = _make_omega_of_t(t_cmd, rpm_cmd, mode)

    def set_CAR(self, enable: bool = True, **overrides) -> None:
        """
        Enable/disable cerebral autoregulation and optionally override
        specific CAR parameters.

        Parameters
        ----------
        enable : bool, optional
            If True, CA is active unless overridden by `car_enable` in
            `simulate` or `rhs`.
        **overrides :
            Any attributes of `CARParams` can be passed here, e.g.:
                set_CAR(SRc_2=0.10, Va_2mean_set=38.0)
        """
        self.car.enable = bool(enable)
        for k, v in overrides.items():
            if hasattr(self.car, k):
                setattr(self.car, k, v)

    # ------------------------------------------------------------------
    # Preset configuration
    # ------------------------------------------------------------------

    def load_preset(self, name: str):
        """
        Apply a predefined set of parameter values by name.

        Currently supported presets
        ---------------------------
        - "average_final" :
            Hemodynamics and CA parameters tuned to represent an
            "average neonate undergoing CPB" as described in the thesis.

        Parameters
        ----------
        name : str
            Name of the preset configuration to load.

        Returns
        -------
        self : RollerPumpCerebralModel
            The model instance (to allow method chaining).
        """
        if name == "average_final":
            # Hemodynamics settings
            self.hemo.Ra = 0.5
            self.hemo.Rs = 2.6735
            self.hemo.Rc_1 = 144.0
            self.hemo.Rc_2 = 60.0
            self.hemo.R_drain = 0.626

            self.hemo.Ca_1 = 0.06
            self.hemo.Ca_2 = 0.0008
            self.hemo.Cc = 0.009
            self.hemo.Cv = 20.0
            self.hemo.La = 0.0010
            self.hemo.Lc = 0.50

            # Cerebral autoregulation settings
            self.car.Rc_2_set = 60.0
            self.car.Va_2mean_set = 36.5
            self.car.SRc_2 = 0.15
            self.car.tau_mean = 2.0
            self.car.tau_Rc_2 = 2.0
            self.car.fact_Rc_2_max = 2.0
            self.car.fact_Rc_2_min = 0.5

            return self

        else:
            raise ValueError(
                f"Unknown preset name: {name}. Available presets: 'average_final'"
            )

    # ------------------------------------------------------------------
    # ODE right-hand side
    # ------------------------------------------------------------------

    def rhs(
        self,
        t: float,
        x,
        *,
        omega_scale: float = 1.0,
        car_enable=None,
        n_occ_of_t=None,
        Rs_of_t=None,
    ):
        """
        Compute the time derivative dx/dt of the state vector.

        State vector
        ------------
        x = [
            Va_1,      # proximal aortic pressure [mmHg]
            Va_2,      # distal aortic pressure [mmHg]
            Vc,        # cerebral arterial pressure [mmHg]
            Vv,        # venous pressure [mmHg]
            iLa,       # aortic inertance flow [mL/s]
            iLc,       # cerebral inertance flow [mL/s]
            phi,       # pump shaft angle [rad]
            Va_2_mean, # running mean of Va_2 [mmHg]
            Rc_2_dyn,  # dynamic distal cerebrovascular resistance [mmHg·s/mL]
        ]

        Optional time-varying hooks
        ---------------------------
        n_occ_of_t : callable or None
            If callable, n_occ_of_t(t) -> float sets the occlusion factor
            at quantised time tq.
        Rs_of_t : callable or None
            If callable, Rs_of_t(t) -> float provides a time-varying
            systemic resistance R_s(t).

        Other parameters
        ----------------
        omega_scale : float, optional
            Multiplicative scaling of pump angular speed. Default 1.0.
        car_enable : bool or None, optional
            If True/False, forces CA on/off, overriding self.car.enable.
            If None, self.car.enable is used.

        Returns
        -------
        dxdt : np.ndarray
            Time derivative of the state vector (length 9).
        """
        Va_1, Va_2, Vc, Vv, iLa, iLc, phi, Va_2_mean, Rc_2_dyn = x

        # ---------- Stage-safe evaluation time for schedules ----------
        tq = _quantize_down(t, self.dt) if self.stage_safe_sampling else t

        # Update scheduled parameters (occlusion, Rs)
        if callable(n_occ_of_t):
            self.pump.set_occlusion(n_occ_of_t(tq))

        Rs_eff = float(Rs_of_t(tq)) if callable(Rs_of_t) else self.hemo.Rs

        # ---------- Pump angular speed and inflow ----------
        omg = float(omega_scale) * self.omega_of_t(tq)
        i_pump = self.pump.Q_out(phi, omg)  # [mL/s]

        # ---------- Cerebral autoregulation (CA) ----------
        if car_enable is None:
            car_on = self.car.enable
        else:
            car_on = bool(car_enable)

        if car_on:
            # Mean Va_2 dynamics (low-pass filter)
            dVa_2_mean_dt = (Va_2 - Va_2_mean) / self.car.tau_mean

            # Target Rc_2 based on deviation from Va_2mean_set
            Rc_2_target = self.car.Rc_2_set * (
                1.0 + self.car.SRc_2 * (Va_2_mean - self.car.Va_2mean_set)
            )
            Rc_2_min = self.car.Rc_2_set * self.car.fact_Rc_2_min
            Rc_2_max = self.car.Rc_2_set * self.car.fact_Rc_2_max

            # First-order relaxation of Rc_2_dyn towards Rc_2_target
            dRc_2_dt = (Rc_2_target - Rc_2_dyn) / self.car.tau_Rc_2

            # Enforce bounds in a step-size invariant way:
            # - if Rc_2_dyn is at lower bound and decreasing, freeze
            # - if Rc_2_dyn is at upper bound and increasing, freeze
            if Rc_2_dyn <= Rc_2_min and dRc_2_dt < 0.0:
                dRc_2_dt = 0.0
            if Rc_2_dyn >= Rc_2_max and dRc_2_dt > 0.0:
                dRc_2_dt = 0.0

            # Effective Rc_2 used in flow equations is clipped to bounds,
            # while the state Rc_2_dyn remains continuous.
            Rc_2_eff = np.clip(Rc_2_dyn, Rc_2_min, Rc_2_max)

        else:
            # CA off: freeze mean and Rc_2_dyn
            dVa_2_mean_dt = 0.0
            dRc_2_dt = 0.0
            Rc_2_eff = self.hemo.Rc_2

        # ---------- Hemodynamics (Windkessel equations) ----------
        p = self.hemo  # shorthand

        # Compliances:
        #   (2) Aortic proximal node Va_1
        dVa_1_dt = (1.0 / p.Ca_1) * (i_pump - iLa)

        #   (3) Aortic distal node Va_2
        dVa_2_dt = (
            1.0
            / p.Ca_2
            * (
                -(1.0 / p.Rc_1 + 1.0 / Rs_eff) * Va_2
                + Vc / p.Rc_1
                + Vv / Rs_eff
                + iLa
            )
        )

        #   (5) Cerebral arterial pressure Vc
        dVc_dt = (1.0 / p.Cc) * ((Va_2 - Vc) / p.Rc_1 - iLc)

        #   (7) Venous pressure Vv (venous reservoir + drainage)
        dVv_dt = (
            1.0
            / p.Cv
            * (
                Va_2 / Rs_eff
                - (1.0 / Rs_eff + 1.0 / p.R_drain) * Vv
                + iLc
            )
        )

        # Inertances:
        #   (4) Aortic flow iLa
        diLa_dt = (1.0 / p.La) * (Va_1 - Va_2 - p.Ra * iLa)

        #   (6) Cerebral flow iLc (through L_c and Rc_2)
        diLc_dt = (1.0 / p.Lc) * (Vc - Vv - Rc_2_eff * iLc)

        # Pump shaft angle:
        #   (1) dφ/dt = ω(t)
        dphi_dt = omg

        return np.array(
            [
                dVa_1_dt,
                dVa_2_dt,
                dVc_dt,
                dVv_dt,
                diLa_dt,
                diLc_dt,
                dphi_dt,
                dVa_2_mean_dt,
                dRc_2_dt,
            ],
            dtype=float,
        )

    # ------------------------------------------------------------------
    # Simulation wrapper
    # ------------------------------------------------------------------

    def simulate(self, T: float = 10.0, y0=None, **rhs_kwargs):
        """
        Integrate the model over [0, T) with fixed step self.dt.

        Parameters
        ----------
        T : float, optional
            Total simulation time [s].
        y0 : array-like or None, optional
            Initial state vector x0 (length 9). If None, a default
            plausible initial state is used.
        **rhs_kwargs :
            Additional keyword arguments passed to `rhs`, allowing
            time-varying inputs such as:
              - n_occ_of_t=...
              - Rs_of_t=...
              - omega_scale=...
              - car_enable=...

        Returns
        -------
        t : np.ndarray
            Time vector [s] from 0 to T (exclusive) with step self.dt.
        sol : np.ndarray
            State trajectories with shape (N, 9).
        out : dict
            Output dictionary with keys:
              - "Q_out"     : pump outlet flow [mL/s]
              - "ic"        : cerebral inflow [mL/s]
              - "Rc_2_eff"  : dynamic Rc_2 (state) [mmHg·s/mL]
              - "Rs"        : effective systemic resistance [mmHg·s/mL]
        """
        if y0 is None:
            # Default initial condition aimed to be close to steady-state
            y0 = np.array(
                [
                    35.0,           # Va_1 [mmHg]
                    16.0,           # Va_2 [mmHg]
                    8.5,            # Vc   [mmHg]
                    4.0,            # Vv   [mmHg]
                    0.0,            # iLa  [mL/s]
                    0.0,            # iLc  [mL/s]
                    0.0,            # phi  [rad]
                    35.0,           # Va_2_mean [mmHg]
                    self.hemo.Rc_2, # Rc_2_dyn [mmHg·s/mL]
                ],
                dtype=float,
            )

        t = np.arange(0.0, float(T), self.dt)

        def _rhs(y, tt):
            return self.rhs(tt, y, **rhs_kwargs)

        # Integrate using scipy's odeint (fixed t grid)
        sol = odeint(_rhs, y0, t)

        # Unpack and compute derived outputs
        Va_1, Va_2, Vc = sol[:, 0], sol[:, 1], sol[:, 2]
        phi = sol[:, 6]
        Rc_2d = sol[:, 8]

        # Evaluate pump flow and Rs on quantised time grid
        tq_series = (
            _quantize_down(t, self.dt) if self.stage_safe_sampling else t
        )

        Q_out = np.array(
            [
                self.pump.Q_out(phi_i, self.omega_of_t(tqi))
                for phi_i, tqi in zip(phi, tq_series)
            ]
        )

        # Cerebral inflow through R_c1 (MCA):
        #   i_c = (Va_2 - Vc) / R_c1  [mL/s]
        ic = (Va_2 - Vc) / self.hemo.Rc_1

        # Report Rc_2_dyn as Rc_2_eff (state value; clipping was only
        # applied inside rhs for the flows).
        Rc_2_eff = Rc_2d

        # Effective Rs series for reporting
        Rs_func = rhs_kwargs.get("Rs_of_t", None)
        if callable(Rs_func):
            Rs_series = np.array(
                [float(Rs_func(tqi)) for tqi in tq_series]
            )
        else:
            Rs_series = np.full_like(t, fill_value=self.hemo.Rs, dtype=float)

        return t, sol, {
            "Q_out": Q_out,
            "ic": ic,
            "Rc_2_eff": Rc_2_eff,
            "Rs": Rs_series,
        }

    # ------------------------------------------------------------------
    # Example analytic RPM fallback (used if no RPM data is provided)
    # ------------------------------------------------------------------

    @staticmethod
    def _rpm_of_t_analytic(tt: float) -> float:
        """
        Example analytic RPM schedule used as a fallback.

        Behaviour
        ---------
        - 0–4 s  : 42 RPM (baseline)
        - 4–6 s  : linear ramp from 42 to 84 RPM
        - > 6 s  : 84 RPM (high steady level)

        This is mainly intended for quick testing and should be
        overridden with measured or synthetic RPM data in most use
        cases via `set_rpm_schedule(...)`.
        """
        if tt < 4.0:
            return 42.0
        elif tt < 6.0:
            return 42.0 + (84.0 - 42.0) * (tt - 4.0) / 2.0
        else:
            return 84.0


__all__ = [
    "PumpGeometry",
    "Hemodynamics",
    "CARParams",
    "RollerPump",
    "RollerPumpCerebralModel",
]
