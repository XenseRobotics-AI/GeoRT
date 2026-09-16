"""Experimental Wuji little-finger disambiguation with a Cartesian error budget.

Angles are radians in config joint order; positions/directions are in the URDF
base frame, the same frame used by GeoRT. No simulator state is changed.
"""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


from geort.kinematics import FingerKinematics


@dataclass(frozen=True)
class PostureResult:
    qpos: np.ndarray
    status: str
    tip_shift_m: float
    before_violation_rad: float
    after_violation_rad: float
    solves: int
    tip_budget_satisfied: bool = True
    mcp_guard_satisfied: bool = True


class WujiPostureCorrector:
    """Select a lower-backbend solution without locking PIP/DIP together.

    A successful solve is independently checked against URDF limits, the original
    tip position and the original one-sided violation. Failed solves fall back to
    the unmodified command in independent-frame mode. With continuity enabled,
    recovery stays inside the joint-step bound and reports task-space relaxation.
    This is a kinematic filter, not collision avoidance.
    """

    def __init__(self, config, tip_budget_mm=2.0, extension_tolerance_deg=5.0,
                 direction_weight=0.05, starts=3, max_iterations=60,
                 mcp_extension_tolerance_deg=15.0, max_joint_step_deg=8.0):
        if config.get('name') != 'wuji_hand2_beta1_right':
            raise ValueError('Posture correction currently supports only Wuji Hand2 Beta1 right')
        for name, value in [('tip_budget_mm', tip_budget_mm),
                            ('extension_tolerance_deg', extension_tolerance_deg),
                            ('direction_weight', direction_weight),
                            ('max_joint_step_deg', max_joint_step_deg),
                            ('mcp_extension_tolerance_deg', mcp_extension_tolerance_deg)]:
            if not np.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        if tip_budget_mm <= 0 or extension_tolerance_deg >= 60 or mcp_extension_tolerance_deg > 60:
            raise ValueError('Require positive tip budget and extension tolerance below 60 degrees')
        if starts not in (1, 3) or not isinstance(max_iterations, int) or max_iterations < 1:
            raise ValueError('starts must be 1 or 3; max_iterations must be positive')
        self.kinematics = FingerKinematics(config)
        self.n_dof = len(config['joint_order'])
        self.budget = tip_budget_mm / 1000
        self.floor = -np.deg2rad(extension_tolerance_deg)
        self.mcp_floor = -np.deg2rad(mcp_extension_tolerance_deg)
        self.mcp_id = self.kinematics.names.index('r_pinky_mcp_flex')
        self.direction_weight = direction_weight
        self.starts = starts
        self.max_iterations = max_iterations
        self.max_step = np.deg2rad(max_joint_step_deg)
        self.bend_ids = np.array([self.kinematics.names.index('r_pinky_pip'),
                                  self.kinematics.names.index('r_pinky_dip')])
        self.previous = None

    def reset(self):
        self.previous = None

    def violation(self, q):
        return np.maximum(self.floor - q[self.bend_ids], 0)

    def correct(self, qpos, human_points=None):
        """Correct one frame, with an optional hard per-call joint-step bound.

        Priority: physical limits and continuity, then Cartesian budget, then
        posture. When these conflict, report Cartesian/MCP relaxation explicitly.
        First frame has no prior command; call reset() at a sequence boundary.
        """
        previous = None if self.previous is None else self.previous.copy()
        result = self._correct_pose(qpos, human_points)
        kin = self.kinematics
        desired = result.qpos[kin.indices]
        if not self.max_step or previous is None or np.max(np.abs(desired - previous)) <= self.max_step:
            return result
        original = np.asarray(qpos, dtype=float)[kin.indices]
        target = kin.forward(original)[0]
        lower = np.maximum(kin.lower, previous - self.max_step)
        upper = np.minimum(kin.upper, previous + self.max_step)
        mcp_floor = min(original[self.mcp_id], self.mcp_floor)
        guarded_lower = lower.copy()
        if mcp_floor <= upper[self.mcp_id]:
            guarded_lower[self.mcp_id] = max(lower[self.mcp_id], mcp_floor)

        def cost(q):
            delta = q - desired
            return float(delta @ delta), 2 * delta

        def tracking(q):
            tip, _, jac, _ = kin.forward(q)
            delta = (tip - target) / self.budget
            return float(delta @ delta), 2 * jac.T @ delta / self.budget

        def constraint(q):
            value, grad = tracking(q)
            return 0.999999 - value, -grad

        def valid(q):
            return (q.shape == original.shape and np.isfinite(q).all()
                    and np.all(q >= guarded_lower) and np.all(q <= upper))

        seed = np.clip(previous, guarded_lower, upper)
        solved = minimize(cost, seed, jac=True, method='SLSQP',
                          bounds=list(zip(guarded_lower, upper)),
                          constraints={'type': 'ineq', 'fun': lambda q: constraint(q)[0],
                                       'jac': lambda q: constraint(q)[1]},
                          options={'maxiter': self.max_iterations, 'ftol': 1e-9})
        candidate = np.asarray(solved.x, dtype=float)
        solves = result.solves + 1
        if solved.success and valid(candidate) and tracking(candidate)[0] <= 1:
            chosen = candidate
        else:
            # Never fall back to an unbounded raw command after a failed solve.
            # Best available tracking inside the continuity box; retaining the
            # clipped previous command remains a valid conservative fallback.
            candidates = [seed, np.clip(desired, guarded_lower, upper)]
            recovered = minimize(tracking, seed, jac=True, method='SLSQP',
                                 bounds=list(zip(guarded_lower, upper)),
                                 options={'maxiter': self.max_iterations, 'ftol': 1e-9})
            solves += 1
            candidate = np.asarray(recovered.x, dtype=float)
            if valid(candidate):
                candidates.append(candidate)
            chosen = min(candidates, key=lambda q: tracking(q)[0]).copy()
        output = result.qpos.copy()
        output[kin.indices] = chosen
        self.previous = chosen.copy()
        shift = float(np.linalg.norm(kin.forward(chosen)[0] - target))
        in_budget = shift <= self.budget
        mcp_ok = chosen[self.mcp_id] >= mcp_floor - 1e-10
        return PostureResult(output, 'continuity_limited' if in_budget else 'tip_budget_relaxed',
                             shift, result.before_violation_rad,
                             float(np.linalg.norm(self.violation(chosen))), solves,
                             in_budget, mcp_ok)

    def _correct_pose(self, qpos, human_points=None):
        qpos = np.asarray(qpos, dtype=float)
        if qpos.shape != (self.n_dof,) or not np.isfinite(qpos).all():
            raise ValueError('Expected one finite angle per configured joint')
        kin = self.kinematics
        original = qpos[kin.indices].copy()
        if np.any(original < kin.lower - 1e-7) or np.any(original > kin.upper + 1e-7):
            raise ValueError('Little-finger input exceeds physical joint limits')
        human_direction = None
        if human_points is not None:
            human_points = np.asarray(human_points, dtype=float)
            if human_points.shape != (21, 3) or not np.isfinite(human_points).all():
                raise ValueError('Expected finite human points with shape (21, 3)')
            vector = human_points[20] - human_points[19]
            if np.linalg.norm(vector) > 1e-8:
                human_direction = vector / np.linalg.norm(vector)
        if self.direction_weight and human_points is None:
            raise ValueError('Human points required for nonzero direction weight')
        target = kin.forward(original)[0]
        before = np.linalg.norm(self.violation(original))
        if before <= 1e-10:
            self.previous = original.copy()
            return PostureResult(qpos.copy(), 'unchanged', 0., before, before, 0)
        previous = original if self.previous is None else self.previous
        # Do not transfer distal backbend into newly excessive MCP extension.
        # Existing wider MCP extension remains feasible, but cannot worsen.
        lower = kin.lower.copy()
        lower[self.mcp_id] = max(lower[self.mcp_id], min(original[self.mcp_id], self.mcp_floor))

        cached_q, cached_fk = None, None

        def fk(q):
            nonlocal cached_q, cached_fk
            if cached_q is None or not np.array_equal(q, cached_q):
                cached_q, cached_fk = q.copy(), kin.forward(q)
            return cached_fk

        def objective(q):
            violation = self.violation(q)
            value = violation @ violation
            grad = np.zeros_like(q)
            grad[self.bend_ids] = -2 * violation
            if human_direction is not None and self.direction_weight:
                _, direction, _, jac = fk(q)
                delta = direction - human_direction
                value += self.direction_weight * (delta @ delta)
                grad += 2 * self.direction_weight * jac.T @ delta
            # Only tie-break among comparable postures, not a fixed human angle mapping.
            value += 1e-4 * np.sum((q - original) ** 2) + 1e-4 * np.sum((q - previous) ** 2)
            grad += 2e-4 * (q - original) + 2e-4 * (q - previous)
            return value, grad

        def constraint(q):
            tip, _, jac, _ = fk(q)
            delta = (tip - target) / self.budget
            return 0.999999 - delta @ delta, -2 * delta @ jac / self.budget

        seeds = [original]
        if self.starts == 3:
            neutral = original.copy()
            neutral[self.bend_ids] = .2
            neutral[0] = .2
            alternate = original.copy()
            alternate[self.bend_ids] = .8
            seeds += [neutral, previous.copy() if self.previous is not None else alternate]
        best, best_score = original, objective(original)[0]
        accepted = False
        for seed in seeds:
            result = minimize(objective, np.clip(seed, lower, kin.upper), jac=True,
                              bounds=list(zip(lower, kin.upper)), method='SLSQP',
                              constraints={'type': 'ineq', 'fun': lambda q: constraint(q)[0],
                                           'jac': lambda q: constraint(q)[1]},
                              options={'maxiter': self.max_iterations, 'ftol': 1e-9})
            q = np.asarray(result.x, dtype=float)
            if (not result.success or not np.isfinite(q).all()
                    or np.any(q < lower) or np.any(q > kin.upper)):
                continue
            shift = np.linalg.norm(kin.forward(q)[0] - target)
            violation = np.linalg.norm(self.violation(q))
            score = objective(q)[0]
            if shift <= self.budget and violation <= before + 1e-10 and score < best_score:
                best, best_score, accepted = q.copy(), score, True
        output = qpos.copy()
        output[kin.indices] = best
        self.previous = best.copy()
        return PostureResult(output, 'corrected' if accepted else 'fallback',
                             float(np.linalg.norm(kin.forward(best)[0] - target)),
                             float(before), float(np.linalg.norm(self.violation(best))), len(seeds))
