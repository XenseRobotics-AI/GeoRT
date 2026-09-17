"""Development gates for paired model targets, not hardware safety criteria."""
import numpy as np


def summarize(q, tips, ref_q, ref_tips, human):
    names = ('index', 'middle', 'ring', 'little')
    severity, mcp, reference_severity, reference_mcp = [], [], [], []
    for i in range(1, 5):
        severity.append(float(np.mean(np.any(q[:, 4*i+2:4*i+4] < -np.deg2rad(15), axis=1))))
        reference_severity.append(float(np.mean(np.any(ref_q[:, 4*i+2:4*i+4] < -np.deg2rad(15), axis=1))))
        mcp.append(float(np.mean(q[:, 4*i] < -np.deg2rad(15))))
        reference_mcp.append(float(np.mean(ref_q[:, 4*i] < -np.deg2rad(15))))
    shifts = np.linalg.norm(tips - ref_tips, axis=-1) * 1000
    steps = np.rad2deg(abs(np.diff(q, axis=0)))
    ref_steps = np.rad2deg(abs(np.diff(ref_q, axis=0)))
    pinch = {}
    changes = []
    for i, name in enumerate(names, 1):
        mask = np.linalg.norm(human[:, 4] - human[:, (i+1)*4], axis=-1) < .015
        old = np.linalg.norm(ref_tips[mask, 0] - ref_tips[mask, i], axis=-1) * 1000
        new = np.linalg.norm(tips[mask, 0] - tips[mask, i], axis=-1) * 1000
        pinch[name] = {'frames': int(mask.sum()), 'reference_mm': float(old.mean()) if len(old) else None,
                       'candidate_mm': float(new.mean()) if len(new) else None,
                       'mean_absolute_change_mm': float(abs(new-old).mean()) if len(new) else None}
        if len(new):
            changes.append(float(abs(new-old).mean()))
    amplitude = np.linalg.norm(np.std(tips, axis=0), axis=-1)
    reference_amplitude = np.linalg.norm(np.std(ref_tips, axis=0), axis=-1)
    ratio = np.divide(amplitude, reference_amplitude, out=np.ones_like(amplitude), where=reference_amplitude>1e-7)
    gates = {
        'backbend_reduction_80pct': np.mean(severity) <= .2*np.mean(reference_severity) + 1e-10,
        'no_finger_backbend_increase_5pp': np.max(np.array(severity)-reference_severity) <= .05 + 1e-10,
        'no_mcp_increase_5pp': np.max(np.array(mcp)-reference_mcp) <= .05 + 1e-10,
        'tip_p95_each_finger_5mm': np.percentile(shifts, 95, axis=0).max() <= 5.,
        'observed_pinch_change_2mm': max(changes, default=0) <= 2.,
        'max_step_increase_5deg': steps.max() <= ref_steps.max() + 5.,
        'motion_amplitude_retained': ratio.min() >= .8,
    }
    return {'severe_fraction': dict(zip(names,severity)), 'reference_severe_fraction': dict(zip(names,reference_severity)),
            'mcp_fraction': dict(zip(names,mcp)), 'reference_mcp_fraction': dict(zip(names,reference_mcp)),
            'tip_shift_mean_mm': float(shifts.mean()), 'tip_shift_p95_mm_per_finger': np.percentile(shifts,95,axis=0).tolist(),
            'max_joint_step_deg': float(steps.max()), 'reference_max_joint_step_deg': float(ref_steps.max()),
            'pinch': pinch, 'motion_amplitude_ratio': ratio.tolist(),
            'gates': {key:bool(v) for key,v in gates.items()}, 'passes_observed_gates': bool(all(gates.values())),
            'unobserved_pinch_pairs': [name for name, data in pinch.items() if not data['frames']],
            'scope': 'contiguous development segment; relative to parent targets, not operational ground truth'}
