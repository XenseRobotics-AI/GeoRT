# TacCap G1 slave

Imported from local `xense-description/taccap/slave`. Original URDF and meshes
are retained; see `SOURCE.json` for hashes and `NOTICE.md` / `LICENSES` for
upstream provenance and license information.

`urdf/taccap_slave.urdf` adds only a fixed `geort_base` above the native base
with RPY `(0, -pi/2, 0)`, so the fingers point along GeoRT +Z and the opening
is along Y. `urdf/taccap_g1_slave.urdf` is the unchanged source copy.

This gripper has one independent opening command and two physical joints:
`grip_right_joint = grip_left_joint`, with opposite axes. The URDF mimic
relation is preserved, and the direct-pose preview explicitly sets both angles
equally. The inherited 0..1 rad range is a simulation range, not a calibrated
hardware opening range.

Use `python -m geort.mocap.replay_evaluation --preview -hand taccap_slave --animate` for a kinematic
preview. No standard GeoRT training config is supplied: its current per-finger
IK networks would predict two independent commands and violate the coupling.
Training retargeting for this gripper requires a shared opening output.
