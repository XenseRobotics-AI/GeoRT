# Wuji Hand2 Beta1 right

Imported from the local `wuji-description/hand2/hand2_beta1/body` model.
This is **Beta1**, right hand, without mount. See `SOURCE.json` for source
revision and SHA-256 checksums; upstream MIT license is retained in `LICENSE`.

`urdf/right.urdf` and its referenced meshes are unchanged copies.
`urdf/wuji_hand2_beta1_right.urdf` adds only a fixed `geort_base` above
`r_wrist`, with RPY `(pi, 0, pi/2)`. Native +Y becomes GeoRT +X (palm
normal), native +X becomes +Y (thumb side), and native -Z becomes +Z
(finger direction). No joint signs, ranges, geometry or inertias are changed.

GeoRT config: `wuji_hand2_beta1_right`. Output order is thumb, index, middle,
ring, little, four joints each. Tip frames are the upstream fixed `*_tip`
links with zero additional offset. Human point IDs are 4, 8, 12, 16, 20.

The source joint limits are retained, including hyperextension. The default
preview sweeps only part of the flexion range. Training uses the full configured
range; this import is not a claim of good retargeting quality or hardware validation.
