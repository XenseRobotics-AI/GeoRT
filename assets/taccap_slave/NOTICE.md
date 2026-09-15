# TacCap asset provenance

## G1 slave and mounting adapter

This private-development package extracts the TacCap G1 slave model and five
STL files from XenseRobotics-AI/robosuite-xense, revision
`865e81be89706f8d1d25ba83119c1d0f3de8d4e9`. Original paths and checksums are in
`config/model.json`; `LICENSES/robosuite.txt` preserves the source repository's
license notice.

The source describes the CAD input as a SolidWorks-exported TacCap URDF and
the adapter as originating from `forward-44R-silex/A-/A-.urdf`. No immutable
original CAD revision or asset-specific redistribution grant was recorded.
The source assembly notice labels project-supplied assemblies Apache-2.0,
while its Forward44R README says redistribution provenance is incomplete.
This extraction preserves that uncertainty and does not infer a license for
meshes from the surrounding Python code. Confirm those terms before public
distribution; no GitHub repository or package-index release is created here.

The Xacro is a mechanical transcription of the existing simulation model.
Generated URDF and MJCF files use local relative mesh paths. They retain the
existing simulated masses, axes, limits and mounting transform, including
approximations documented in `config/model.json`. They are not hardware
calibration data. The source filename `viusal_link.STL` is intentionally
preserved for provenance.

Only the TacCap component and its Asm_link adapter are included. No Flexiv arm,
Forward44R stand or Wuji hand assets are redistributed by this package.
