# Changelog

## 0.2.2

- Automatically verify and prepare the private Kitchen 0.2.35 backend at ComfyUI startup; download only for first installation or repair. Setup failures leave ComfyUI and Native mode available while Legacy reports an explicit error.
- Add the supplied H3 I2V sample workflow and concise English usage documentation, including environment requirements and the author's latest quality comparison. Native follows the user's running ComfyUI environment.

- Rename the repository/folder to `ComfyUI-Licon-H3-INT8-Backend` and the displayed node to `Licon H3 INT8 Backend`.
- Use `licon-h3-int8-backend` as the intended Registry package ID.
- Show `Licon` in this node's source badge while retaining its actual module and Manager metadata.
- Keep `H3Int8Backend`, both mode values, model hooks and numerical implementations unchanged.
- Separate source distribution from optional preinstalled native binaries; add Git/Registry exclusions, CI and release instructions.
- The intended repository URL is configured; Registry publisher identity remains to be filled before publication.

## 0.2.1

- Remove Kitchen 0.2.34; retain Kitchen 0.2.35 and Native modes.
- Keep the 0.2.35 numerical implementation and official native binaries unchanged.
