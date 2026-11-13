# Licon H3 INT8 Backend

[Sample workflow](sample-workflow.json)

A ComfyUI node that switches a MiniMax H3 INT8 model between **Kitchen 0.2.35** and the **current native backend**.

In Licon's tests with H3 INT8 weights, **Kitchen 0.2.35 produced better generation results than the native backend in ComfyUI versions after 0.37**. Use the mode selector to compare them in your own workflow.

## Modes and ComfyUI versions

| Mode | INT8 implementation | ComfyUI version |
| --- | --- | --- |
| **Legacy INT8 (Kitchen 0.2.35)** | Fixed, plugin-private Kitchen **0.2.35**, matching the INT8 backend from the original ComfyUI 0.37 setup | Runs inside the user's current ComfyUI installation |
| **Native** | The native backend provided by the user's currently running ComfyUI environment | **The user's currently running ComfyUI version**; follows that installation and its dependencies |

**0.2.35 is a Kitchen version, not a ComfyUI version.** Legacy changes only the H3 INT8 path. Native uses the installed environment's normal behavior without pinning a ComfyUI or Kitchen version.

## Installation

1. Copy or clone this repository into `ComfyUI/custom_nodes/ComfyUI-Licon-H3-INT8-Backend`.
2. Start or restart ComfyUI. Wait for the startup hook to prepare the backend on first launch.
3. Refresh the ComfyUI page.

The startup hook uses **ComfyUI's Python** to verify the plugin-private Kitchen 0.2.35 extension. It downloads and verifies the official extension only when it is missing or invalid; an already verified installation needs no download. Existing Torch and Kitchen packages stay unchanged. ComfyUI-Manager can also run `install.py` during installation.

If setup fails, check the startup log. ComfyUI and Native mode remain available; Legacy mode reports an error instead of silently falling back. To retry manually, run:

```text
<ComfyUI Python> <ComfyUI>/custom_nodes/ComfyUI-Licon-H3-INT8-Backend/install.py
```

For offline setup, add `--wheel-file <path-to-the-pinned-official-wheel>` to that command, then restart ComfyUI.

**Requirements:** H3 INT8 ConvRot weights, NVIDIA GPU, CUDA 13 PyTorch, standard CPython 3.12+, and Linux x86_64 (glibc 2.27+) or Windows x64. Linux GPU execution has been tested; Windows GPU execution is not yet tested.

## Usage

Search for **Licon H3 INT8 Backend** and place it after all model and LoRA loaders:

```text
H3 INT8 model → LoRA loader(s) → Licon H3 INT8 Backend
                                  ├→ BasicGuider.model
                                  └→ BasicScheduler.model
```

Connect its output to **every downstream MODEL input**. Keep your existing LoRA order and strengths. Select `Legacy INT8 (Kitchen 0.2.35)` (default) or `Native`; keep the seed and other settings fixed when comparing.

Drag [sample-workflow.json](sample-workflow.json) into ComfyUI, select your model files and load your own input image. The sample uses **H3-Licon-VBVR-I2V-V2**, a Turbo LoRA and **rgthree-comfy** for the reference-image group switch.

Legacy mode uses existing INT8 weights; it does not convert BF16/FP8 models to INT8. Start with standard ComfyUI sampling; Torch Compile and other execution overrides need separate compatibility checks.

## How it works

The node clones the MODEL and selects a private Kitchen 0.2.35 backend within that model branch. It covers INT8 forward operations and weight processing, including dequantization and requantization after ordinary LoRA merging. Backend selection is thread-local, leaving the global backend priority unchanged; Native removes this node's override.

The text encoder, VAE and attention backend remain in the current environment. If the legacy backend is unavailable, the node reports an error instead of silently switching modes.

## License

[Apache-2.0](LICENSE). Includes adapted INT8 Python code from [Comfy-Org/comfy-kitchen](https://github.com/Comfy-Org/comfy-kitchen), with package wiring and imports changed for private use. The installer retrieves the unmodified official CUDA extension and preserves its original licenses. See [NOTICE](NOTICE) for upstream attribution.
