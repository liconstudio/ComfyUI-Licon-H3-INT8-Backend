# Author update — 2026-11-13

Licon has verified that Kitchen 0.2.35 produces better generation results than Native in some cases when loading H3 INT8 weights. This is the author's latest testing result, reflected in README.md. The historical single-workflow equality result below remains valid for that particular test and does not contradict the newer observation. No new quantitative benchmark is asserted here.

Native always means the backend of the user's currently running ComfyUI environment. Any ComfyUI/Kitchen versions recorded below describe specific historical test environments, not a version requirement or a fixed definition of Native. Legacy pins only its private Kitchen 0.2.35 INT8 implementation.

# Release 0.2.1 validation

Kitchen 0.2.34 has been removed from the node, installer, manifest and bundled payload. The available modes are Kitchen 0.2.35 and Native. The 0.2.35 numerical implementation and official native binaries are unchanged.

The model-scope and installer regression checks are recorded in `tests/validation-model-scope.json` and `tests/validation-installer.json`. Numerical and full-workflow evidence below was collected on 2026-10-28 for the unchanged 0.2.35 implementation; it is not presented as a new full-workflow run for 0.2.1.

# Release 0.1.0 validation

Validated on 2026-10-28. Runtime target: Linux x86_64, standard CPython 3.12, ComfyUI 0.42, PyTorch 2.9/cu130, NVIDIA RTX PRO 6000 Blackwell. Windows native payload provenance was checked; Windows GPU execution was not tested.

## Implementation fidelity

The private CUDA algorithms were copied from Kitchen 0.2.35 commit `b2a2972ac68c395bbda8ad9030e8ae1089287815`. AST comparison passed for 46 CUDA definitions, 14 CPU weight definitions, and the activation/INT8 utility files. See `legacy_runtime/source_verification.json`.

The two native payloads came from pinned official PyPI wheels. The installer verifies whole-wheel hashes and extracted binary hashes, rejects invalid assets, and supports an offline check. No pip installation or global Kitchen replacement is performed.

## Numerical tests

Nine standalone GPU linear cases compared the private plugin against an independent Kitchen 0.2.35 installation. All output tensor hashes matched exactly. Cases covered H3 AdaLN, QKV, output projection, fc1, fused SwiGLU fc2, small and large token counts, BF16 and FP16. Current Kitchen 0.2.39 also matched the old implementation for these nine cases.

Twenty-four weight cases compared CPU and CUDA, BF16/FP16/FP32, ConvRot groups 64/256, and stochastic seeds 0/11235. Quantized bytes, scales, and dequantized results all matched the independent old implementation exactly.

See `tests/validation-gpu-linear.json` and `tests/validation-weight-ops.json`.

## Model lifecycle

Tests exercised the actual ComfyUI ModelPatcher API with a clearly identified CPU test provider. They verified independent cloned state, weight patch/load/unpatch routing, existing wrapper preservation, Native restoration, thread-local cleanup after errors, conflict rejection, and explicit failure when no INT8 forward was executed. This test provider is not evidence of native GPU execution; the integration test below provides that evidence. Dynamic-patcher metadata tests used a mock unused HostBuffer.

See `tests/test_model_scope.py` and `tests/validation-model-scope.json`.

## Full workflow

A real H3 INT8 ConvRot workflow with two standard LoRA loaders completed 8 sampling steps and encoded 243 video frames at 864×480, 24 FPS (10.125 seconds), with an audio stream. The first model forward reported 250 legacy INT8 linear invocations and 900 private CUDA entry calls, including INT8 GEMV/GEMM and ConvRot weight operations. The plugin supplied its own old native extension while the current installed Kitchen remained present.

The same workflow and fixed seed completed after switching the node to Native. Both outputs decoded successfully. Their decoded video SHA-256 hashes were identical:

```text
91405b1b21bc0c0314704a7a0bd427727bede0d07fedefd4becb3c09ddfd3313
```

This verifies an operational backend selector and successful switching. It does **not** demonstrate recovery of earlier generation quality. For this tested workflow, changing only these INT8 implementations produced no decoded-frame difference. The test does not prove equivalence for all checkpoints, dimensions, seeds or environments. Cold/warm job timings are not a controlled performance benchmark.

Private prompts, input images, model files, service history and generated videos are not included in the public plugin package.

## Cross-version video comparison caveat

A saved ComfyUI 0.37 baseline used the same generation parameters, seed, model-file contents and input-image contents as the new run. Its decoded video hash differed, but the actual H.264 bitstreams revealed a confounding encoding change: the old output used CRF 23 and the new output used CRF 18. Across 243 frames, the saved videos had SSIM 0.98610127 and PSNR 44.1111 dB. These compressed-output measurements do not establish whether the tensors before video encoding were identical or different, and are not evidence that the INT8 replacement repaired a generation regression.
