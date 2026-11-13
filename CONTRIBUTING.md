# Contributing

Use ComfyUI's existing environment for model-scope tests. The plugin intentionally does not install or downgrade Torch, ComfyUI or the installed comfy-kitchen. Report the OS, GPU, driver, Python, Torch/CUDA, ComfyUI and Kitchen versions with a bug report; remove private paths, prompts and tokens from logs.

CPU model-scope regression:

```text
<ComfyUI Python> tests/test_model_scope.py --comfy-root <ComfyUI>
```

This uses the real ComfyUI APIs and a simulated CPU provider; it does not establish native GPU correctness. Installer regression runs in GitHub Actions on Linux and Windows with pinned official wheels. It does not load a GPU.

For numerical runtime changes, repeat the GPU linear, weight-operation and complete H3 workflow comparisons described in VALIDATION.md. Keep old results clearly dated; do not represent historical evidence as a new test.

Preserve the existing node ID and mode values for workflow compatibility. Retain upstream licenses and provenance; record modifications to copied code. Contributions are under the repository's Apache-2.0 license.
