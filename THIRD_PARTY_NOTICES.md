# Third-party notices

This plugin includes an isolated legacy INT8 runtime derived from **comfy-kitchen 0.2.35**, published by the Comfy-Org/comfy-kitchen contributors under Apache License 2.0.

- Upstream project: https://github.com/Comfy-Org/comfy-kitchen
- Official distribution: https://pypi.org/project/comfy-kitchen/0.2.35/
- Source wheel manifest: `wheel-manifest.json`
- Original upstream license: [UPSTREAM_LICENSE](UPSTREAM_LICENSE)

The plugin's legacy Python runtime extracts and adapts the required INT8 operations into a private module. Package wiring and imports are changed for coexistence with the installed Kitchen. The private CUDA native extension is copied **unmodified** from a pinned, SHA-256-verified official PyPI wheel. It is not taken from a user's unverified environment.

The runtime is isolated in `vendor/comfy_kitchen_035/`. The original wheel LICENSE and NOTICE are copied without changes into `artifacts/<wheel-hash-prefix>/comfy_kitchen-0.2.35.dist-info/licenses/` under that directory. Each platform wheel's original METADATA and WHEEL are retained beside them. The directory's `provenance.json` records the source URLs and extracted file hashes.

The bundled extension incorporates upstream third-party components. The accompanying upstream NOTICE lists attribution and licenses for CUDA Runtime, PyTorch, nanobind, FlashAttention, torchao and SageAttention. This plugin does not replace or abbreviate that NOTICE. The unmodified top-level `UPSTREAM_NOTICE` also preserves this notice in a source-only distribution.

No endorsement by Comfy-Org or other upstream contributors is implied.
