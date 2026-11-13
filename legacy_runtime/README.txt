Private Kitchen 0.2.35 INT8 runtime

This directory is internal to the node plugin. It does not replace or reload
the installed comfy_kitchen package. No attention implementation is exported.

_algorithms.py contains 46 original CUDA-backend definitions selected from
official commit b2a2972ac68c395bbda8ad9030e8ae1089287815, preserving their AST.
_eager.py contains 14 original pure helpers for legitimate CPU weight
quantization/dequantization. _activations.py and _int8_utils.py preserve their
complete old AST. source_manifest.json identifies upstream files/lines;
source_verification.json records the comparison.

Native calls use only the plugin-private _C module, loaded from the plugin's
vendor/comfy_kitchen_035 directory and checked against fixed official binary
SHA256 fingerprints. The current system _C remains a separate module object.
The old wrapper's own CUDA/cuBLAS or old Torch fallback branches are preserved;
failure never selects a current-Kitchen numerical implementation.

load_backend() returns nine callable operations. ensure_available() verifies
and imports the extension without executing GPU kernels. register_backend()
registers passive capabilities and never changes the global priority. Its
constraints are None so unsupported calls reach our explicit checks rather
than being silently rerouted to another Kitchen backend.

execution_scope() collects thread-local counters only; the caller owns the
Kitchen backend context. Nested scopes accumulate calls in each containing
scope. Stats include int8_linear count, native_calls total, native_by_name,
python_calls and cpu_weight_calls. Native calls are counted at the actual
private extension boundary. No CUDA synchronization is added to production.

CPU weight operations use the private old eager helpers, matching old
Kitchen's CPU dispatch. Forward int8_linear requires CUDA. This supports
offloaded weight patching without accidentally invoking the installed package.

Audited artifacts target standard GIL-enabled CPython 3.12+ on Linux x86_64 or
Windows x86_64, with a CUDA 13 Torch build. Linux simultaneous old/new extension
loading and GPU execution were tested on RTX PRO 6000 Blackwell. The Windows
artifact was verified against the official wheel but not executed on Windows.

Validation performed for this release: exact original Python AST; successful
private native execution; nine actual H3 matrix-shape cases matching an
independent canonical .35 process bit for bit; 24 CPU/CUDA weight-quantization
cases with qdata, scale and dequantized outputs matching canonical .35 bit for
bit. This validates the selected legacy runtime, not a claim of image-quality
improvement or a diagnosed GEMM regression.

The copied upstream code is Apache-2.0. Preserve its copyright notices and the
upstream LICENSE/NOTICE shipped with the vendor artifacts.
