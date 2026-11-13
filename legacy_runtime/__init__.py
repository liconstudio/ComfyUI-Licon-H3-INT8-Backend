"""Private, verified Kitchen 0.2.35 INT8 runtime. Never swaps system modules.

The Python algorithms are an exact AST closure of the upstream .35 CUDA backend.
Its native calls go only to the private, hash-verified .35 extension. Current
Kitchen supplies the registration interface, not numerical implementations.
"""
from contextlib import contextmanager
import ctypes
import functools
import hashlib
import importlib
import importlib.util
import os
from pathlib import Path
import platform
import sys
import sysconfig
import threading
from types import SimpleNamespace

import torch

BACKEND_NAME = "h3_int8_legacy_cuda_035_v1"
VERSION = "0.2.35"
_NATIVE_SHA256 = {
    "linux": "948447031483b98828137480f8642ade050262218e142f9c0ae0ddadccbfc3de",
    "win32": "e91d27c67607dbcb9a1a2c331ee31173f548fd9a3420e59d23fb2ca74ba78e62",
}
CAPABILITIES = (
    "int8_linear", "quantize_int8_tensorwise", "quantize_int8_rowwise",
    "quantize_int8_convrot_weight", "rotate_int8_convrot_weight",
    "dequantize_int8_simple", "dequantize_int8_simple_dtype",
    "dequantize_int8_convrot_weight", "dequantize_int8_convrot_weight_dtype",
)
_lock = threading.RLock()
_local = threading.local()
_loaded = None
_native = None
_registered = False
_dll_handles = []


def _record(kind, name):
    for stats in getattr(_local, "frames", ()):
        counter = stats["native_by_name" if kind == "native" else "cpu_weight_calls" if kind == "cpu" else "python_calls"]
        counter[name] = counter.get(name, 0) + 1
        if kind == "native":
            stats["native_calls"] += 1
        elif name == "int8_linear":
            stats["int8_linear"] += 1


@contextmanager
def execution_scope():
    """Collect per-thread old-runtime calls; caller owns registry.use_backend()."""
    stats = {"backend": BACKEND_NAME, "native_calls": 0, "native_by_name": {},
             "python_calls": {}, "cpu_weight_calls": {}, "int8_linear": 0}
    frames = getattr(_local, "frames", None)
    if frames is None:
        frames = _local.frames = []
    frames.append(stats)
    try:
        yield stats
    finally:
        if not frames or frames[-1] is not stats:
            raise RuntimeError("Legacy INT8 execution scopes exited out of order")
        frames.pop()
        if not frames:
            del _local.frames


class _NativeProxy:
    """Count actual private .35 native entry calls, without altering operands."""
    def __init__(self, module):
        self.module = module
        self.functions = {}

    def __getattr__(self, name):
        value = getattr(self.module, name)
        if not callable(value):
            return value
        if name not in self.functions:
            def call(*args, **kwargs):
                _record("native", name)
                return value(*args, **kwargs)
            self.functions[name] = call
        return self.functions[name]


def _prepare_native_libraries():
    # Upstream .35's loader prepares cuBLASLt similarly. The active Torch/Kitchen
    # normally loaded these already; this also supports a direct private import.
    try:
        import nvidia.cu13
        roots = list(nvidia.cu13.__path__)
    except ImportError:
        roots = list(torch.__path__)
    for root in roots:
        for directory, _, files in os.walk(root):
            candidates = [f for f in files if "cublasLt" in f]
            if not candidates:
                continue
            if sys.platform == "win32":
                _dll_handles.append(os.add_dll_directory(directory))
            else:
                for filename in candidates:
                    if ".so" in filename:
                        try:
                            ctypes.CDLL(str(Path(directory) / filename), mode=ctypes.RTLD_GLOBAL)
                        except OSError:
                            pass  # Native import below reports an actionable failure.
            return


def _load_native():
    if sys.platform not in _NATIVE_SHA256:
        raise RuntimeError("Legacy INT8 CUDA currently supports Linux x86_64 and Windows x86_64 only")
    if platform.machine().lower() not in ("x86_64", "amd64"):
        raise RuntimeError("The audited private Kitchen .35 native artifacts require x86_64")
    if platform.python_implementation() != "CPython" or sysconfig.get_config_var("Py_GIL_DISABLED"):
        raise RuntimeError("The audited private Kitchen .35 runtime requires standard GIL-enabled CPython")
    cuda_version = getattr(torch.version, "cuda", None)
    if not cuda_version or int(cuda_version.split(".")[0]) < 13:
        raise RuntimeError("Kitchen .35 native runtime requires a CUDA 13 Torch build; no current-kernel fallback is allowed")
    if sys.version_info < (3, 12):
        raise RuntimeError("The audited Kitchen .35 native wheels require Python 3.12 or newer")
    filename = "_C.abi3.pyd" if sys.platform == "win32" else "_C.abi3.so"
    path = (Path(__file__).resolve().parents[1] / "vendor" / "comfy_kitchen_035" /
            "comfy_kitchen" / "backends" / "cuda" / filename)
    if not path.is_file():
        raise RuntimeError(f"Missing private Kitchen .35 runtime: {path}. Run this plugin's install.py with the ComfyUI Python interpreter.")
    expected = _NATIVE_SHA256[sys.platform]
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise RuntimeError(f"Private Kitchen .35 native SHA256 mismatch: {path}")
    name = __name__ + "._C"
    existing = sys.modules.get(name)
    if existing is not None:
        if Path(getattr(existing, "__file__", "")).resolve() != path.resolve():
            raise RuntimeError("Private native module name is occupied by another file")
        return existing
    _prepare_native_libraries()
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load private Kitchen .35 extension: {path}")
    try:
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    except Exception as error:
        sys.modules.pop(name, None)
        raise RuntimeError(f"Private Kitchen .35 native import failed; current CUDA kernels will not be used: {error}") from error
    if module is sys.modules.get("comfy_kitchen.backends.cuda._C"):
        raise RuntimeError("Private legacy runtime unexpectedly resolved to the system Kitchen extension")
    required = ("quantize_int8_rowwise_convrot64", "int8_linear_m1", "cutlass_int8_dequant",
                "cublas_gemm_int8", "dequantize_int8_convrot_weight")
    if any(not hasattr(module, entry) for entry in required):
        raise RuntimeError("Private Kitchen .35 native entry points are incomplete")
    return module


def _strict_function(name, function, cpu_function=None):
    @functools.wraps(function)
    def call(*args, **kwargs):
        first_name = function.__code__.co_varnames[0]
        first = args[0] if args else kwargs.get(first_name)
        if isinstance(first, torch.Tensor) and first.device.type == "cpu" and cpu_function is not None:
            _record("python", name)
            _record("cpu", name)
            return cpu_function(*args, **kwargs)
        if not isinstance(first, torch.Tensor) or first.device.type != "cuda":
            raise RuntimeError(f"Legacy INT8 {name} requires a CUDA tensor; no fallback to current Kitchen is allowed")
        if torch.version.hip:
            raise RuntimeError("The private .35 runtime is NVIDIA CUDA only")
        if name == "int8_linear":
            weight = args[1] if len(args) > 1 else kwargs.get("weight")
            if not isinstance(weight, torch.Tensor) or weight.dtype != torch.int8:
                raise TypeError("Legacy INT8 linear requires an INT8 weight")
            if first.dtype not in (torch.float16, torch.bfloat16, torch.float32):
                raise TypeError("Legacy INT8 activations must be float16, bfloat16 or float32")
            if weight.device != first.device:
                raise ValueError("Legacy INT8 weight and activation must be on the same CUDA device")
        _record("python", name)
        return function(*args, **kwargs)
    return call


def load_backend():
    """Return the private backend's callable interface; import/hash only until called."""
    global _loaded, _native
    with _lock:
        if _loaded is None:
            native = _load_native()
            algorithms = importlib.import_module(__name__ + "._algorithms")
            eager = importlib.import_module(__name__ + "._eager")
            algorithms._C = _NativeProxy(native)
            functions = {name: _strict_function(name, getattr(algorithms, name),
                         None if name == "int8_linear" else getattr(eager, name)) for name in CAPABILITIES}
            _loaded = SimpleNamespace(**functions)
            _native = native
        return _loaded


def ensure_available():
    load_backend()
    return {"version": VERSION, "backend": BACKEND_NAME,
            "native_path": _native.__file__, "native_sha256": _NATIVE_SHA256[sys.platform],
            "private_module": _native.__name__, "capabilities": list(CAPABILITIES)}


def register_backend():
    """Register passive capabilities only; never change Kitchen priority or enable flags."""
    global _registered
    from comfy_kitchen.registry import registry
    with _lock:
        backend = load_backend()
        if not _registered:
            if BACKEND_NAME in registry.list_backends():
                raise RuntimeError("Legacy INT8 backend name already exists; refusing to replace another plugin's backend")
            # None means registry does not silently decline a shape and fall back.
            # CUDA/device/type validation happens inside our strict callable and
            # remaining .35 constraints either execute its own fallback or raise.
            registry.register(BACKEND_NAME, backend, {name: None for name in CAPABILITIES})
            _registered = True
        return BACKEND_NAME


def __getattr__(name):
    if name in CAPABILITIES:
        return getattr(load_backend(), name)
    raise AttributeError(name)
