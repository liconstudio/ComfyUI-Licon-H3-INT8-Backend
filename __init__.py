"""Model-scoped legacy Kitchen INT8 execution for MiniMax H3."""

from contextlib import contextmanager
import logging
import threading
import types
import uuid

import comfy.model_base
from comfy.patcher_extension import CallbacksMP, WrappersMP
import comfy_kitchen as ck
from comfy_kitchen.registry import registry


LEGACY_MODE = "Legacy INT8 (Kitchen 0.2.35)"
NATIVE_MODE = "Native"
_KEY = "comfy_h3_int8_backend.legacy035"
_STATE_ATTR = "_h3_int8_backend_legacy035_state"
_METHODS = ("load", "patch_weight_to_device", "patch_hook_weight_to_device")
_ABSENT = object()
_LOG = logging.getLogger(__name__)


def _runtime(mode=LEGACY_MODE):
    # Import lazily: Native remains usable without the optional native payload.
    if mode == LEGACY_MODE:
        from . import legacy_runtime
        return legacy_runtime
    raise ValueError(f"Unsupported legacy H3 INT8 mode: {mode}. Supported mode: {LEGACY_MODE}.")


def _native_count(stats):
    calls = stats.get("native_calls", {})
    return sum(int(value) for value in calls.values()) if isinstance(calls, dict) else int(calls)


def _snapshot(stats):
    return {key: dict(value) if isinstance(value, dict) else value
            for key, value in stats.items()}


class _ModelScope:
    def __init__(self, backend, layer_count, runtime, mode):
        self.backend = backend
        self.layer_count = layer_count
        # Keep the provider on this model instance, never in a global selector.
        self.runtime = runtime
        self.mode = mode
        self.original_methods = {}
        self.installed_methods = {}
        self.last_scopes = {}
        self.forward_count = 0
        self._lock = threading.Lock()

    @contextmanager
    def activate(self, label, require_linear=False):
        thread_local = getattr(registry, "_thread_local", None)
        if thread_local is None:
            raise RuntimeError("Kitchen lacks the thread-local backend interface required by H3 INT8 Backend.")
        previous = getattr(thread_local, "backend_override", None)
        # Kitchen falls back to global priority, not to an outer override, for
        # operations absent from this backend. Refuse that ambiguous composition.
        if previous not in (None, self.backend):
            raise RuntimeError("Another Kitchen backend override is active. H3 INT8 Backend cannot be combined with it.")
        completed = False
        with self.runtime.execution_scope() as stats:
            try:
                with ck.use_backend(self.backend):
                    yield stats
                if require_linear and (int(stats.get("int8_linear", 0)) == 0 or _native_count(stats) == 0):
                    raise RuntimeError(
                        "Legacy H3 INT8 received no verified legacy native INT8 linear calls. "
                        "A floating-GEMM patch, compiled graph, or another execution override may have bypassed it. "
                        "Use Native or remove the conflicting override; no fallback was performed."
                    )
                completed = True
            finally:
                observed = _snapshot(stats)
                observed["mode"] = self.mode
                observed["completed"] = completed
                with self._lock:
                    self.last_scopes[label] = observed
                    if label == "forward" and completed:
                        self.forward_count += 1
                        first = self.forward_count == 1
                    else:
                        first = False
                if first:
                    _LOG.info("H3 INT8: Kitchen %s native route verified: %s", self.runtime.VERSION, observed)


class _ForwardWrapper:
    def __init__(self, state):
        self.state = state

    def __call__(self, executor, *args, **kwargs):
        with self.state.activate("forward", require_linear=True):
            return executor(*args, **kwargs)


def _make_scoped_method(original, state, name):
    def scoped_method(_patcher, *args, **kwargs):
        with state.activate(name):
            return original(*args, **kwargs)
    scoped_method.__name__ = name
    scoped_method.__doc__ = getattr(original, "__doc__", None)
    return scoped_method


def _check_method_ownership(patcher):
    state = getattr(patcher, _STATE_ATTR, None)
    if state is not None:
        for name, installed in state.installed_methods.items():
            if patcher.__dict__.get(name, _ABSENT) is not installed:
                raise RuntimeError(
                    f"Another extension replaced ModelPatcher.{name} after H3 INT8 Backend. "
                    "Rebuild the model branch without that combination."
                )
    return state


def _remove_scope(patcher):
    state = _check_method_ownership(patcher)
    if state is not None:
        for name, original in state.original_methods.items():
            if original is _ABSENT:
                delattr(patcher, name)
            else:
                setattr(patcher, name, original)
        delattr(patcher, _STATE_ATTR)
    patcher.remove_wrappers_with_key(WrappersMP.DIFFUSION_MODEL, _KEY)
    patcher.remove_callbacks_with_key(CallbacksMP.ON_CLONE, _KEY)
    patcher.remove_attachments(_KEY)


def _clone_scope(source, target):
    state = _check_method_ownership(source)
    if state is None:
        raise RuntimeError("H3 INT8 clone callback lost its model-local state.")
    _install_scope(target, state.backend, state.layer_count, state.runtime, state.mode)


def _install_scope(patcher, backend, layer_count, runtime, mode):
    _remove_scope(patcher)
    state = _ModelScope(backend, layer_count, runtime, mode)
    # ON_LOAD runs after weight patching. Wrap this patcher's methods instead of
    # PREPARE_SAMPLING, which would also encompass unrelated auxiliary models.
    originals = {}
    for name in _METHODS:
        method = getattr(patcher, name, None)
        if not callable(method):
            raise RuntimeError(f"Unsupported ComfyUI ModelPatcher: missing {name}().")
        originals[name] = method
    for name, original in originals.items():
        state.original_methods[name] = patcher.__dict__.get(name, _ABSENT)
        installed = types.MethodType(_make_scoped_method(original, state, name), patcher)
        state.installed_methods[name] = installed
        setattr(patcher, name, installed)
    setattr(patcher, _STATE_ATTR, state)
    patcher.set_attachments(_KEY, state)
    patcher.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, _KEY, _ForwardWrapper(state))
    patcher.add_callback_with_key(CallbacksMP.ON_CLONE, _KEY, _clone_scope)


class H3Int8Backend:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "mode": ([LEGACY_MODE, NATIVE_MODE], {"default": LEGACY_MODE}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "configure"
    CATEGORY = "model_patches/H3"
    DESCRIPTION = (
        "Use the privately bundled Kitchen 0.2.35 INT8 implementation for this "
        "H3 DiT model branch, including ordinary weight patching and forward execution. "
        "Native removes this node's selection. Neither mode changes LoRA strength."
    )

    def configure(self, model, mode):
        if not isinstance(model.model, comfy.model_base.MiniMaxH3):
            raise ValueError("H3 INT8 Backend requires a MiniMax H3 model.")
        if mode not in (LEGACY_MODE, NATIVE_MODE):
            raise ValueError(
                f"Unsupported H3 INT8 mode: {mode}. Supported modes: {LEGACY_MODE}, {NATIVE_MODE}."
            )
        layers = [module for module in model.model.modules()
                  if getattr(module, "quant_format", None) == "int8_tensorwise"]
        if mode != NATIVE_MODE:
            if not layers:
                raise ValueError("This H3 model has no INT8 linear layers.")
            runtime = _runtime(mode)
            runtime.ensure_available()
            backend = runtime.register_backend()
        configured = model.clone()
        _remove_scope(configured)
        if mode != NATIVE_MODE:
            _install_scope(configured, backend, len(layers), runtime, mode)
        # Makes ComfyUI invalidate resident patched-weight state on mode changes.
        configured.patches_uuid = uuid.uuid4()
        _LOG.info("H3 INT8 backend: %s (H3 DiT model branch).", mode)
        return (configured,)


NODE_CLASS_MAPPINGS = {"H3Int8Backend": H3Int8Backend}
NODE_DISPLAY_NAME_MAPPINGS = {"H3Int8Backend": "Licon H3 INT8 Backend"}
WEB_DIRECTORY = "./web"
