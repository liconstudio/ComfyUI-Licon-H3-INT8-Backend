"""CPU tests of real ComfyUI routing, with an explicitly simulated provider.

This suite does not validate the private CUDA binary. Run the separate runtime
tests for that. No model downloads, checkpoint loads, or GPU allocations occur.
"""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
import threading
import types
from unittest.mock import patch


def run_tests(plugin_source=None):
    import comfy.options
    comfy.options.enable_args_parsing()
    import torch
    import comfy.ops
    import comfy.model_base
    import comfy.model_patcher
    import comfy.patcher_extension as pe
    import comfy.ldm.minimax.model
    import comfy_kitchen as ck
    from comfy.quant_ops import QuantizedTensor, get_layout_class
    from comfy_kitchen.registry import registry

    torch.set_num_threads(1)
    torch.manual_seed(321)
    package = types.ModuleType("h3_int8_model_scope_test")
    package.__package__ = package.__name__
    package.__path__ = []
    sys.modules[package.__name__] = package
    if plugin_source is None:
        plugin_source = (Path(__file__).resolve().parents[1] / "__init__.py").read_text(encoding="utf-8")
    exec(compile(plugin_source, "h3_int8_backend/__init__.py", "exec"), package.__dict__)

    # Deliberately use current eager CPU functions as a test double. Counters in
    # this suite are simulated, never evidence of old CUDA execution.
    operations = ("int8_linear", "quantize_int8_rowwise", "quantize_int8_tensorwise",
                  "quantize_int8_convrot_weight", "rotate_int8_convrot_weight",
                  "dequantize_int8_simple", "dequantize_int8_simple_dtype",
                  "dequantize_int8_convrot_weight", "dequantize_int8_convrot_weight_dtype")
    def make_provider(module_name, version):
        fake = types.ModuleType(package.__name__ + "." + module_name)
        fake.VERSION = version
        fake.backend = "h3_int8_cpu_scope_test_double_" + version.replace(".", "_")
        tls = threading.local()
        calls = []
        def implementation(name):
            def call(*args, **kwargs):
                fn = registry.get_implementation(name, backend="eager", kwargs=kwargs or None)
                calls.append({"version": version, "op": name, "input_act": kwargs.get("input_act")})
                for frame in getattr(tls, "frames", ()):
                    frame["python_calls"][name] = frame["python_calls"].get(name, 0) + 1
                    frame["native_calls"] += 1  # Explicit test double only.
                    frame["native_by_name"][name] = frame["native_by_name"].get(name, 0) + 1
                    if name == "int8_linear":
                        frame["int8_linear"] += 1
                return fn(*args, **kwargs)
            return call
        funcs = types.SimpleNamespace(**{name: implementation(name) for name in operations})
        def register():
            if fake.backend not in registry.list_backends():
                registry.register(fake.backend, funcs, {name: None for name in operations})
            return fake.backend
        @contextmanager
        def execution_scope():
            frame = {"version": version, "int8_linear": 0, "native_calls": 0,
                     "native_by_name": {}, "python_calls": {}}
            if not hasattr(tls, "frames"):
                tls.frames = []
            tls.frames.append(frame)
            try:
                yield frame
            finally:
                tls.frames.pop()
                if not tls.frames:
                    del tls.frames
        fake.ensure_available = lambda: {"test_double": True, "version": version}
        fake.register_backend = register
        fake.execution_scope = execution_scope
        setattr(package, module_name, fake)
        sys.modules[fake.__name__] = fake
        return fake, calls, tls
    fake, calls, tls = make_provider("legacy_runtime", "0.2.35")

    class TinyH3(comfy.model_base.MiniMaxH3):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self.diffusion_model = torch.nn.Module()
            ops = comfy.ops.mixed_precision_ops(compute_dtype=torch.bfloat16)
            self.diffusion_model.proj = ops.Linear(64, 64, bias=False)
            self.diffusion_model.mlp = comfy.ldm.minimax.model.MLP(64, 64, operations=ops)
            layout = get_layout_class("TensorWiseINT8Layout")
            for layer in self.modules():
                if isinstance(layer, comfy.ops.MixedPrecisionOp):
                    layer.quant_format = "int8_tensorwise"
                    layer.layout_type = "TensorWiseINT8Layout"
                    shape = (layer.out_features, layer.in_features)
                    params = layout.Params(scale=torch.full((shape[0], 1), .01),
                                           orig_dtype=torch.bfloat16, orig_shape=shape,
                                           convrot=True, convrot_groupsize=64)
                    layer.weight = torch.nn.Parameter(QuantizedTensor(
                        torch.randint(-8, 8, shape, dtype=torch.int8),
                        "TensorWiseINT8Layout", params), requires_grad=False)

    model = TinyH3()
    cpu = torch.device("cpu")
    base = comfy.model_patcher.ModelPatcher(model, cpu, cpu)
    key = "diffusion_model.proj.weight"
    base.add_patches({key: ("diff", (torch.full((64, 64), .001),))})
    base.set_injections("existing", [pe.PatcherInjection(lambda _: None, lambda _: None)])
    base.add_wrapper_with_key(pe.WrappersMP.DIFFUSION_MODEL, "existing", lambda executor, *a, **k: executor(*a, **k))
    base.add_callback_with_key(pe.CallbacksMP.ON_CLONE, "existing", lambda _old, _new: None)
    base_data = [(m, m.weight._qdata.clone(), m.weight._params.scale.clone()) for m in model.modules()
                 if isinstance(getattr(m, "weight", None), QuantizedTensor)]
    priority, disabled = tuple(registry._priority), set(registry._disabled)
    node = package.H3Int8Backend()
    mode_schema = node.INPUT_TYPES()["required"]["mode"]
    assert mode_schema[1]["default"] == package.LEGACY_MODE == "Legacy INT8 (Kitchen 0.2.35)"
    assert mode_schema[0] == [package.LEGACY_MODE, package.NATIVE_MODE]
    assert package._runtime() is fake
    selected, = node.configure(base, package.LEGACY_MODE)
    downstream = selected.clone()  # Normal LoRA loaders clone the selected branch.
    downstream.add_patches({key: ("diff", (torch.zeros(64, 64),))})
    native, = node.configure(downstream, package.NATIVE_MODE)
    assert selected.patches == base.patches and downstream.patches != base.patches
    assert selected.get_injections("existing") == base.get_injections("existing")
    assert selected.get_attachment(package._KEY) is not downstream.get_attachment(package._KEY)
    for name in package._METHODS:
        assert name not in base.__dict__ and name not in native.__dict__
        assert name in selected.__dict__ and name in downstream.__dict__
    assert not native.get_attachment(package._KEY)
    assert native.get_wrappers(pe.WrappersMP.DIFFUSION_MODEL, "existing")
    assert native.get_callbacks(pe.CallbacksMP.ON_CLONE, "existing")
    assert not native.get_callbacks(pe.CallbacksMP.ON_CLONE, package._KEY)
    for layer, data, scale in base_data:
        assert torch.equal(layer.weight._qdata, data) and torch.equal(layer.weight._params.scale, scale)

    def override():
        return getattr(registry._thread_local, "backend_override", None)
    def execute(patcher, fn, *args):
        options = {"wrappers": {}}
        pe.merge_nested_dicts(options["wrappers"], patcher.wrappers, copy_dict1=False)
        wrappers = pe.get_all_wrappers(pe.WrappersMP.DIFFUSION_MODEL, options)
        return pe.WrapperExecutor.new_class_executor(fn, model.diffusion_model, wrappers).execute(*args)
    def forward(x):
        return model.diffusion_model.proj(x), model.diffusion_model.mlp(x)
    x = torch.randn(3, 64, dtype=torch.bfloat16)
    results = {"scope_provider": "CPU eager TEST DOUBLE; native counters simulated",
               "actual_comfy_model_patcher": True, "configure_does_not_mutate_weight_storage": True,
               "clones_preserve_patches_injections_and_independent_scope": True,
               "native_restores_methods_and_only_removes_owned_hooks": True}

    # Exercise the real QT F.linear and H3 fused MLP entry paths.
    y = execute(downstream, forward, x)
    linear = [c for c in calls if c["op"] == "int8_linear"]
    assert [c["input_act"] for c in linear] == [None, None, "swiglu"]
    assert all(torch.isfinite(out).all() for out in y)
    assert override() is None and not hasattr(tls, "frames")
    results["ordinary_and_fused_linear_routes"] = linear
    before = len(calls)
    native_y = execute(native, forward, x)
    assert len(calls) == before
    for a, b in zip(y, native_y):
        torch.testing.assert_close(a, b, rtol=0, atol=0)

    # Actual CPU LoRA calculation and ConvRot re-quantization before forward.
    calls.clear()
    patched = downstream.patch_weight_to_device(key, return_weight=True)
    assert isinstance(patched, QuantizedTensor)
    assert any(c["op"] == "quantize_int8_convrot_weight" for c in calls)
    assert any(c["op"].startswith("dequantize_int8") for c in calls)
    assert override() is None
    results["real_patch_weight_to_device_routes"] = list(calls)
    assert torch.equal(model.diffusion_model.proj.weight._qdata, base_data[0][1])

    # Exercise ordinary complete load and undo it, using core APIs.
    calls.clear()
    downstream.load(cpu, full_load=True)
    assert downstream.get_attachment(package._KEY).last_scopes["load"]["completed"]
    assert any(c["op"] == "quantize_int8_convrot_weight" for c in calls)
    downstream.unpatch_model(cpu)
    assert torch.equal(model.diffusion_model.proj.weight._qdata, base_data[0][1])
    assert override() is None
    results["real_load_and_unpatch_restore_base_weight"] = True

    # Scope selection is thread-local and non-INT8 operators stay unchanged.
    other = "rms_rope"
    expected = registry.get_implementation(other)
    def context_probe(a):
        assert registry.get_implementation(other) is expected
        observed = []
        thread = threading.Thread(target=lambda: observed.append(override()))
        thread.start(); thread.join(timeout=5)
        assert not thread.is_alive() and observed == [None]
        return forward(a)
    execute(selected, context_probe, x)
    with ck.use_backend(fake.backend):
        execute(selected, forward, x)
        assert override() == fake.backend
    with ck.use_backend("eager"):
        try:
            execute(selected, forward, x)
            raise AssertionError("Foreign override should fail")
        except RuntimeError as exc:
            assert "Another Kitchen backend override" in str(exc)
        assert override() == "eager"
    assert override() is None
    results["thread_isolation_other_ops_and_context_restoration"] = True
    for fn, message in ((lambda a: a, "no verified legacy"),
                        (lambda a: (_ for _ in ()).throw(ValueError("executor failure")), "executor failure")):
        try:
            execute(selected, fn, x)
            raise AssertionError("Expected failure")
        except (RuntimeError, ValueError) as exc:
            assert message in str(exc)
        assert override() is None and not hasattr(tls, "frames")
    results["zero_hits_and_executor_errors_fail_with_restored_context"] = True

    # Already full-precision linear execution must not masquerade as legacy INT8.
    for layer, _, _ in base_data:
        layer._full_precision_mm = True
    try:
        execute(selected, forward, x)
        raise AssertionError("Full-precision bypass should fail")
    except RuntimeError as exc:
        assert "no verified legacy" in str(exc)
    finally:
        for layer, _, _ in base_data:
            layer._full_precision_mm = False
    results["full_precision_bypass_detected"] = True

    # The removed workflow value must fail before loading any provider or cloning
    # a model. It must never silently select the remaining legacy runtime.
    removed_mode = "Legacy INT8 (Kitchen 0.2.34)"
    with patch.object(package, "_runtime", side_effect=AssertionError("Unexpected runtime selection")):
        try:
            node.configure(base, removed_mode)
            raise AssertionError("Removed legacy mode should fail")
        except ValueError as exc:
            assert "Unsupported H3 INT8 mode" in str(exc) and removed_mode in str(exc)
    try:
        package._runtime(removed_mode)
        raise AssertionError("Removed provider should fail")
    except ValueError as exc:
        assert "Unsupported legacy H3 INT8 mode" in str(exc) and removed_mode in str(exc)
    results["removed_034_workflow_is_rejected_without_fallback"] = True

    # Retain model-local provider state across Native round trips and ordinary
    # downstream clones. Reconfiguring must not accumulate wrappers/callbacks.
    from_native, = node.configure(native, package.LEGACY_MODE)
    roundtrip_clone = from_native.clone()
    reconfigured, = node.configure(roundtrip_clone, package.LEGACY_MODE)
    assert not native.get_attachment(package._KEY)
    assert all(name not in native.__dict__ for name in package._METHODS)
    assert roundtrip_clone.get_attachment(package._KEY) is not from_native.get_attachment(package._KEY)
    for branch in (selected, from_native, roundtrip_clone, reconfigured):
        state = branch.get_attachment(package._KEY)
        assert state.runtime is fake and state.mode == package.LEGACY_MODE and state.backend == fake.backend
        before = len(calls)
        execute(branch, forward, x)
        assert len(calls) > before
        assert state.last_scopes["forward"]["version"] == "0.2.35"
        assert state.last_scopes["forward"]["mode"] == package.LEGACY_MODE
        assert not hasattr(tls, "frames") and override() is None
        assert branch.get_injections("existing") == base.get_injections("existing")
        assert len(branch.get_callbacks(pe.CallbacksMP.ON_CLONE, package._KEY)) == 1
        assert len(branch.get_wrappers(pe.WrappersMP.DIFFUSION_MODEL, package._KEY)) == 1
    results["legacy_native_roundtrip_preserves_model_local_runtime"] = True
    results["legacy_035_default_and_native_workflow_compatibility"] = True

    # Dynamic patchers and the core non-dynamic delegate use ON_CLONE too.
    # No dynamic VRAM allocation/load is performed in this CPU suite.
    # A CUDA device descriptor avoids core's automatic CPU -> static patcher
    # conversion. The tensors stay on CPU and no load/VRAM call is made.
    # Aimdo is disabled by --cpu. Replace only the unused host-buffer constructor
    # for this metadata/clone test; this is not a dynamic-load test.
    import comfy_aimdo.host_buffer
    with patch.object(comfy_aimdo.host_buffer, "HostBuffer", lambda *a, **k: None):
        dynamic_base = comfy.model_patcher.ModelPatcherDynamic(TinyH3(), torch.device("cuda:0"), cpu)
        dynamic_selected, = node.configure(dynamic_base, package.LEGACY_MODE)
        dynamic_clone = dynamic_selected.clone()
        delegate = dynamic_selected.clone(disable_dynamic=True,
                                         model_override=dynamic_selected.get_clone_model_override())
        assert dynamic_clone.is_dynamic() and not delegate.is_dynamic()
        assert dynamic_clone.get_attachment(package._KEY) is not dynamic_selected.get_attachment(package._KEY)
        assert all(name in delegate.__dict__ for name in package._METHODS)
        delegate_native, = node.configure(delegate, package.NATIVE_MODE)
        assert all(name not in delegate_native.__dict__ for name in package._METHODS)
    results["dynamic_clone_and_non_dynamic_delegate_preserve_scope"] = "metadata only; unused Aimdo HostBuffer is mocked"

    # Do not silently erase a later extension's replacement during cloning.
    conflicting = selected.clone()
    saved_load = conflicting.load
    conflicting.load = lambda *a, **k: None
    try:
        conflicting.clone()
        raise AssertionError("Changed model-local method should fail")
    except RuntimeError as exc:
        assert "Another extension replaced" in str(exc)
    finally:
        conflicting.load = saved_load
    results["later_instance_method_conflict_rejected"] = True

    # Native cannot require the private binary; legacy installation errors survive.
    def missing():
        raise RuntimeError("test missing private runtime")
    available = fake.ensure_available
    fake.ensure_available = missing
    try:
        native_again, = node.configure(selected, package.NATIVE_MODE)
        assert not native_again.get_attachment(package._KEY)
        try:
            node.configure(base, package.LEGACY_MODE)
            raise AssertionError("Unavailable legacy runtime should fail")
        except RuntimeError as exc:
            assert "missing private runtime" in str(exc)
    finally:
        fake.ensure_available = available
    assert tuple(registry._priority) == priority and set(registry._disabled) == disabled
    results["native_works_without_payload_legacy_failure_is_explicit"] = True
    results["global_priority_and_disabled_backends_unchanged"] = True
    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfy-root", default=os.environ.get("COMFYUI_ROOT"))
    args = parser.parse_args()
    if not args.comfy_root:
        parser.error("Supply --comfy-root or COMFYUI_ROOT")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sys.path.insert(0, str(Path(args.comfy_root).resolve()))
    sys.argv = [sys.argv[0], "--cpu"]
    run_tests()
