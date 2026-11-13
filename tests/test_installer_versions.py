"""CPU-only installer regression checks; never import the plugin or torch.

Run with --plugin-root PATH --wheel-035 PATH [--other-wheel-035 PATH].
Use --reference-035 PATH to verify an existing bundle remains unchanged.
"""
from __future__ import annotations

import argparse
from collections import namedtuple
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch


def tree_hashes(root):
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


def run(plugin_root, wheel_035, other_wheel_035=None, reference_035=None):
    spec = importlib.util.spec_from_file_location("isolated_h3_installer", plugin_root / "install.py")
    installer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(installer)
    manifest = installer.load_manifest()
    checks = []
    assert "torch" not in sys.modules, "Installer must not initialize PyTorch/GPU"
    assert len(manifest["wheels"]) == 2
    assert {installer.asset_version(a) for a in manifest["wheels"]} == {"0.2.35"}
    assert manifest["versions"] == ["0.2.35"]
    assert set(manifest["sources"]) == {"0.2.35"}
    checks.append("only two official Kitchen 0.2.35 assets: Linux and Windows")

    Version = namedtuple("Version", "major minor micro releaselevel serial")
    for system, machine, pyminor, libc, allowed in [
        ("Linux", "x86_64", 12, ("glibc", "2.35"), True),
        ("Windows", "AMD64", 13, ("", ""), True),
        ("Linux", "x86_64", 11, ("glibc", "2.35"), False),
        ("Linux", "aarch64", 12, ("glibc", "2.35"), False),
        ("Linux", "x86_64", 12, ("glibc", "2.26"), False),
        ("Darwin", "arm64", 12, ("", ""), False),
    ]:
        with patch.object(installer.platform, "system", return_value=system), \
             patch.object(installer.platform, "machine", return_value=machine), \
             patch.object(installer.platform, "libc_ver", return_value=libc), \
             patch.object(installer.sys, "version_info", Version(3, pyminor, 0, "final", 0)):
            for version in installer.SUPPORTED_VERSIONS:
                try:
                    asset = installer.select_wheel(manifest, version)
                except RuntimeError:
                    assert not allowed, (system, machine, pyminor, version)
                else:
                    assert allowed and installer.asset_version(asset) == version
    with patch.object(installer.sysconfig, "get_config_var", return_value=1):
        try:
            installer.select_wheel(manifest, "0.2.35")
        except RuntimeError:
            pass
        else:
            raise AssertionError("Free-threaded Python accepted")
    checks.append("Windows/Linux ABI selection and unsupported-platform/GIL rejection")

    assets = [next(a for a in manifest["wheels"] if a["filename"] == wheel.name)
              for wheel in (wheel_035, other_wheel_035) if wheel]
    assert len({a["filename"] for a in assets}) == len(assets)
    reference_before = tree_hashes(reference_035) if reference_035 else None
    with tempfile.TemporaryDirectory(prefix="h3-installer-regression-") as temporary:
        temp_root = Path(temporary)
        with patch.object(installer, "ROOT", temp_root), \
             patch.object(installer, "download", side_effect=AssertionError("Unexpected network download")) as download:
            # Removed versions fail before a download or installation is attempted.
            for version_call in (
                lambda: installer.vendor_dir("0.2.34"),
                lambda: installer.select_wheel(manifest, "0.2.34"),
                lambda: installer.asset_version(dict(assets[0], version="0.2.34")),
                lambda: installer.main(["--wheel-file", "comfy_kitchen-0.2.34-cp312-abi3-win_amd64.whl"]),
                lambda: installer.main(["--bundle-wheel", "--wheel-file", "comfy_kitchen-0.2.34-cp312-abi3-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"]),
            ):
                try:
                    version_call()
                except RuntimeError:
                    pass
                else:
                    raise AssertionError("Removed Kitchen 0.2.34 accepted")
            with redirect_stderr(io.StringIO()):
                try:
                    installer.main(["--version", "0.2.34"])
                except SystemExit as error:
                    assert error.code == 2
                else:
                    raise AssertionError("Removed --version 0.2.34 accepted")
            assert not list(temp_root.iterdir())
            download.assert_not_called()
            checks.append("0.2.34 CLI/API/offline wheels rejected before downloads or writes")

            for wheel, asset in zip((wheel_035, other_wheel_035), assets):
                assert not installer.installed_and_verified(asset)
                # Offline --wheel-file and --bundle-wheel retain their semantics.
                before = tree_hashes(installer.vendor_dir("0.2.35"))
                installer.main(["--bundle-wheel", "--wheel-file", str(wheel)])
                assert installer.installed_and_verified(asset)
                after = tree_hashes(installer.vendor_dir("0.2.35"))
                assert all(after[name] == digest for name, digest in before.items()
                           if name != "provenance.json")
                installer.main(["--bundle-wheel", "--wheel-file", str(wheel), "--version", "0.2.35"])
                assert tree_hashes(installer.vendor_dir("0.2.35")) == after
                # Exercise the default, explicit and backward-compatible 'all' checks.
                system = "Windows" if "win_amd64" in wheel.name else "Linux"
                with patch.object(installer.platform, "system", return_value=system), \
                     patch.object(installer.platform, "machine", return_value="AMD64" if system == "Windows" else "x86_64"), \
                     patch.object(installer.platform, "libc_ver", return_value=("glibc", "2.35")):
                    installer.main(["--check"])
                    installer.main(["--version", "0.2.35", "--check"])
                    installer.main(["--version", "all", "--check"])
                    installer.main(["--wheel-file", str(wheel), "--check"])
            try:
                installer.main(["--bundle-wheel", "--wheel-file", str(wheel_035), "--version", "all"])
            except RuntimeError:
                pass
            else:
                raise AssertionError("Ambiguous all-plus-single-wheel selection accepted")
            download.assert_not_called()
            assert not (temp_root / "vendor" / "comfy_kitchen_034").exists()
            assert all(installer.installed_and_verified(a) for a in assets)
            checks.append("offline CLI, --check, --version 0.2.35/all, idempotence and platform coexistence")
            before_035 = tree_hashes(installer.vendor_dir("0.2.35"))
            try:
                installer.extract_verified_wheel(wheel_035, dict(assets[0], sha256="0" * 64))
            except RuntimeError:
                pass
            else:
                raise AssertionError("Wrong wheel digest accepted")
            assert tree_hashes(installer.vendor_dir("0.2.35")) == before_035
            provenance = installer.read_provenance("0.2.35")
            entry = provenance["artifacts"][assets[0]["filename"]]
            native = installer.vendor_dir("0.2.35") / entry["native"]
            with native.open("r+b") as handle:
                handle.seek(100)
                previous = handle.read(1)
                handle.seek(100)
                handle.write(bytes([previous[0] ^ 1]))
            assert not installer.installed_and_verified(assets[0])
            checks.append("wrong wheel hash rejected before writes; corrupted native detected")
    if reference_035:
        assert tree_hashes(reference_035) == reference_before
        checks.append("every existing vendor/comfy_kitchen_035 file unchanged")
    return {"passed": True, "checks": checks, "python": sys.version.split()[0],
            "host_platform": sys.platform, "versions": ["0.2.35"],
            "official_assets_verified": len(assets),
            "wheels": [asset["filename"] for asset in assets],
            "preserved_035_files": len(reference_before) if reference_before is not None else None,
            "torch_imported": "torch" in sys.modules, "gpu_or_model_loaded": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plugin-root", type=Path, required=True)
    parser.add_argument("--wheel-035", type=Path, required=True)
    parser.add_argument("--other-wheel-035", type=Path)
    parser.add_argument("--reference-035", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with redirect_stdout(io.StringIO()):
        result = run(args.plugin_root.resolve(), args.wheel_035.resolve(),
                     args.other_wheel_035.resolve() if args.other_wheel_035 else None,
                     args.reference_035.resolve() if args.reference_035 else None)
    data = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(data + "\n", encoding="utf-8")
    print(data)
