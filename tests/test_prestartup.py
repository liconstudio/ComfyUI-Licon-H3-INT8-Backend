"""Exercise ComfyUI's startup hook with real pinned wheels and no network/GPU."""
from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
from pathlib import Path
import shutil
import sys
import tempfile
from unittest.mock import patch
import urllib.request


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(plugin_root, wheels):
    checks = []
    with tempfile.TemporaryDirectory(prefix="licon-prestartup-") as temporary:
        root = Path(temporary) / "custom_nodes" / "plugin with spaces"
        root.mkdir(parents=True)
        for name in ("prestartup_script.py", "install.py", "wheel-manifest.json"):
            shutil.copy2(plugin_root / name, root / name)
        installer = load("startup_test_installer", root / "install.py")
        asset = installer.select_wheel(installer.load_manifest())
        wheel = next(p for p in wheels if p.name == asset["filename"])

        def start():
            output, errors = io.StringIO(), io.StringIO()
            arguments = ["ComfyUI/main.py", "--listen", "0.0.0.0", "--port", "8187"]
            with patch.object(sys, "argv", arguments.copy()), \
                 redirect_stdout(output), redirect_stderr(errors):
                # Like ComfyUI, load as a module, not as __main__ or a package.
                load("comfy_prestartup_test", root / "prestartup_script.py")
                assert sys.argv == arguments
            assert "torch" not in sys.modules
            return output.getvalue(), errors.getvalue()

        with patch.object(tempfile, "gettempdir", return_value=temporary):
            with patch.object(urllib.request, "urlopen", side_effect=OSError("offline")) as request:
                _, errors = start()
                request.assert_called_once()
                assert "Automatic setup failed" in errors and "offline" in errors
                assert "ComfyUI will continue" in errors
                assert not installer.installed_and_verified(asset)
            checks.append("failed first download reports recovery and does not abort startup")

            with patch.object(urllib.request, "urlopen", side_effect=lambda *a, **kw: wheel.open("rb")) as request:
                output, errors = start()
                request.assert_called_once_with(asset["url"], timeout=60)
                assert not errors and "Installed private Kitchen" in output
                assert installer.installed_and_verified(asset)
            checks.append("next startup installs a verified platform-matching official wheel")

            before = {p: p.stat().st_mtime_ns for p in (root / "vendor").rglob("*") if p.is_file()}
            with patch.object(urllib.request, "urlopen", side_effect=AssertionError("Unexpected download")) as request:
                output, errors = start()
                request.assert_not_called()
                assert not errors and "Private legacy extension verified" in output
                assert all(p.stat().st_mtime_ns == timestamp for p, timestamp in before.items())
            checks.append("repeat startup uses verified files without downloads or rewrites")

            entry = installer.read_provenance()["artifacts"][asset["filename"]]
            native = installer.vendor_dir("0.2.35") / entry["native"]
            with native.open("r+b") as handle:
                handle.seek(100)
                byte = handle.read(1)
                handle.seek(100)
                handle.write(bytes([byte[0] ^ 1]))
            assert not installer.installed_and_verified(asset)
            with patch.object(urllib.request, "urlopen", side_effect=AssertionError("Unexpected download")) as request:
                output, errors = start()
                request.assert_not_called()
                assert not errors and installer.installed_and_verified(asset)
            checks.append("corrupted native repaired from a verified cached wheel without network")
    return {"passed": True, "checks": checks, "python": sys.version.split()[0],
            "host_platform": sys.platform, "torch_imported": "torch" in sys.modules}
