"""Install the pinned Kitchen 0.2.35 CUDA extension privately, without pip.

ComfyUI-Manager calls this file with ComfyUI's Python after installation.
prestartup_script.py also calls main([]) when ComfyUI starts.
Only the Python standard library is required. No model or GPU is initialized.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import sys
import sysconfig
import tempfile
import urllib.parse
import urllib.request
import zipfile


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "wheel-manifest.json"
SUPPORTED_VERSIONS = ("0.2.35",)


def vendor_dir(version: str) -> Path:
    if version not in SUPPORTED_VERSIONS:
        raise RuntimeError(f"Unsupported private Kitchen version: {version}")
    return ROOT / "vendor" / ("comfy_kitchen_0" + version.rsplit(".", 1)[1])


def asset_version(asset: dict) -> str:
    # The filename check also accepts the original single-version manifest shape.
    filename_version = asset["filename"].split("-", 2)[1]
    version = asset.get("version", filename_version)
    if version not in SUPPORTED_VERSIONS or version != filename_version:
        raise RuntimeError("Pinned asset version and wheel filename do not agree.")
    return version


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest() -> dict:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("distribution") != "comfy-kitchen":
        raise RuntimeError("Unexpected legacy wheel manifest; expected comfy-kitchen.")
    versions = {asset_version(asset) for asset in manifest.get("wheels", [])}
    if versions != set(SUPPORTED_VERSIONS):
        raise RuntimeError("The release manifest must pin only Kitchen 0.2.35.")
    return manifest


def select_wheel(manifest: dict, version: str = "0.2.35") -> dict:
    if version not in SUPPORTED_VERSIONS:
        raise RuntimeError(f"Unsupported private Kitchen version: {version}")
    if sys.implementation.name != "cpython" or sys.version_info < (3, 12):
        raise RuntimeError("This plugin release requires CPython 3.12 or newer (abi3).")
    if sysconfig.get_config_var("Py_GIL_DISABLED"):
        raise RuntimeError("Free-threaded Python is not supported by these upstream wheels.")
    machine = platform.machine().lower()
    system = platform.system()
    if system == "Windows" and machine in ("amd64", "x86_64"):
        platform_tag = "win_amd64"
    elif system == "Linux" and machine in ("x86_64", "amd64"):
        platform_tag = "manylinux_2_27_x86_64.manylinux_2_28_x86_64"
        libc, libc_version = platform.libc_ver()
        minimum = (2, 27)
        try:
            found = tuple(int(part) for part in libc_version.split(".")[:2])
        except ValueError:
            found = ()
        if libc != "glibc" or found < minimum:
            raise RuntimeError(f"This upstream wheel needs glibc {'.'.join(map(str, minimum))}+; found {libc} {libc_version}.")
    else:
        raise RuntimeError(f"This release supports Linux x86_64 and Windows x64; found {system}/{machine}.")
    abi = "cp312-abi3"
    filename = f"comfy_kitchen-{version}-{abi}-{platform_tag}.whl"
    for asset in manifest["wheels"]:
        if asset["filename"] == filename:
            return asset
    raise RuntimeError(f"No pinned upstream wheel matches {filename}.")


def read_provenance(version: str = "0.2.35") -> dict:
    path = vendor_dir(version) / "provenance.json"
    if path.exists():
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if value.get("version") == version and isinstance(value.get("artifacts"), dict):
                return value
        except (ValueError, OSError):
            pass
    return {"schema": 1, "distribution": "comfy-kitchen", "version": version, "artifacts": {}}


def installed_and_verified(asset: dict) -> bool:
    version = asset_version(asset)
    vendor = vendor_dir(version)
    entry = read_provenance(version)["artifacts"].get(asset["filename"], {})
    if entry.get("wheel_sha256") != asset["sha256"] or not entry.get("files"):
        return False
    native = entry.get("native", "")
    if not native.startswith("comfy_kitchen/backends/cuda/_C.") or native not in entry["files"]:
        return False
    known_native = asset.get("known_native_files", {}).get(native)
    if known_native and entry["files"][native] != known_native["sha256"]:
        return False
    for name, digest in entry["files"].items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts:
            return False
        path = vendor.joinpath(*relative.parts)
        if not path.is_file() or sha256_file(path) != digest:
            return False
    return True


def download(asset: dict, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / asset["filename"]
    if target.exists() and sha256_file(target) == asset["sha256"]:
        return target
    parsed = urllib.parse.urlparse(asset["url"])
    if parsed.scheme != "https" or parsed.hostname != "files.pythonhosted.org":
        raise RuntimeError("Pinned wheel URL must use HTTPS on files.pythonhosted.org.")
    print(f"[H3 INT8 Backend] Downloading official {asset['filename']} ({asset['size']:,} bytes)")
    descriptor, temporary_name = tempfile.mkstemp(prefix="h3-int8-download-", suffix=".whl", dir=cache_dir)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output, urllib.request.urlopen(asset["url"], timeout=60) as response:
            while block := response.read(1024 * 1024):
                output.write(block)
        if temporary.stat().st_size != asset["size"] or sha256_file(temporary) != asset["sha256"]:
            raise RuntimeError("Official legacy wheel size or SHA-256 verification failed.")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".h3-int8-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(data)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def extract_verified_wheel(wheel: Path, asset: dict) -> dict:
    version = asset_version(asset)
    vendor = vendor_dir(version)
    if wheel.stat().st_size != asset["size"] or sha256_file(wheel) != asset["sha256"]:
        raise RuntimeError(f"Wheel does not match the pinned official SHA-256: {wheel.name}")
    cuda_prefix = "comfy_kitchen/backends/cuda/"
    info_prefix = f"comfy_kitchen-{version}.dist-info/"
    with zipfile.ZipFile(wheel) as archive:
        natives = [name for name in archive.namelist() if name.startswith(cuda_prefix + "_C.") and name.endswith((".so", ".pyd"))]
        if len(natives) != 1:
            raise RuntimeError("Expected exactly one upstream CUDA extension in this wheel.")
        required = [info_prefix + "licenses/LICENSE", info_prefix + "licenses/NOTICE", info_prefix + "METADATA", info_prefix + "WHEEL"]
        names = natives + required
        files = {}
        # Validate every selected member before changing the existing installation.
        payloads = []
        for name in names:
            relative = PurePosixPath(name)
            if relative.is_absolute() or ".." in relative.parts or "\\" in name:
                raise RuntimeError("Unsafe member path in pinned wheel.")
            payload = archive.read(name)
            digest = hashlib.sha256(payload).hexdigest()
            known = asset.get("known_native_files", {}).get(name)
            if known and digest != known["sha256"]:
                raise RuntimeError("Legacy native extension does not match its pinned hash.")
            # Platform wheels can contain different license line endings/metadata.
            # Preserve each wheel's originals without invalidating another platform.
            installed_name = name if name in natives else f"artifacts/{asset['sha256'][:16]}/{name}"
            files[installed_name] = digest
            payloads.append((vendor.joinpath(*PurePosixPath(installed_name).parts), payload))
        for destination, payload in payloads:
            atomic_write(destination, payload)
    provenance = read_provenance(version)
    provenance["artifacts"][asset["filename"]] = {
        "source_url": asset["url"], "wheel_sha256": asset["sha256"],
        "wheel_size": asset["size"], "native": natives[0], "files": files,
    }
    atomic_write(vendor / "provenance.json", (json.dumps(provenance, indent=2) + "\n").encode("utf-8"))
    return provenance["artifacts"][asset["filename"]]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel-file", type=Path, help="Use an already downloaded, pinned official wheel; no network.")
    parser.add_argument("--bundle-wheel", action="store_true", help="Release preparation only: allow a pinned wheel for another platform to be bundled alongside this platform.")
    parser.add_argument("--cache-dir", type=Path, default=Path(tempfile.gettempdir()) / "h3-int8-backend-wheel-cache")
    parser.add_argument("--version", choices=(*SUPPORTED_VERSIONS, "all"), help="Select the private Kitchen version. Default: 0.2.35; 'all' is a compatibility alias for 0.2.35.")
    parser.add_argument("--check", action="store_true", help="Verify this Python/platform's private Kitchen 0.2.35 installation without downloading.")
    options = parser.parse_args(argv)
    manifest = load_manifest()
    if options.bundle_wheel and not options.wheel_file:
        raise RuntimeError("--bundle-wheel requires --wheel-file.")
    if options.wheel_file:
        matches = [item for item in manifest["wheels"] if item["filename"] == options.wheel_file.name]
        if len(matches) != 1:
            if options.wheel_file.name.startswith("comfy_kitchen-0.2.34-"):
                raise RuntimeError("Kitchen 0.2.34 support has been removed; use the pinned Kitchen 0.2.35 wheel.")
            raise RuntimeError("The supplied wheel is absent from the pinned manifest.")
        asset = matches[0]
        version = asset_version(asset)
        if options.version not in (None, version):
            raise RuntimeError("--wheel-file installs one version; --version must be omitted or match that wheel.")
        if not options.bundle_wheel and select_wheel(manifest, version)["filename"] != asset["filename"]:
            raise RuntimeError(f"This wheel does not match the current Python/platform: {asset['filename']}.")
        assets = [asset]
    else:
        versions = SUPPORTED_VERSIONS if options.version in (None, "all") else (options.version,)
        assets = [select_wheel(manifest, version) for version in versions]
    for asset in assets:
        if installed_and_verified(asset):
            print(f"[H3 INT8 Backend] Private legacy extension verified: {asset['filename']}")
            continue
        if options.check:
            raise RuntimeError(f"A verified Kitchen {asset_version(asset)} extension is not installed for this Python/platform.")
        wheel = options.wheel_file if options.wheel_file else download(asset, options.cache_dir)
        entry = extract_verified_wheel(wheel, asset)
        print(f"[H3 INT8 Backend] Installed private Kitchen {asset_version(asset)} extension: {entry['native']}")
    print("[H3 INT8 Backend] Existing PyTorch, comfy-kitchen, startup flags and model files were not changed.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"[H3 INT8 Backend] INSTALL FAILED: {error}", file=sys.stderr)
        print("Legacy mode will report an error; it will not silently use Native or Reference. Check your Python/platform and network, then rerun install.py with ComfyUI's Python.", file=sys.stderr)
        raise SystemExit(1)
