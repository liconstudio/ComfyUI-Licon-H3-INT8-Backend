"""Run the existing offline installer regression against verified official wheels."""
import ast
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import tomllib

ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    for path in ROOT.rglob('*.py'):
        if not set(path.relative_to(ROOT).parts) & {'vendor', '.venv'}:
            ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    with (ROOT / 'pyproject.toml').open('rb') as handle:
        metadata = tomllib.load(handle)
    print('Package:', metadata['project']['name'], metadata['project']['version'])
    installer = load('licon_installer_ci', ROOT / 'install.py')
    suite = load('licon_installer_regression', ROOT / 'tests' / 'test_installer_versions.py')
    startup = load('licon_startup_regression', ROOT / 'tests' / 'test_prestartup.py')
    with tempfile.TemporaryDirectory(prefix='licon-installer-ci-') as temporary:
        wheels = [installer.download(asset, Path(temporary))
                  for asset in installer.load_manifest()['wheels']]
        result = suite.run(ROOT, wheels[0], wheels[1])
        result['prestartup'] = startup.run(ROOT, wheels)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
