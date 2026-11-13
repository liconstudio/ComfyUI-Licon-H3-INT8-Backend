# Release preparation

This repository distributes the plugin source. GitHub uploads do not publish a Registry version. The Registry workflow runs only when manually dispatched.

| Identity | Value |
| --- | --- |
| GitHub owner | `liconstudio` |
| Repository/folder | `ComfyUI-Licon-H3-INT8-Backend` |
| Repository URL | `https://github.com/liconstudio/ComfyUI-Licon-H3-INT8-Backend` |
| Registry package ID | `licon-h3-int8-backend` |
| Node type (workflow compatibility) | `H3Int8Backend` |
| Version | `0.2.2` |
| Registry PublisherId | Not assigned; fill the actual registered ID before publishing |

## GitHub source release

Keep the contents of the source folder at the repository root: `__init__.py`, `pyproject.toml` and `install.py` must be at the top level. Include hidden files such as `.gitignore`, `.comfyignore` and `.github/`.

Do not upload the live installation directory wholesale. `vendor/` is generated from verified official wheels and excluded from Git. Keep LICENSE, UPSTREAM_LICENSE, UPSTREAM_NOTICE, THIRD_PARTY_NOTICES.md, wheel-manifest.json and the legacy runtime's source/provenance manifests. Do not add environments, caches, model weights, private workflows or generated media.

The source package needs `install.py` to obtain the platform binary. Optionally provide a separately labeled preinstalled ZIP as a GitHub Release attachment, with only verified native artifacts and their original licenses/metadata. State the platforms included and that Windows GPU execution has not been validated. This repository does not need a PyPI package or `setup.py` to load in ComfyUI.

## Registry and Manager

GitHub upload alone does not publish a Registry version. Before the first publication:

1. Create a publisher at https://registry.comfy.org and copy its actual ID to `[tool.comfy].PublisherId`. A GitHub username does not automatically establish a Registry publisher.
2. Confirm the public Repository URL, the globally unique package ID and version in `pyproject.toml`. Package and publisher IDs cannot be renamed after creation; versions cannot be overwritten.
3. Create a Registry publishing API key. Store it in the GitHub repository's Actions secret named `REGISTRY_ACCESS_TOKEN`, never in source files or documentation.
4. Run the installer CI and the CPU model-scope test documented in CONTRIBUTING.md. Review `.comfyignore`; it keeps development scripts/tests out of the Registry archive while preserving the runtime, installer and all required notices.
5. When ready, manually run the **Publish to Comfy Registry** workflow, or run `comfy node publish` locally using Comfy CLI. Review the Registry security scan; this plugin downloads an official native wheel at installation time, and passing that review has not yet been established.
6. Verify a clean Manager installation, run `install.py --check`, confirm the node and test a real H3 workflow. Publishing to Registry does not claim acceptance into the separate legacy `custom-node-list.json` list. If that legacy channel is required, it needs its own Manager submission.

Later updates require a new version and another publication. The included workflow is manual to keep source pushes separate from releases; it can be changed to a version-file or release trigger after the first validated release.

## Scope and optional improvements

Independent means self-contained plugin source, not dependency-free or universally portable. It uses existing ComfyUI/Torch/Kitchen and obtains a pinned private Kitchen extension. It does not require another custom node. Tested versions and Windows limitations are in README.md.

The sample workflow is included as `sample-workflow.json` in the repository root; users supply their own model weights and images. Optional improvements include screenshots, an icon/banner, Windows GPU validation and testing on additional ComfyUI versions. README.md reflects the author's latest INT8 quality comparisons; dated technical evidence remains in VALIDATION.md.

## Official references

- [Metadata and naming](https://docs.comfy.org/registry/specifications)
- [Publishing and .comfyignore](https://docs.comfy.org/registry/publishing)
- [Registry scanning and immutable versions](https://docs.comfy.org/registry/overview)
- [Registry security standards](https://docs.comfy.org/registry/standards)
- [Manager install hooks and legacy listing](https://docs.comfy.org/custom-nodes/backend/manager)
- [Apache-2.0 redistribution conditions](https://www.apache.org/licenses/LICENSE-2.0)
