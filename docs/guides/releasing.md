# Release Guide

## GitHub Release

The release workflow is triggered by a published GitHub Release or can be started manually from the Actions page. It builds the wheel and sdist, publishes to PyPI with OIDC trusted publishing, and uploads the distributions to the GitHub Release.

## PyPI trusted publishing

Before publishing the first release, create a pending publisher at PyPI for the project name `pytorch-instance-segmentation-lab`:

- Owner: `Doithoo`
- Repository: `pytorch-instance-segmentation-lab`
- Workflow: `.github/workflows/publish.yml`
- Environment: `pypi`

The GitHub workflow already requests `id-token: write` and uses `environment: pypi`. The OIDC claims must match the repository, exact workflow path, and environment. A `422 invalid-publisher` response means this PyPI-side entry is missing or differs from those values.

After saving the pending publisher, rerun the failed workflow:

```bash
gh run rerun 32578729496 --failed
# Or run the Publish workflow manually from GitHub Actions.
```

For a new release, create the tag, push it, and publish the GitHub Release. Do not put a PyPI API token in repository secrets when trusted publishing is available.

## Release assets

For the protocol-v2 baseline, upload the wheel, sdist, model card, trusted `best.pt`, and a checksum file. Verify the checkpoint before upload and verify all downloaded assets after upload:

```bash
sha256sum -c SHA256SUMS-v0.2.0.txt
```

The checkpoint is a trusted pickle-based PyTorch artifact. Publish its SHA-256 beside the file and keep the model card with intended-use and limitation details.
