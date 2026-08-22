# Release Guide

## GitHub Release

The release-assets workflow runs when a GitHub Release is published and can also be started manually for an existing release tag. It builds the wheel and sdist, then uploads both files to the GitHub Release. This project does not publish to PyPI.

For a manual run, open the workflow in GitHub Actions and provide the existing release tag in the `tag` input:

```bash
gh workflow run release.yml -f tag=v0.2.0
```

For a new release, create the tag, push it, and publish the GitHub Release. The published-release event supplies the tag automatically. The workflow uses only `contents: write`; it does not create a PyPI environment or request OIDC publishing permissions.

## Release Assets

For the protocol-v2 baseline, upload the wheel, sdist, model card, trusted `best.pt`, and a checksum file. Verify the checkpoint before upload and verify all downloaded assets after upload:

```bash
sha256sum -c SHA256SUMS-v0.2.0.txt
```

The checkpoint is a trusted pickle-based PyTorch artifact. Publish its SHA-256 beside the file and keep the model card with intended-use and limitation details.

The package can still be built locally without publishing it:

```bash
uv sync --locked --extra dev
uv build
uv run twine check dist/*
```
