# Security Policy

## Reporting a vulnerability

Report suspected vulnerabilities through [GitHub private vulnerability reporting](https://github.com/Doithoo/pytorch-instance-segmentation-lab/security/advisories/new). Do not disclose an unpatched vulnerability in a public issue.

Include the affected version, reproduction steps, impact, and any proposed mitigation. The maintainer will acknowledge a complete report within seven days and coordinate disclosure after a fix is available.

## Trusted inputs

PyTorch resume checkpoints are pickle-based. This project calls `torch.load(..., weights_only=False)` because checkpoints include optimizer, scheduler, configuration, and RNG state. Load `.pt` files only from a trusted source and verify published SHA-256 hashes before use.

Dataset archives and committed manifests are checksum-verified. External model and dataset factories execute Python code and must be treated as trusted plugins.
