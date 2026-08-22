# Changelog

All notable changes follow [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and semantic versioning.

## [Unreleased]

## [0.2.0] - 2026-08-22

### Changed

- Corrected COCO AP evaluation to preserve the full confidence ranking.
- Replaced lexicographic Penn-Fudan splits with deterministic source-stratified manifests.
- Added strict resume, manifest, run-comparison, provenance, and final-epoch evaluation contracts.
- Published the completed protocol-v2 Kaggle baseline while preserving protocol-v1 artifacts under `legacy-v1/`.

### Added

- Built-in COCO polygon/RLE dataset support, multiclass and empty-image handling.
- MobileNetV3-Large Mask R-CNN model with an inspectable custom backbone configuration.
- Ranked worst-case overlays and `doctor`, `init-config`, `prepare-coco`, and `compare-runs` commands.
- Coverage enforcement, pinned CI actions, citation, conduct, and expanded security guidance.

## [0.1.0] - 2026-08-22

### Added

- Initial Penn-Fudan, Mask R-CNN, Kaggle runner, bilingual documentation, and protocol-v1 recorded run.

[Unreleased]: https://github.com/Doithoo/pytorch-instance-segmentation-lab/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/Doithoo/pytorch-instance-segmentation-lab/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Doithoo/pytorch-instance-segmentation-lab/releases/tag/v0.1.0
