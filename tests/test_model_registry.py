from __future__ import annotations

from instance_segmenter.models.registry import build_model, get_model_spec, list_models


def test_registry_exposes_stable_maskrcnn_names() -> None:
    assert "maskrcnn_resnet50_fpn" in list_models()
    assert get_model_spec("maskrcnn_resnet50_fpn").supported_weights == ("none", "coco_v1")


def test_lightweight_maskrcnn_replaces_both_heads() -> None:
    model = build_model("maskrcnn_mobilenet_v3_large", 3, "none", {"min_size": 64, "max_size": 64})
    assert model.roi_heads.box_predictor.cls_score.out_features == 3
    assert model.roi_heads.mask_predictor.mask_fcn_logits.out_channels == 3


def test_maskrcnn_replaces_both_predictors_for_label_schema() -> None:
    model = build_model("maskrcnn_resnet50_fpn", 2, "none", {"min_size": 64, "max_size": 64})
    assert model.roi_heads.box_predictor.cls_score.out_features == 2
    assert model.roi_heads.mask_predictor.mask_fcn_logits.out_channels == 2


def test_model_registry_rejects_unknown_weight_policy() -> None:
    try:
        build_model("maskrcnn_resnet50_fpn", 2, "imagenet")
    except ValueError as error:
        assert "supports weights" in str(error)
    else:
        raise AssertionError("expected unsupported weight policy to fail")
