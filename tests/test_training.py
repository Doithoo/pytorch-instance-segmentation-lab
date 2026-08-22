from __future__ import annotations

import torch

from instance_segmenter.data.collate import instance_collate
from instance_segmenter.training.trainer import dry_run, train_one_epoch
from tests.fixtures.external_models import ContractInstanceModel
from tests.fixtures.synthetic_instances import sample_image_and_target


def test_one_epoch_and_dry_run_update_a_contract_model() -> None:
    image, target = sample_image_and_target()
    batch = instance_collate([(image, target)])
    model = ContractInstanceModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    before = float(model.scale.detach())
    losses = train_one_epoch(model, [batch], optimizer, torch.device("cpu"), amp=False, grad_clip_norm=None)
    assert losses["loss_total"] > 0
    assert float(model.scale.detach()) != before
    diagnostics = dry_run(model, [batch], optimizer, torch.device("cpu"), amp=False, grad_clip_norm=1.0)
    assert diagnostics.batch_size == 1
    assert diagnostics.target_counts == (2,)
    assert {"loss_classifier", "loss_box_reg", "loss_mask", "loss_objectness", "loss_rpn_box_reg"} <= set(
        diagnostics.losses
    )
