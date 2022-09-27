from ..eval import Evaluator

import torch


def test_evaluator():
    y_true = torch.tensor([[0, 1, -1]])
    y_pred = torch.tensor([[0.1, 0.6, 0.3]])

    evaluator = Evaluator()
    evaluator.evaluate(y_true, y_pred)
    metrics = evaluator.report()
    assert metrics["acc"] == 0.5
    assert metrics["auc"] == 1.0
