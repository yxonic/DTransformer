from ..eval import Evaluator

import torch


def test_evaluator():
    y_true = torch.tensor([[0, 1, -1]])
    y_pred = torch.tensor([[0.1, 0.6, 0.3]])

    evaluator = Evaluator()
    evaluator.evaluate(y_true, y_pred)
    metrics = evaluator.report()
    assert metrics["acc"] == 1.0
    assert metrics["auc"] == 1.0
    assert round(metrics["mae"], 2) == 0.25
    assert round(metrics["rmse"], 2) == 0.29
