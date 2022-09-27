import numpy as np
import torch
from sklearn import metrics


class Evaluator:
    def __init__(self):
        self.y_true = []
        self.y_pred = []

    def evaluate(self, y_true, y_pred):
        mask = y_true > -0.9
        y_true = y_true[mask]
        y_pred = torch.sigmoid(y_pred[mask])
        self.y_true.extend(y_true.cpu().tolist())
        self.y_pred.extend(y_pred.cpu().tolist())

    def report(self):
        return {
            "acc": metrics.accuracy_score(
                self.y_true, np.asarray(self.y_pred, dtype=int)
            ),
            "auc": metrics.roc_auc_score(self.y_true, self.y_pred),
        }
