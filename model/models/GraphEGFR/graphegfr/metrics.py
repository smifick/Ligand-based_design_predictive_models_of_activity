import torch
import torch.nn.functional as F
import numpy as np
from torchmetrics.functional.regression import pearson_corrcoef, mean_absolute_error, mean_squared_error, spearman_corrcoef, r2_score
from imblearn.metrics import geometric_mean_score
from abc import ABC

class BasicCriterion(ABC):

    def __init__(self):
        super().__init__()
        self.name = None

    def __call__(self,answer, label):
        return round(self.compute(answer, label),4)
    
    def __len__(self):
        return 1

class RMSE(BasicCriterion):
    def __init__(self):
        super(RMSE, self).__init__()
        self.name = 'RMSE'

    def compute(self, answer, label):
        assert len(answer) == len(label)
        answer = torch.Tensor(answer).squeeze(-1)
        label = torch.Tensor(label)
        RMSE = mean_squared_error(answer, label).sqrt()
        return RMSE.item()


class MAE(BasicCriterion):
    def __init__(self):
        super(MAE, self).__init__()
        self.name = 'MAE'

    def compute(self, answer, label):
        assert len(answer) == len(label)
        answer = torch.Tensor(answer)
        label = torch.Tensor(label)
        #print("Size for MAE")
        MAE = mean_absolute_error(answer, label)
        return MAE.item()


class MSE(BasicCriterion):
    def __init__(self):
        super(MSE, self).__init__()
        self.name = 'MSE'

    def compute(self, answer, label):
        assert len(answer) == len(label)
        answer = torch.Tensor(answer)
        label = torch.Tensor(label)
        #print("Size for MAE")
        MSE = mean_squared_error(answer, label)
        return MSE.item()


class PCC(BasicCriterion):
    def __init__(self):
        super(PCC, self).__init__()
        self.name = 'PCC'

    def compute(self, answer, label):
        assert len(answer) == len(label)
        answer = torch.Tensor(answer)
        label = torch.Tensor(label)
        #print("Size for MAE")
        pcc = pearson_corrcoef(answer, label)
        return pcc.item()


class R2(BasicCriterion):
    def __init__(self):
        super(R2, self).__init__()
        self.name = 'R2'

    def compute(self, answer, label):
        assert len(answer) == len(label)
        answer = torch.Tensor(answer)
        label = torch.Tensor(label)
        #print("Size for MAE")
        r_squared = r2_score(answer, label)
        return r_squared.item()


class SRCC(BasicCriterion):
    def __init__(self):
        super(SRCC, self).__init__()
        self.name = 'Spearman Rank Cor. Coef.'

    def compute(self, answer, label):
        assert len(answer) == len(label)
        answer = torch.Tensor(answer)
        label = torch.Tensor(label)
        srcc = spearman_corrcoef(answer, label)
        return srcc.item()