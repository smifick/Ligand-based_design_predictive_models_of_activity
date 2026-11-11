import torch
import pickle as pkl
import random
import numpy as np
import pandas as pd
import os
from itertools import product
import re
import json
import errno
from graphegfr.configs import Configs
from typing import Tuple
"""
NOTE
configs param
n_tasks
max_epoch
REPEAT
"""
def compute_confidence_interval(data):
    """
    Compute 95% confidence interval
    :param data: An array of mean accuracy (or mAP) across a number of sampled episodes.
    :return: the 95% confidence interval for this data.
    """
    a = 1.0 * np.array(data)
    m = np.mean(a)
    std = np.std(a)
    pm = 1.96 * (std / np.sqrt(len(a)))
    return round(m, 4), round(pm, 4)

def set_seed(new_seed=None):
    if new_seed is None:
        new_seed = random.randrange(1000)
    torch.manual_seed(new_seed)
    np.random.seed(new_seed)
    return new_seed

def set_split_seed(new_seed=None):
    if new_seed is None:
        new_seed = random.randrange(1000)
    return new_seed

def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pkl.load(f)

def load_tensor(file_name, dtype, device):
    return [dtype(d).to(device) for d in np.load(file_name + '.npy')]

def isnan(x):
    return x!=x

def calculate_score(pred, label, criterion, n_tasks:int):
    squeeze_total_pred=np.array([])
    squeeze_total_label=np.array([])
    score_list = []
    for i in range(pred.shape[1]):
        task_label = label[:,i]
        task_pred = pred[:,i]
        task_masks = ~np.isnan(task_label)
        masked_label = np.atleast_1d(np.squeeze(task_label[task_masks]))
        masked_pred = np.atleast_1d(np.squeeze(task_pred[task_masks]))
        squeeze_total_label = np.concatenate((squeeze_total_label,masked_label))
        squeeze_total_pred = np.concatenate((squeeze_total_pred,masked_pred))
        score_list.append(float(criterion(masked_label,masked_pred)))
    if n_tasks > 1:
        score_list.insert(0,float(criterion(squeeze_total_label,squeeze_total_pred)))
    return score_list

class ResultsReport(object):
    def __init__(self, configs: Configs):
        assert all(param is not None for param in (configs['target_list'], configs['metrics'])) 
        self._metrics = configs['metrics']
        self._metrics_name = [criterion.name for criterion in configs['metrics']]
        self._target = configs['target_list']
        self._run_dir = configs['run_dir']
        self._resultsdf_col = [i for i in self._target]
        self.configs = configs
        if configs['n_tasks'] > 1:
            self._resultsdf_col.insert(0,'Overall')
        results_summary_dir= f'{self._run_dir}/ResultSummary.csv'
        if os.path.exists(results_summary_dir):
            self._resultsdf = pd.read_csv(results_summary_dir,header=[0,1],index_col=[0,1])
        else:
            index = pd.MultiIndex.from_product([list(range(configs['REPEATS'])),list(range(configs['FOLDS']))])
            index = index.set_names(['repeat','fold'])
            self._resultsdf = pd.DataFrame(columns=pd.MultiIndex.from_product((self._resultsdf_col,self._metrics_name)),index=index)

    def report_score(self,test_pred, test_label, repeat, fold):

        for criterion in self._metrics:
            score = calculate_score(test_pred, test_label, criterion, self.configs['n_tasks'])
            self._resultsdf.loc[(repeat,fold), pd.IndexSlice[:,criterion.name]] = score
        self._resultsdf.to_csv(self._run_dir+'/ResultSummary.csv',float_format='%.4f')
        return self._resultsdf

    def report_by_target(self):
        outtext_by_target = []
        for col in self._resultsdf.columns:
            mean, interval = compute_confidence_interval(self._resultsdf[col])
            outtext_by_target.extend((mean,interval))
        resultsdf_by_target = pd.DataFrame(columns=pd.MultiIndex.from_product((self._resultsdf_col,self._metrics_name,['mean','interval'])),index=[0])
        resultsdf_by_target.iloc[0] = outtext_by_target
        resultsdf_by_target = resultsdf_by_target.stack(0).droplevel(0) #.swaplevel(0,1)
        resultsdf_by_target = resultsdf_by_target.reindex(columns=product(self._metrics_name,['mean','interval']),index=self._resultsdf_col)
        resultsdf_by_target.to_csv(self._run_dir+'/ResultSummary_ByTarget.csv',float_format='%.4f')
        return resultsdf_by_target

    def get_dataframe(self):
        return self._resultsdf
    
class Saver(object):
    def __init__(self,round_dir, configs, ckpt_init=0):
        super(Saver, self).__init__()
        self.round_dir = round_dir
        if self.round_dir[-1] != '/':
            self.round_dir += '/'
        self.ckpt_dir = self.round_dir + 'checkpoints/'
        self.results_dir = self.round_dir + 'results/'

        self.ckpt_count = ckpt_init
        self.EarlyStopController = EarlyStopController(configs)
        self.max_epoch = configs['max_epoch']

    def SaveModel(self, model, optim, epoch, scores) -> Tuple[bool, int, int]:
        """
        Save a list`[model, optim]` current state in .pt format
        """
        # state = {'model': model, 'optimizer': optimizer, 'epoch': epoch}
        ckpt_name = os.path.join(self.ckpt_dir, f'epoch{epoch}.pt')
        if not os.path.exists(os.path.dirname(ckpt_name)):
            try:
                os.makedirs(os.path.dirname(ckpt_name))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
        torch.save([model, optim], ckpt_name)

        result_file_name = self.results_dir + str(epoch) + '.json'
        if not os.path.exists(os.path.dirname(result_file_name)):
            try:
                os.makedirs(os.path.dirname(result_file_name))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        with open(result_file_name, 'w') as f:
            json.dump(scores, f, indent=4)
        ShouldStop = self.EarlyStopController.ShouldStop(scores, self.ckpt_count)

        if ShouldStop:
            BestValue, BestModelCkpt = self.EarlyStopController.BestModel()
            print("Early stop.")
            print("The Best model's ckpt idx is: ", BestModelCkpt)
            print("The Best Valid Value is: ", BestValue)
            # delete other models
            self.DeleteUselessCkpt(BestModelCkpt)
            return True, BestModelCkpt, BestValue

        elif self.ckpt_count == self.max_epoch-1: # ckpt_count start from 0 while max_epoch is with respect to 1
            BestValue, BestModelCkpt = self.EarlyStopController.BestModel()
            print("The model didn't stop.")
            print("The Best model's ckpt idx is: ", BestModelCkpt)
            print("The Best Valid Value is: ", BestValue)
            self.DeleteUselessCkpt(BestModelCkpt)
            return False, BestModelCkpt, BestValue
        else:
            self.ckpt_count += 1
            BestValue, BestModelCkpt= self.EarlyStopController.BestModel()
            return False, BestModelCkpt, BestValue

    def DeleteUselessCkpt(self, BestModelCkpt):
        file_names = os.listdir(self.ckpt_dir)
        for file_name in file_names:
            ckpt_idx = int(re.findall('\d+',file_name)[-1])
            if int(ckpt_idx) != BestModelCkpt:
                exact_file_path = self.ckpt_dir + file_name
                os.remove(exact_file_path)

    def LoadModel(self):
        dir_files = os.listdir(self.ckpt_dir)  # list of the checkpoint files
        if dir_files:
            dir_files = sorted(dir_files, key=lambda x: os.path.getctime(os.path.join(self.ckpt_dir, x)))
            last_model_ckpt = dir_files[-1]   # find the latest checkpoint file.
            model, optim = torch.load(os.path.join(self.ckpt_dir, last_model_ckpt))
            current_epoch = int(re.findall('\d+',last_model_ckpt)[-1])
            self.ckpt_count = current_epoch + 1  # update the ckpt_count, get rid of overwriting the existed checkpoint files.
            return model, optim
        else:
            return None, None
        
class EarlyStopController(object):
    def __init__(self, configs):
        self.MetricName = 'valid_loss'
        self.MaxResult = 9e8
        self.MaxResultModelIdx = None
        self.LastResult = 0
        self.LowerThanMaxNum = 0
        self.DecreasingNum = 0
        self.LowerThanMaxLimit = configs['LowerThanMaxLimit']
        self.DecreasingLimit = configs['DecreasingLimit']
        self.TestResult = []

    def ShouldStop(self, score, ckpt_idx):
        MainScore = score[self.MetricName]
        if self.MaxResult > MainScore:
            self.MaxResult = MainScore
            self.MaxResultModelIdx = ckpt_idx
            self.LowerThanMaxNum = 0
            self.DecreasingNum = 0
        else:
            self.LowerThanMaxNum += 1
            if MainScore > self.LastResult:
                self.DecreasingNum += 1
            else:
                self.DecreasingNum = 0
        self.LastResult = MainScore

        if self.LowerThanMaxNum > self.LowerThanMaxLimit:
            return True
        if self.DecreasingNum > self.DecreasingLimit:
            return True
        return False

    def BestModel(self):
        return self.MaxResult, self.MaxResultModelIdx
