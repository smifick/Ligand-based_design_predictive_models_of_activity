import json
import yaml
import pickle as pkl
import os
import shutil
from deepchem.splits import RandomStratifiedSplitter, ScaffoldSplitter
from graphegfr.metrics import *
import torch

# NOTE Is ScaffoldSplitter still used?

param_types = {
    'target':str,
    'database':str,
    'hyperparam': dict,
    'result_folder': str,
    'no_fea':bool,
    'no_feaf':bool,
    'no_feac':bool,
    'device':str,
    'metrics':list,
    'mainmetrics':str,
    'split':str,
    'overwrite':bool,
    'pt_model_path':str,
    'pt_load':str,
    'freeze_pt':bool,
}

target_dict = {
    r'MTL_HER124': ['HER1','HER2','HER4'],
    r'MTL_ALL_WT_MT': ['HER1','HER2','HER4','T790M_L858R','L858R','delE746_A750','T790M'],
    r'MTL_HER1_ALL_MT': ['HER1','T790M_L858R','L858R','delE746_A750','T790M'],
    r'MTL_ALL_MT': ['T790M_L858R','L858R','delE746_A750','T790M'],
    r'HER1':['HER1'],
    r'HER2':['HER2'],
    r'HER4':['HER4'],
    r'T790M_L858R':['T790M_L858R'],
    r'L858R':['L858R'],
    r'delE746_A750':['delE746_A750'],
    r'T790M':['T790M']
}

database_path_dict = {
    r'LigEGFR':'./resources'
}

pt_layers_names_dict = {
    'all': ['mol_model', 'cnn', 'dnn'],
    'graph': ['mol_model'],
    'fp': ['cnn', 'dnn'],
}

metrics_dict = {x.name: x for x in (
    RMSE(),
    MAE(),
    MSE(),
    PCC(),
    R2(),
    SRCC()
)}

class Configs(object):
    def __init__(self, configs:dict):
        # Check
        newconfigs = {}
        
        # check required parameters
        for required_param in ['target','hyperparam']:
            if required_param not in configs:
                raise ValueError(f"'{required_param}' parameter is required.")
            
        # check matching parameter type
        for param in configs.keys():
            try:
                param_type = param_types[param]
                configs_param_value = configs[param]
                assert isinstance(configs_param_value, param_type)
            except KeyError:
                raise KeyError(f'Invalid key: {param}')
            except AssertionError:
                raise TypeError(f"Invalid type for {param} (is {type(configs_param_value)}; expecting {param_type})")
        
        # check valid options
        if configs['target'] in target_dict.keys():
            newconfigs['target_list'] = target_dict.get(configs['target'])
        else:
            raise ValueError('Invalid target name')
        newconfigs['n_tasks'] = len(newconfigs['target_list'])
        
        result_folder = configs.get('result_folder', 'results')
        run_dir = os.path.join(result_folder, configs['database'], configs['target']).rstrip('/')
        if os.path.exists(run_dir):
            if configs.get('overwrite', False) == True:
                shutil.rmtree(run_dir)
            else:
                raise FileExistsError("This path exists. If you want to overwrite, flag 'overwrite': true in your configs.json file.")
        newconfigs['run_dir'] = run_dir
        
        splittername = configs.get('split','random') 
        if splittername not in {'random', 'scaffold'}:
            raise ValueError(f"Invalid value for 'splitter'")
        newconfigs['database'] = configs.get('database', 'LigEGFR')
        newconfigs['splitter'] = RandomStratifiedSplitter() if splittername == 'random' else ScaffoldSplitter()
        newconfigs['enable_fea'] = configs.get('enable_fea', True)
        newconfigs['enable_feaf'] = configs.get('enable_feaf', True)
        newconfigs['enable_feac'] = configs.get('enable_feac', True)
        newconfigs['device'] = torch.device(configs.get('device', 'cuda:0'))
        if configs.get('metrics') is not None:
            newconfigs['metrics'] = [metrics_dict[name] for name in configs.get('metrics')]
        else:
            newconfigs['metrics'] = [MSE(), RMSE(), PCC(), R2(), SRCC()]
           
        if configs.get('pt_model_path', None) is not None:
            path = configs['pt_model_path']
            if not os.path.exists(path): raise FileNotFoundError('Path not found: ' + path)
            newconfigs['pt_model_path'] = path
            newconfigs['pt_layers_names'] = pt_layers_names_dict[configs.get('pt_load', 'all')]
            newconfigs['freeze_pt'] = configs.get('freeze_pt',False)
        else:
            newconfigs['pt_model_path'] = None
                
        newconfigs['mainmetrics'] = metrics_dict[configs.get('mainmetrics', newconfigs['metrics'][0].name)]

        newconfigs['target'] = configs['target']
        newconfigs['database_path'] = database_path_dict[configs['database']]
        newconfigs['database'] = configs['database']
        newconfigs['hyperparam'] = configs['hyperparam']

        self.configs = {k:v for k,v in newconfigs.items()}


    def __getitem__(self, key:str):
        try:
            return self.configs[key]
        except:
            raise KeyError(f"'{key}' is invalid")
        
    @staticmethod
    def parseJSON(config_dir: str):
        try:
            configs_dict = json.load(open(config_dir, 'r'))
        except FileNotFoundError: 
            raise FileNotFoundError("Configs file does not exist:", config_dir)
        return Configs(configs_dict)
    
    @staticmethod
    def parseYAML(config_dir: str):
        try:
            configs_dict = yaml.safe_load(open(config_dir, 'r'))
        except FileNotFoundError: 
            raise FileNotFoundError("Configs file does not exist:", config_dir)
        return Configs(configs_dict)

    @staticmethod
    def parse(config_dir: str):
        if config_dir[-4:] == '.yml' or config_dir[-5:] == '.yaml':
            return Configs.parseYAML(config_dir)
        elif config_dir[-5:] == '.json':
            return Configs.parseJSON(config_dir)
        else:
            raise NotImplementedError("No parsing method for file with this extension:", config_dir)

    def to_dict(self):
        return self.configs