from .utils import ResultsReport, Saver, compute_confidence_interval
from .utils import load_tensor, set_split_seed, set_seed, calculate_score

import shutil
import os
from .configs import Configs
import pickle as pkl
import json
from .dataset import load_dataset, KFoldDataset
from .models import GraphEGFR, ModelWrapper
from torch import optim
from torch_geometric.loader import DataLoader
import arrow
import torch
import numpy as np
import pandas as pd
import gc
import random
import deepchem as dc
def run(configs: Configs, DEBUG):

    start_time = arrow.now()
    start_time_formatted = start_time.format('DD/MM/YYYY HH:mm:ss')
    print('Start time:',start_time_formatted)

    random.seed = 42

    if DEBUG:
        
        configs.update(dict(
            REPEATS = 2,
            FOLDS = 2,
            max_epoch = 10,
            printrate = 1,
            LowerThanMaxLimit = 2,
            DecreasingLimit = 2
        ))
        
    else:
        configs.update(dict(
            REPEATS = 5,
            FOLDS = 5,
            max_epoch = 500,
            printrate = 20,
            LowerThanMaxLimit = 50,
            DecreasingLimit = 30
        ))

    FOLDS = configs['FOLDS']
    REPEATS = configs['REPEATS']

    target = configs['target']
    n_tasks = configs['n_tasks']
    device = configs['device']
    splitter = configs['splitter']
    database = configs['database']
    database_path = configs['database_path']
    RUN_DIR = configs['run_dir']
    kfolddataset = KFoldDataset(RUN_DIR)
    
    # ------- Load Feature -------

    print('Loading feature...')
    try:
        np_label = np.load(f'{database_path}/npdata/{target}/label.npy')
        dataframe = pd.read_csv(f'{database_path}/data/' + target + '.csv')
        smiles = dataframe['SMILES_NS'].values
        
        Adj = load_tensor(f'{database_path}/npdata/{target}/adj', torch.FloatTensor, configs['device'])
        for i in range(len(Adj)):
            Adj[i] = Adj[i].unsqueeze(0) 

        Label = load_tensor(f'{database_path}/npdata/{target}/label', torch.FloatTensor, configs['device'])
        fpc = pd.read_csv(f'{database_path}/fingerprint/{target}nonhash.csv').to_numpy()
        fpf = pd.read_csv(f'{database_path}/fingerprint/{target}hash.csv').to_numpy()
        os.makedirs(configs['run_dir'])
    except FileNotFoundError:
        raise FileNotFoundError(f"{database} database does not contain files for '{target}' target.")
    fingfeaf=fpf.shape[-1]
    fingfeac=fpc.shape[-1]
    fpfs=[]
    fpcs=[]
    for i in fpf:
        fpfs.append(torch.cuda.FloatTensor(i))
    for i in fpc:
        fpcs.append(torch.cuda.FloatTensor(i).unsqueeze(1))

    dataset, smiles_list = load_dataset(Adj, smiles, Label, fpfs, fpcs)

    num_atom_features = 75
    edge_dim = 11
    
    dc_dataset = dc.data.NumpyDataset(X=smiles_list,y=np_label,ids=list(range(len(np_label))),w=~np.isnan(np_label))
    dc_tvset, dc_testset = splitter.train_test_split(dc_dataset,frac_train=0.8, seed=42)

    tvset = dataset[dc_tvset.ids.tolist()]
    testset = dataset[dc_testset.ids.tolist()]
    dc_tvset = dc.data.NumpyDataset(X=dc_tvset.X,y=dc_tvset.y,ids=list(range(len(tvset)))) #reindex

    test_loader = DataLoader(testset,batch_size=1,shuffle=False)

    del dc_testset
    gc.collect()  

    print('- Done!')
    print("Loading hyperparameters...")
    hyperparam = configs['hyperparam']
    
    num_layers = hyperparam.get('num_layers')
    num_timesteps = hyperparam.get('num_timesteps')
    dropout = hyperparam.get('dropout')
    batchsize = hyperparam.get('batchsize')
    lr = hyperparam.get('lr')
    weight_decay = hyperparam.get('weight_decay')
    fingerprint_dim = hyperparam.get('fingerprint_dim')
    
    print('- Done!')
    print('====================')
    results_report = ResultsReport(configs)

    # ------- Run Experiment -------

    for repeat in range(REPEATS):

        print(f'Repeat:{repeat+1}/{REPEATS}')
        print('====================')
        split_seed = set_split_seed()

        # if not resume_flag:
        kfolddataset.create_dataset(dc_tvset,folds=FOLDS,seed=split_seed,splitter=splitter)

        for fold in range(FOLDS):

            dc_trainset, dc_validset = kfolddataset.get_dataset(fold)
            trainset = tvset[dc_trainset.ids.tolist()]
            validset = tvset[dc_validset.ids.tolist()]

            print(f'Fold:{fold+1}/{FOLDS}')
            print('====================')
            seed = set_seed()

            round_dir = f'{RUN_DIR}/repeat_{repeat}-fold_{fold}-split_seed_{split_seed}'
            if os.path.exists(round_dir):
                shutil.rmtree(round_dir) # reset training process on this round
            os.makedirs(round_dir)
            saver = Saver(round_dir, configs)

            train_loader = DataLoader(trainset,batch_size=batchsize,shuffle=True,worker_init_fn=np.random.seed(seed))
            valid_loader = DataLoader(validset,batch_size=1,shuffle=False)

            model = GraphEGFR(num_atom_features,edge_dim, fingerprint_dim,
                                          num_layers, num_timesteps, dropout,fingfeaf, fingfeac, configs).to(device)
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

            if configs.get('pt_model_path', None) is not None:
                try:
                    pretrained_model, *_ = torch.load(configs['pt_model_path'],map_location=configs['device'])
                except:
                    pretrained_model = torch.load(configs['pt_model_path'],map_location=configs['device'])
                
                pt_model_dict = pretrained_model.state_dict()
                model_dict = model.state_dict()

                # 1. filter out unnecessary keys
                pt_model_dict_filtered = {k: pt_model_dict[k] for k in pt_model_dict if k in model_dict and pt_model_dict[k].size() == model_dict[k].size() and 'predict_property' not in k and any(x in k for x in configs['pt_layers_names'])}
                print('original layers', [layer for layer in model_dict])
                print('original pt layers', [layer for layer in pt_model_dict])
                print("pretrained layers:", [layer for layer in pt_model_dict_filtered])
                # 2. overwrite entries in the existing state dict
                model_dict.update(pt_model_dict_filtered) 
                model.load_state_dict(model_dict)
                freeze_pt = configs['freeze_pt']
                if freeze_pt:
                    for name,params in model.named_parameters():
                        if name in pt_model_dict_filtered:
                            params.requires_grad = False
                            print(name)
                print('freeze pretrained:', freeze_pt)
                del pretrained_model, pt_model_dict_filtered
                gc.collect()
                print('Finished loading pretrained model')

            wrapper = ModelWrapper(model,configs,optimizer)

            stop_flag = False

            loss_train_list = []
            loss_valid_list = []

            for epoch in range(configs['max_epoch']):
                if stop_flag:
                    break
                if epoch % configs['printrate'] == 0:
                    print(f"Epoch:{epoch+1}/{configs['max_epoch']}")
              
                wrapper.fit(train_loader)
                train_pred,train_label = wrapper.predict(train_loader)
                valid_pred,valid_label = wrapper.predict(valid_loader)

                train_loss = calculate_score(train_pred, train_label, configs['mainmetrics'], n_tasks)
                valid_loss = calculate_score(valid_pred, valid_label, configs['mainmetrics'], n_tasks)

                loss_train_list.append(train_loss)
                loss_valid_list.append(valid_loss)

                scores = {
                    'train_loss':train_loss[0],
                    'valid_loss':valid_loss[0],
                }

                stop_flag, *_ = saver.SaveModel(wrapper.model,wrapper.optimizer,epoch,scores)

                if epoch % 10 == 0:
                    torch.cuda.empty_cache()

            # final model assessnent

            bestmodel, *_ = saver.LoadModel()
            bestwrapper = ModelWrapper(bestmodel, configs)
            train_pred, train_label = bestwrapper.predict(train_loader)
            test_pred, test_label = bestwrapper.predict(test_loader)

            pt_dict = {
                'train_p':train_pred,
                'train_t':train_label,
                'test_p':test_pred,
                'test_t':test_label,
            }

            pkl.dump(pt_dict,open(round_dir+"/pred_true.pkl", "wb"))

            results_report.report_score(test_pred, test_label, repeat, fold)
            
            loss_valid_np = np.asarray(loss_valid_list)
            loss_train_np = np.asarray(loss_train_list)
            np.savetxt(round_dir+'/validloss.csv',loss_valid_np,delimiter = ',',fmt='%.3f')
            np.savetxt(round_dir+'/trainloss.csv',loss_train_np,delimiter = ',',fmt='%.3f')

            del train_loader, valid_loader, dc_trainset, dc_validset
            gc.collect()
            print('====================')
        # ck_fold=0

    print('Writing output..')

    resultsdf = results_report.get_dataframe()

    with open(RUN_DIR + '/results.json', 'w') as jsonfile:
        jsondict = {}
        for target in configs['target_list']:
            jsondict[target] = {}
            for metrics in configs['metrics']:
                jsondict[target][metrics.name] = {}

        for col in resultsdf.columns:
            mean, interval = compute_confidence_interval(resultsdf[col])
            target = col[0]
            criterion = col[1]
            jsondict[target][criterion]['mean'] = mean
            jsondict[target][criterion]['interval'] = interval
        json.dump(jsondict,jsonfile, indent = 2)

    results_report.report_by_target()
    torch.cuda.empty_cache()
    kfolddataset.delete()

    end_time = arrow.now()
    end_time_formatted = end_time.format('DD/MM/YYYY HH:mm:ss')
    print('Finish time:',end_time_formatted)
