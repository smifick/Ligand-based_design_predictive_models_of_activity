import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn.models.attentive_fp import AttentiveFP
from graphegfr.utils import isnan
from dgllife.model.gnn import AttentiveFPGNN
from dgllife.model.readout import AttentiveFPReadout
from dgl import batch

class GraphEGFR(nn.Module):
    def __init__(self,node_feat_size: int, edge_feat_size: int,
                 graph_feat_size: int, num_layers: int,
                 num_timesteps: int, dropout: float, fingfeaf:int, 
                 fingfeac:int, configs):
        super().__init__()
        self.mol_model = AttentiveFP_DGL(node_feat_size,
                 edge_feat_size,
                 num_layers,
                 num_timesteps,
                 graph_feat_size,
                 dropout)
        self.fingfeaf = fingfeaf
        self.fingfeac = fingfeac
        self.enable_fea = configs['enable_fea']
        self.enable_feaf = configs['enable_feaf']
        self.enable_feac = configs['enable_feac']
        self.n_tasks = configs["n_tasks"]

        self.cnn1 = nn.Conv1d(fingfeac, 256, 1)
        self.cnn2 = nn.Conv1d(256, 128, 1)

        self.cnn_line = nn.Linear(fingfeaf, fingfeaf)

        self.dnn1 = nn.Linear(fingfeaf, 512)
        self.dnn2 = nn.Linear(512, 1024)
        self.dnn3 = nn.Linear(1024, fingfeaf)

        self.predict_property = nn.Linear((self.enable_fea*graph_feat_size)+(self.enable_feaf*fingfeaf)+(self.enable_feac*128), self.n_tasks)

    def DNN(self, x_words):

        x_words = F.relu(self.dnn1(x_words))
        x_words = F.relu(self.dnn2(x_words))
        x_words = self.dnn3(x_words)

        return x_words

    def forward(self, data):
        node_features, edge_features, fpf, fpc, dgl_g = data.x, data.edge_attr, data.fpf, data.fpc, batch(data.dgl)
        fpf = torch.reshape(fpf, (-1, self.fingfeaf))
        fpc = torch.reshape(fpc, (-1, self.fingfeac, 1))
        
        fpf = list(fpf)
        for i in range(len(fpf)):
            fpf[i] = torch.unsqueeze(fpf[i], 0)
        fpf = torch.cat(fpf, 0)

        fpc = list(fpc)
        for i in range(len(fpc)):
            fpc[i] = torch.unsqueeze(fpc[i], 0)
        fpc = torch.cat(fpc, 0)
        
        # Concatenate molecule and fingerprint
        y_final = torch.Tensor().to(self.get_model_device())
        if self.enable_fea:
            y_molecules = self.mol_model(dgl_g,node_features,edge_features)
            y_final = torch.cat((y_final, y_molecules), 1)
        if self.enable_feac:
            Fringerc = F.relu(self.cnn1(fpc))
            Fringerc = F.relu(self.cnn2(Fringerc))
            Fringerc = Fringerc.squeeze(2)
            y_final = torch.cat((y_final, Fringerc), 1)
        if self.enable_feaf:
            Fringerf = self.DNN(fpf)
            y_final = torch.cat((y_final, Fringerf), 1)    
        output = self.predict_property(y_final)  
        return output
    
    def get_model_device(self):
        return next(self.parameters()).device

class AttentiveFP_DGL(nn.Module):
    def __init__(self,
                 node_feat_size,
                 edge_feat_size,
                 num_layers=2,
                 num_timesteps=2,
                 graph_feat_size=200,
                 dropout=0.):
        super().__init__()

        self.gnn = AttentiveFPGNN(node_feat_size=node_feat_size,
                                  edge_feat_size=edge_feat_size,
                                  num_layers=num_layers,
                                  graph_feat_size=graph_feat_size,
                                  dropout=dropout)
        self.readout = AttentiveFPReadout(feat_size=graph_feat_size,
                                          num_timesteps=num_timesteps,
                                          dropout=dropout)
    def forward(self, g, node_feats, edge_feats, get_node_weight=False):
        node_feats = self.gnn(g, node_feats, edge_feats)
        return self.readout(g, node_feats, get_node_weight)

class ModelWrapper(object):
    def __init__(self, model, configs, optimizer=None):
        self.model = model
        self.optimizer = optimizer
        self.n_tasks = configs["n_tasks"]
        self.device = configs["device"]

    def fit(self, train_loader):
        self.model.train()
        self.model = self.model.to(self.device)

        total_pred=torch.Tensor([]).reshape(-1, self.n_tasks).to(self.device)
        total_label=torch.Tensor([]).reshape(-1, self.n_tasks).to(self.device)

        for i, data in enumerate(train_loader):
            data = data.to(self.device)
            label = data.y
            label = torch.reshape(label, (-1, self.n_tasks))
            self.optimizer.zero_grad()
            pred = self.model(data)

            loss = self.multiloss(pred,label)
            total_pred = torch.cat((total_pred,pred),axis=0)
            total_label = torch.cat((total_label,label),axis=0)
            loss.backward()
            self.optimizer.step()
            torch.cuda.empty_cache()
        # return total_pred.detach().cpu().numpy(), total_label.detach().cpu().numpy()

    def predict(self,test_loader):
        self.model.eval()
        self.model = self.model.to(self.device)
        with torch.no_grad():

            total_pred=torch.Tensor([]).reshape(-1, self.n_tasks).to(self.device)
            total_label=torch.Tensor([]).reshape(-1, self.n_tasks).to(self.device)
            
            for i, data in enumerate(test_loader):
                data = data.to(self.device)
                label = data.y
                label = torch.reshape(label, (-1, self.n_tasks))
                pred = self.model(data)
                total_pred = torch.cat((total_pred,pred),axis=0)
                total_label = torch.cat((total_label,label),axis=0)
                torch.cuda.empty_cache()

        return total_pred.detach().cpu().numpy(), total_label.detach().cpu().numpy()
        
    def multiloss(self,output_vec, target_vec):
        total_output=torch.Tensor([]).to(self.device)
        total_target=torch.Tensor([]).to(self.device)
        criterion = torch.nn.MSELoss(reduction='sum')
        for x in range(len(target_vec)):
            masks = ~isnan(target_vec[x])
            #masks[x] = tmpmasks>0
            if target_vec[x][masks].nelement() == 0:
                loss = [torch.sqrt(torch.tensor(1e-20)),torch.tensor(0.0)]
                continue
            else: # non nans
                task_output_vec = output_vec[x][masks]
                task_target_vec = target_vec[x][masks]
                total_output = torch.cat((total_output,task_output_vec))
                total_target = torch.cat((total_target,task_target_vec))
            overall_loss = criterion(total_output,total_target)
        return overall_loss
