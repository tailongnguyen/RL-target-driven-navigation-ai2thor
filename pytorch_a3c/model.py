import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable
from layers import GraphConvolution
from utils import *

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.gc3 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc3(x, adj))
        return x

class ActorCritic(torch.nn.Module):    

    def __init__(self, config, arguments, gpu_id=-1):
        super(ActorCritic, self).__init__()

        self.config = config
        self.arguments = arguments

        if gpu_id != -1:
            torch.cuda.set_device(gpu_id)
            self.dtype = torch.cuda.FloatTensor
        else:
            self.dtype = torch.FloatTensor
        try:
            self.use_lstm = arguments['lstm']
        except KeyError:
            self.use_lstm = False

        self.history_size = arguments['history_size']

        self.input_size = 2048

        if arguments['onehot']:
            self.input_size = 109
        
        if arguments['train_cnn']:
            self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1)
            self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
            self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
            self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
            self.conv5 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
            self.input_size = 32 * 4 * 4

        if self.use_lstm:
            assert arguments['history_size'] == 1, "History size should be 1 if you want to use lstm."
            self.visual_ft = nn.LSTMCell(input_size=self.input_size, hidden_size=256)
        else:
            self.visual_ft = nn.Linear(in_features=self.input_size * self.history_size, out_features=512)

        if arguments["embed"] == 0: 
            self.embeddings = pickle.load(open(config["embeddings_onehot"], 'rb'))
        else:
            self.embeddings = pickle.load(open(config["embeddings_fasttext"], 'rb'))

        self.semantic_size = list(self.embeddings.values())[0].shape[0]
        self.semantic_ft = nn.Linear(in_features=self.semantic_size, out_features=512)

        self.categories = list(config['new_objects'].keys())
        self.cate2idx = config['new_objects']
        self.num_objects = len(self.categories)
        
        self.all_embeddings = torch.stack([torch.from_numpy(self.embeddings[w]) for w in self.categories], 0).type(self.dtype)
        
        if arguments['use_gcn']:
            self.categories = list(config['new_objects'].keys())
            self.num_objects = len(self.categories)

            fused_size = 512 * 3
            self.adj = normalize(np.load(self.config['adj_file']))
            self.adj = torch.from_numpy(self.adj).type(self.dtype)

            if not arguments['yolo_gcn']:
                self.score_to_512 = nn.Linear(in_features=1000, out_features=512)
                self.gcn = GCN(nfeat=1024, nhid=1024, nclass=1, dropout=0.5)
            else:
                self.gcn = GCN(nfeat=self.num_objects + 512, nhid=self.num_objects + 512, nclass=1, dropout=0.5)

            self.gcn_to_512 = nn.Linear(in_features=self.num_objects, out_features=512)

        elif arguments['use_graph']:
            self.adj = np.load(self.config['adj_file'])
            self.adj = torch.from_numpy(self.adj).type(self.dtype)

            self.graph_ft = nn.Linear(in_features=self.num_objects, out_features=self.num_objects)
            fused_size = 512 * 2 + self.num_objects
        else:
            fused_size = 512 * 2

        self.hidden_mlp = nn.Linear(in_features=fused_size, out_features=512)
        self.critic_linear = nn.Linear(512, 1)
        self.actor_linear = nn.Linear(512, arguments['action_size'])

        self.apply(kaiming_weights_init)
        self.actor_linear.weight.data = normalized_columns_initializer(
                                            self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = normalized_columns_initializer(
                                            self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

    def learned_embedding(self, word):
        embeded = torch.from_numpy(self.embeddings[word]).type(self.dtype)
        embeded = embeded.view(1, embeded.size(0))
        semantic = F.relu(self.semantic_ft(embeded))

        # if self.arguments['use_graph']:
        #     relations = self.adj[self.cate2idx[word]]
        #     r = F.relu(self.graph_ft(relations))
        #     r = r.view(1, r.numel())
        #     joint_embeddings = torch.cat((semantic, r), 1)
        #     return joint_embeddings

        return semantic


    def forward(self, inputs, scores, word):
        if self.arguments['lstm']:
            inputs, (hx, cx) = inputs

        if self.arguments['train_cnn']:
            assert inputs.shape == (self.history_size, 3, 128, 128)
            inputs = torch.from_numpy(inputs).type(self.dtype)
            x = F.elu(self.conv1(inputs))
            x = F.elu(self.conv2(x))
            x = F.elu(self.conv3(x))
            x = F.elu(self.conv4(x))
            x = F.elu(self.conv5(x))
            feature = x.view(-1, self.input_size * self.history_size)
            visual = F.relu(self.visual_ft(feature))

        else:
            torch_inputs = [torch.from_numpy(inp).type(self.dtype) for inp in inputs]    

            if not self.use_lstm:
                joint_features = torch.cat(torch_inputs)
                joint_features = joint_features.view(1, -1)
                visual = F.relu(self.visual_ft(joint_features))
            else:    
                feature = torch_inputs[0].view(-1, self.input_size)
                hx, cx = self.visual_ft(feature, (hx, cx))
                visual = hx.view(1, -1)
            
        embeded = torch.from_numpy(self.embeddings[word]).type(self.dtype)
        embeded = embeded.view(1, embeded.size(0))
        semantic = F.relu(self.semantic_ft(embeded))
        
        if self.arguments['use_gcn']:
            scores = torch.from_numpy(scores).type(self.dtype)
            scores = scores.view(1, scores.numel())
            
            if not self.arguments['yolo_gcn']:
                scores_512 = F.relu(self.score_to_512(scores))

            nodes = []
            ems_512 = F.relu(self.semantic_ft(self.all_embeddings))
            
            if not self.arguments['yolo_gcn']:
                nodes = torch.cat((scores_512.repeat(self.num_objects, 1), ems_512), 1)
            else:
                nodes = torch.cat((scores.repeat(self.num_objects, 1), ems_512), 1)
                
            gcn_out = self.gcn(nodes, self.adj)
            gcn_out = gcn_out.view(1, gcn_out.numel())
            gcn_512 = F.relu(self.gcn_to_512(gcn_out))

            joint_embeddings = torch.cat((visual, semantic, gcn_512), 1)

        elif self.arguments['use_graph']:
            # relations = self.adj[self.cate2idx[word]].numpy()
            # detections = inputs[0][:-4].astype(bool).astype(int)
            # revec = torch.from_numpy(relations * detections).type(self.dtype)
            revec = self.adj[self.cate2idx[word]] 
            r = F.relu(self.graph_ft(revec))
            r = r.view(1, r.numel())
            joint_embeddings = torch.cat((visual, semantic, r), 1)

        else:
            joint_embeddings = torch.cat((visual, semantic), 1)

        x = self.hidden_mlp(joint_embeddings)
        x = F.relu(x)
        
        if self.arguments['lstm']:
            return self.critic_linear(x), self.actor_linear(x), (hx, cx)
        else:
            return self.critic_linear(x), self.actor_linear(x)
