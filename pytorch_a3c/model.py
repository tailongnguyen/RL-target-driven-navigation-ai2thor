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

    def __init__(self, config, arguments):
        super(ActorCritic, self).__init__()

        self.config = config
        self.arguments = arguments
        self.dtype = torch.FloatTensor if not arguments['use_gpu'] else torch.cuda.FloatTensor
        self.train_resnet = arguments['train_resnet']
        self.history_size = arguments['history_size']

        if self.train_resnet:
            self.extractor = models.resnet50(pretrained=True)
            modules = list(self.extractor.children())[:-1]
            self.extractor = nn.Sequential(*modules)

        self.visual_ft = nn.Linear(in_features=2048 * self.history_size, out_features=512)

        if arguments["embed"] == 0: 
            self.embeddings = pickle.load(open(config["embeddings_onehot"], 'rb'))
        else:
            self.embeddings = pickle.load(open(config["embeddings_fasttext"], 'rb'))

        self.semantic_size = list(self.embeddings.values())[0].shape[0]
        self.semantic_ft = nn.Linear(in_features=self.semantic_size, out_features=512)

        self.num_objects = len(list(self.embeddings.values()))
        self.all_embeddings = list(self.embeddings.values())

        if arguments['use_gcn']:
            fused_size = 512 * 3
            self.adj = normalize(np.load(self.config['adj_file']))
            self.adj = torch.from_numpy(self.adj).type(self.dtype)

            self.score_to_512 = nn.Linear(in_features=1000, out_features=512)
            self.gcn = GCN(nfeat=1024, nhid=1024, nclass=1, dropout=0.5)
            self.gcn_to_512 = nn.Linear(in_features=self.num_objects, out_features=512)
        else:
            fused_size = 512 * 2

        self.hidden_mlp = nn.Linear(in_features=fused_size, out_features=512)
        self.critic_linear = nn.Linear(512, 1)
        self.actor_linear = nn.Linear(512, arguments['action_size'])

        self.apply(weights_init)
        self.actor_linear.weight.data = normalized_columns_initializer(
                                            self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = normalized_columns_initializer(
                                            self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

    def forward(self, inputs, scores, word):
        assert len(inputs) == self.history_size
        inputs = [torch.from_numpy(inp).type(self.dtype) for inp in inputs]    

        joint_features = torch.cat(inputs)
        joint_features = joint_features.view(1, -1)
        visual = F.relu(self.visual_ft(joint_features))
        
        embeded = torch.from_numpy(self.embeddings[word]).type(self.dtype)
        embeded = embeded.view(1, embeded.size(0))
        semantic = F.relu(self.semantic_ft(embeded))
        
        if self.arguments['use_gcn']:
            scores = torch.from_numpy(scores).type(self.dtype)
            scores = scores.view(1, scores.numel())
            scores_512 = F.relu(self.score_to_512(scores))
            nodes = []
            for i in range(self.num_objects):
                em = torch.from_numpy(self.all_embeddings[i]).type(self.dtype)
                em = em.view(1, em.size(0))
                em_512 = F.relu(self.semantic_ft(em))
                nodes.append(torch.cat((scores_512, em_512), 1))

            nodes = torch.stack(nodes).squeeze()
            gcn_out = self.gcn(nodes, self.adj)
            gcn_out = gcn_out.view(1, gcn_out.numel())
            gcn_512 = F.relu(self.gcn_to_512(gcn_out))

            joint_embeddings = torch.cat((visual, semantic, gcn_512), 1)
        else:
            joint_embeddings = torch.cat((visual, semantic), 1)

        x = self.hidden_mlp(joint_embeddings)
        x = F.relu(x)
        
        return self.critic_linear(x), self.actor_linear(x)
