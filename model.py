import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable

def normalized_columns_initializer(weights, std=1.0):
    """
    Weights are normalized over their column. Also, allows control over std which is useful for
    initialising action logit output so that all actions have similar likelihood
    """

    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True))
    return out


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)


class ActorCritic(torch.nn.Module):    

    def __init__(self, config, num_actions, train_resnet=False, use_gpu=False):
        super(ActorCritic, self).__init__()

        self.config = config
        self.dtype = torch.FloatTensor if not use_gpu else torch.cuda.FloatTensor
        self.train_resnet = train_resnet

        if self.train_resnet:
            self.extractor = models.resnet50(pretrained=True)
            modules = list(self.extractor.children())[:-1]
            self.extractor = nn.Sequential(*modules)

        self.visual_ft = nn.Linear(in_features=2048 * self.config['history_size'], out_features=512)

        if config["embeddings"] is None: 
            self.semantic_size = 100
            self.embeddings = lambda x: np.random.sample((100, )).astype(np.float32)
        else:
            self.embeddings = pickle.load(open(config["embeddings"], 'rb'))
            self.semantic_size = list(self.embeddings.values())[0].shape[0]
            
        self.semantic_ft = nn.Linear(in_features=self.semantic_size, out_features=512)

        if self.config['graph']:
            fused_size = 512 * 3
        else:
            fused_size = 512 * 2

        self.hidden_mlp = nn.Linear(in_features=fused_size, out_features=512)
        self.critic_linear = nn.Linear(512, 1)
        self.actor_linear = nn.Linear(512, num_actions)

        self.apply(weights_init)
        self.actor_linear.weight.data = normalized_columns_initializer(
                                            self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = normalized_columns_initializer(
                                            self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

    def forward(self, inputs, word):
        assert len(inputs) == self.config['history_size']

        inputs = [torch.from_numpy(inp).type(self.dtype) for inp in inputs]
        if self.train_resnet:
            features = [self.extractor(inp.unsqueeze(0)) for inp in inputs]
        else:
            features = inputs
            
        joint_features = torch.cat(features)
        joint_features = joint_features.view(1, joint_features.size(0))
        visual = F.relu(self.visual_ft(joint_features))
        
        embeded = torch.from_numpy(self.embeddings[word]).type(self.dtype)
        embeded = embeded.view(1, embeded.size(0))
        semantic = F.relu(self.semantic_ft(embeded))
        joint_embeddings = torch.cat((visual, semantic), 1)
        
        x = self.hidden_mlp(joint_embeddings)
        x = F.relu(x)
        
        return self.critic_linear(x), self.actor_linear(x)
