# stripped down version of the tripletnet from the fashion_compatibility repo
# only used for loading the model

import torch.nn as nn

def make_fc_1d(f_in, f_out):
    return nn.Sequential(nn.Linear(f_in, f_out),
                         nn.BatchNorm1d(f_out,eps=0.001,momentum=0.01),
                         nn.ReLU(inplace=True))

class EmbedBranch(nn.Module):
    def __init__(self, feat_dim, embedding_dim):
        super(EmbedBranch, self).__init__()
        self.fc1 = make_fc_1d(feat_dim, embedding_dim)
        self.fc2 = nn.Linear(embedding_dim, embedding_dim)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        
        # L2 normalize each feature vector
        x = nn.functional.normalize(x)
        return x

class Tripletnet(nn.Module):
    def __init__(self, args, embeddingnet, text_dim):
        super(Tripletnet, self).__init__()
        self.embeddingnet = embeddingnet
        self.text_branch = EmbedBranch(text_dim, args.dim_embed)
        self.metric_branch = None


