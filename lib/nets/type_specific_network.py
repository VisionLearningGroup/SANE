import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

class ListModule(nn.Module):
    def __init__(self, *args):
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)

class TypeSpecificNet(nn.Module):
    def __init__(self, args, embeddingnet, n_conditions):
        """ args: Input arguments from the main script
            embeddingnet: The network that projects the inputs into an embedding of embedding_size
            embedding_size: Number of dimensions of the embedding output from the embeddingnet
            n_conditions: Integer defining number of different similarity notions
        """
        super(TypeSpecificNet, self).__init__()
        self.embeddingnet = embeddingnet

        # define masks with gradients
        self.masks = torch.nn.Embedding(n_conditions, args.dim_embed)
        # initialize weights
        self.masks.weight.data.normal_(0.9, 0.7) # 0.1, 0.005


    def forward(self, x, c = None):
        """ x: input image data
            c: type specific embedding to compute for the images, returns all embeddings
               when None including the general embedding concatenated onto the end
        """
        embedded_x = self.embeddingnet(x)
        if c is None:
            # used during testing, wants all type specific embeddings returned for an image
            masks = Variable(self.masks.weight.data)
            masks = torch.nn.functional.relu(masks)
                    
            masks = masks.unsqueeze(0).repeat(embedded_x.size(0), 1, 1)
            embedded_x = embedded_x.unsqueeze(1)
            masked_embedding = embedded_x.expand_as(masks) * masks
            masked_embedding = torch.nn.functional.normalize(masked_embedding)
            return torch.cat((masked_embedding, embedded_x), 1)
            
        self.mask = self.masks(c)
        self.mask = torch.nn.functional.relu(self.mask)
        
        masked_embedding = embedded_x * self.mask
        masked_embedding = torch.nn.functional.normalize(masked_embedding)
        return masked_embedding
