import torch.nn as nn

class AwAWrapper(nn.Module):
    def __init__(self, embeddingnet):
        super(AwAWrapper, self).__init__()
        self.embeddingnet = embeddingnet




