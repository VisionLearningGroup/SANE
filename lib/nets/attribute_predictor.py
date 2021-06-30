import torch
import torch.nn as nn
import torchvision.models as models

class AttributePredictor(nn.Module):
    def __init__(self, num_classes):
        """ args: Input arguments from the main script
            embeddingnet: The network that projects the inputs into an embedding of embedding_size
            embedding_size: Number of dimensions of the embedding output from the embeddingnet
            n_conditions: Integer defining number of different similarity notions
        """
        super(AttributePredictor, self).__init__()
        embeddingnet = models.resnet50(pretrained=True)

        # pulling off last fully connected layer
        layers = list(embeddingnet.children())[:-1]
        self.embeddingnet = nn.Sequential(*layers[:-1])

        # this is just an average pooling layer
        self.cls_net = layers[-1]

        self.heatmap_net = nn.Conv2d(2048, num_classes, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)   

    def forward(self, x):
        """ x: input image data
            c: type specific embedding to compute for the images, returns all embeddings
               when None including the general embedding concatenated onto the end
        """
        embedding = self.embeddingnet(x)
        heatmap = self.heatmap_net(embedding)
        cls_score = self.cls_net(heatmap).squeeze()
        num_batch, num_attr = heatmap.size()[:2]
        heatmap = torch.nn.functional.softmax(heatmap.view(num_batch, num_attr, -1), 2)
        return cls_score, heatmap
