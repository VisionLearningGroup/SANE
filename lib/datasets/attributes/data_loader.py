import os
import torch
import scipy
import numpy as np
import sklearn.metrics as skm

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 
import tempfile
from torchvision import transforms
from tqdm import tqdm
from datasets.data_loader import DataLoader

class AttributeDataLoader(DataLoader):
    def __init__(self, args, split, transform=None, visualize=False):
        super(AttributeDataLoader, self).__init__(args, split, transform)
        self.max_num_heatmaps = args.num_heatmaps
        if self.is_train:
            # we want the loaded heatmaps to be on the CPU for batch creation
            # during training
            self.cuda = False

        typedir = 'var'
        if args.fixed_ref:
            typedir = 'fixed'

    def process_heatmap(self, heatmap):
        if heatmap is None:
            return heatmap
        
        w, h = heatmap.size()[-2:]
        heatmap = heatmap.view(-1, 1, w, h)
        heatmap = torch.nn.functional.interpolate(heatmap, size=(self.mask_size, self.mask_size), mode='bilinear', align_corners=True)
        if self.is_train:
            heatmap = heatmap.view(-1, self.mask_size * self.mask_size)
        else:
            heatmap = heatmap.view(-1)

        return heatmap

    def test(self, attribute_scores, attribute_labels):
        attribute_scores = attribute_scores.cpu().numpy()
        ap = []
        for i in range(self.num_attr):
            labels = attribute_labels[:, i]
            if np.sum(labels) > 0:
                ap.append(skm.average_precision_score(labels, attribute_scores[:, i]))

        mAP = np.mean(ap)
        return mAP

    def __getitem__(self, index):
        imfn = self.images[index]
        img = self.loader(os.path.join(self.impath, imfn + '.jpg'))
        if self.transform is not None:
            img = self.transform(img)

        attr = self.attribute_data[imfn]
        labels = np.zeros(self.num_attr, np.float32)
        num_gt = len(attr)
        if num_gt > 0:
            labels[attr] = (1. / num_gt)

        if self.is_train:
            heatmap = np.zeros((self.max_num_heatmaps, self.mask_size * self.mask_size), np.float32)
            all_heatmaps = self.load_heatmap(imfn)
            num_heatmaps = min(self.max_num_heatmaps, len(all_heatmaps))
            heatmap_mask = np.zeros(self.max_num_heatmaps, np.float32)
            heatmap_mask[:num_heatmaps] = 1
            heatmap[:num_heatmaps] = all_heatmaps[:num_heatmaps]
            return img, labels, heatmap, heatmap_mask
        else:
            return img, labels

