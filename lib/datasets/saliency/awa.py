import os
import numpy as np
from datasets.saliency.data_loader import SaliencyDataLoader

class AwALoader(SaliencyDataLoader):
    def __init__(self, args, split, transform=None):
        super(AwALoader, self).__init__(args, split, transform)
        awa_dir = os.path.join(args.datadir, 'Animals_with_Attributes2')
        self.impath = os.path.join(awa_dir, 'JPEGImages')
        if split == 'train':
            with open(os.path.join(awa_dir, 'classes.txt'), 'r') as f:
                classes = []
                for line in f:
                    line = line.strip()
                    if line:
                        classes.append(line.split()[1])

            with open(os.path.join(awa_dir, 'predicates.txt'), 'r') as f:
                predicates = []
                for line in f:
                    line = line.strip()
                    if line:
                        predicates.append(line.split()[1])

            with open(os.path.join(awa_dir, split + 'classes.txt'), 'r') as f:
                splitclasses = []
                for i, line in enumerate(f):
                    line = line.strip()
                    if line:
                        splitclasses.append(line)

            images = []
            cls2ims = {}
            for i, cls in enumerate(splitclasses):
                imlist = os.listdir(os.path.join(self.impath, cls))
                imlist = [im.split('.')[0] for im in imlist]
                cls2ims[cls] = [os.path.join(image_id.split('_')[0], image_id) for image_id in imlist]
                images += imlist

            np.random.shuffle(images)
            self.cls2ims = cls2ims
            self.images = images

        self.images = [os.path.join(image_id.split('_')[0], image_id) for image_id in self.images]
        self.image2index = dict(zip(self.images, range(len(self.images))))
        self._background_pixel_value = (127, 127, 127)
        for i in range(len(self.pairs)):
            img1, img2, label = self.pairs[i]
            img1 = os.path.join(img1.split('_')[0], img1)
            img2 = os.path.join(img2.split('_')[0], img2)
            self.pairs[i] = (img1, img2, label)
        
    def get_heatmap_cachefn(self, img1, img2 = None):
        img1 = img1.split(os.path.sep)[1]
        if img2 is not None:
            img2 = img2.split(os.path.sep)[1]

        return super(AwALoader, self).get_heatmap_cachefn(img1, img2)

    def sample_positive_pairs(self, index):
        cls = self.images[index].split(os.path.sep)[0]
        candidates = list(self.cls2ims[cls])
        candidates.remove(self.images[index])
        if len(candidates) > self.max_num_heatmaps:
            candidates = np.random.choice(candidates, self.max_num_heatmaps, replace=False)

        inds = [self.image2index[c] for c in candidates]
        return inds, candidates
