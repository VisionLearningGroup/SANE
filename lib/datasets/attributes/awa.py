import os
import numpy as np
from datasets.attributes.data_loader import AttributeDataLoader

class AwALoader(AttributeDataLoader):
    def __init__(self, args, split, transform=None, visualize=False):
        super(AwALoader, self).__init__(args, split, transform, visualize)
        awa_dir = os.path.join(args.datadir, 'Animals_with_Attributes2')
        self.impath = os.path.join(awa_dir, 'JPEGImages')
        # there isn't a set validation set, so we create one from the training set
        if split == 'valid':
            split = 'train'

        with open(os.path.join(awa_dir, 'predicate-matrix-binary.txt'), 'r') as f:
            classes2attr = []
            for line in f:
                line = line.strip()
                if line:
                    classes2attr.append(np.nonzero([int(x) for x in line.split()])[0])

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
            class2index = {}
            for i, line in enumerate(f):
                line = line.strip()
                if line:
                    splitclasses.append(line)
                    class2index[line] = i

        classes2attr = [classes2attr[classes.index(c)] for c in splitclasses]
        self.num_attr = len(predicates)
        self.attribute_names = predicates
        images = []
        # number of images per class for the validation set
        n_val = 20
        for i, cls in enumerate(splitclasses):
            imlist = os.listdir(os.path.join(self.impath, cls))
            if split != 'test' and not args.train_val:
                if self.is_train:
                    imlist = imlist[:-n_val]
                elif self.split == 'valid':
                    imlist = imlist[-n_val:]

            images += imlist

        self.images = [os.path.join(image_id.split('_')[0], image_id.split('.')[0]) for image_id in images]
        self.attribute_data = {}
        for im in self.images:
            c = class2index[im.split(os.path.sep)[0]]
            self.attribute_data[im] = classes2attr[c]
            
        self.image2index = dict(zip(self.images, range(len(self.images))))
        for i in range(len(self.pairs)):
            img1, img2, label = self.pairs[i]
            img1 = os.path.join(img1.split('_')[0], img1)
            img2 = os.path.join(img2.split('_')[0], img2)
            self.pairs[i] = (img1, img2, label)

    def get_heatmap_cachefn(self, img1, img2):
        img1 = img1.split(os.path.sep)[1]
        if img2 is not None:
            img2 = img2.split(os.path.sep)[1]
        
        return super(AwALoader, self).get_heatmap_cachefn(img1, img2)
