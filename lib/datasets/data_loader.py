from PIL import Image
import os
import torch.utils.data
import numpy as np

def default_image_loader(path):
    return Image.open(path).convert('RGB')

class DataLoader(torch.utils.data.Dataset):
    def __init__(self, args, split, transform=None, loader=default_image_loader):
        typedir = 'var'
        if args.fixed_ref:
            typedir = 'fixed'

        self.cache_dir = os.path.join(args.datadir, 'cache', args.dataset, typedir, args.method)
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

        if split == 'train':
            self.resized_cache_dir = os.path.join(args.datadir, 'cache', args.dataset, 'resized', typedir, args.method)
            if not os.path.exists(self.resized_cache_dir):
                os.makedirs(self.resized_cache_dir)

        self.transform = transform
        self.is_train = split == 'train'
        self.loader = loader
        self.split = split
        self.pairs = []
        if split != 'train':
            images = set()
            with open(os.path.join('pairs', split, args.dataset + '_pairs.txt'), 'r') as f:
                for line in f:
                    img1, img2, label = line.strip().split()
                    label = int(label)
                    images.update([img1, img2])
                    self.pairs.append((img1, img2, label))

            self.images = list(images)
            self.image2index = dict(zip(self.images, range(len(self.images))))

        self.method = args.method
        self.dataset = args.dataset
        self.fixed_ref = args.fixed_ref
        self.cuda = args.cuda
        self.mask_size = args.mask_size

    def get_typespace(self, anchor, pair):
        return None

    def get_heatmap_cachefn(self, img1, img2 = None):
        if img2 is None:
            return os.path.join(self.resized_cache_dir, img1 + '.npy')
        else:
            return os.path.join(self.cache_dir, img1 + '+' + img2 + '.npy')

    def process_heatmap(self, heatmap):
        return heatmap

    def load_heatmap(self, img1, img2 = None, model = None, embeddings = None):
        if self.method == 'random':
            heatmap = torch.rand(112, 112)
            if self.cuda:
                heatmap = heatmap.cuda()
        else:
            outfn = self.get_heatmap_cachefn(img1, img2)
            heatmap = None
            if os.path.exists(outfn):
                heatmap = torch.from_numpy(np.load(outfn))
                if self.cuda:
                    heatmap = heatmap.cuda()

        return self.process_heatmap(heatmap)

    def __getitem__(self, index):
        imfn = os.path.join(self.impath, self.images[index] + '.jpg')
        img = self.loader(imfn)
        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.images)
