import os
import json
import pickle
import numpy as np
from datasets.saliency.data_loader import SaliencyDataLoader

def load_typespaces(rootdir):
    """ loads a mapping of pairs of types to the embedding used to
        compare them

        rand_typespaces: Boolean indicator of randomly assigning type
                         specific spaces to their embedding
        num_rand_embed: number of embeddings to use when
                        rand_typespaces is true
    """
    typespace_fn = os.path.join(rootdir, 'typespaces.p')
    typespaces = pickle.load(open(typespace_fn, 'rb'))
    ts = {}
    for index, t in enumerate(typespaces):
        ts[t] = index
        
    typespaces = ts
    return typespaces

class OutfitsLoader(SaliencyDataLoader):
    def __init__(self, args, split, meta_data, transform=None):
        super(OutfitsLoader, self).__init__(args, split, transform)
        rootdir = os.path.join(args.datadir, 'polyvore_outfits', args.polyvore_split)
        self.impath = os.path.join(args.datadir, 'polyvore_outfits', 'images')

        data_json = os.path.join(rootdir, '%s.json' % split)
        outfit_data = json.load(open(data_json, 'r'))

        # get list of images and make a mapping used to quickly organize the data
        im2type = {}
        category2ims = {}
        imnames = set()
        id2im = {}
        for outfit in outfit_data:
            outfit_id = outfit['set_id']
            for item in outfit['items']:
                im = item['item_id']
                imnames.add(im)
                category = meta_data[im]['semantic_category']
                im2type[im] = category
                if im not in id2im:
                    id2im[im] = set()

                for item2 in outfit['items']:
                    im2 = item2['item_id']
                    if im == im2:
                        continue

                    id2im[im].add(im2)

        pairs = []
        for imid, outfit in id2im.items():
            id2im[imid] = list(outfit)

        self.im2outfit = id2im
        self.im2type = im2type
        self.typespaces = load_typespaces(rootdir)
        self.images = list(imnames)
        self.image2index = dict(zip(self.images, range(len(self.images))))

    def sample_positive_pairs(self, index):
        candidates = self.im2outfit[self.images[index]]
        if len(candidates) > self.max_num_heatmaps:
            candidates = np.random.choice(candidates, self.max_num_heatmaps, replace=False)

        inds = [self.image2index[c] for c in candidates]
        return inds, candidates

    def get_typespace(self, im1, im2):
        """ Returns the index of the type specific embedding
        for the pair of item types provided as input
        """
        anchor, pair = self.im2type[im1], self.im2type[im2]
        query = (anchor, pair)
        if query not in self.typespaces:
            query = (pair, anchor)
            
        return self.typespaces[query]


