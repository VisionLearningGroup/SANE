import os
import json
import pickle
from datasets.attributes.data_loader import AttributeDataLoader

class OutfitsLoader(AttributeDataLoader):
    def __init__(self, args, split, transform=None, visualize=False):
        super(OutfitsLoader, self).__init__(args, split, transform, visualize)
        rootdir = os.path.join(args.datadir, 'polyvore_outfits', args.polyvore_split)
        self.impath = os.path.join(args.datadir, 'polyvore_outfits', 'images')
        data_json = os.path.join(rootdir, '%s.json' % split)
        outfit_data = json.load(open(data_json, 'r'))
        fn = os.path.join(args.datadir, 'polyvore_outfits', 'polyvore_attributes.json')
        attribute_data = json.load(open(fn, 'r'))
        self.attribute_data = attribute_data['annotations']
        self.num_attr = len(attribute_data['attributes'])
        fn = os.path.join(args.datadir, 'polyvore_outfits', 'attribute_list.txt')
        self.attribute_names = [line.strip() for line in open(fn, 'r').readlines()]

        # get list of images from the outfit data
        fn = os.path.join(args.datadir, args.dataset, 'polyvore_item_metadata.json')
        meta_data = json.load(open(fn, 'r'))
        images = set()
        for outfit in outfit_data:
            for item in outfit['items']:
                item_id = item['item_id']
                if len(attribute_data['annotations'][item_id]) > 0 or not self.is_train:
                    images.add(item_id)

        self.images = list(images)
        self.image2index = dict(zip(self.images, range(len(self.images))))


