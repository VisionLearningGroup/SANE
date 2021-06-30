from __future__ import print_function
import _init_paths
import argparse
import os

import json
import torch

import torch.backends.cudnn as cudnn
from torchvision import transforms
from torch.autograd import Variable
from tqdm import tqdm
import numpy as np

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import nets.Resnet_18
from nets.type_specific_network import TypeSpecificNet
from nets.awa_net_wrapper import AwAWrapper
from nets.tripletnet import Tripletnet

from misc.saliency_maps import SaliencyModel
from datasets.saliency.awa import AwALoader
from datasets.saliency.polyvore_outfits import OutfitsLoader

# Training settings
parser = argparse.ArgumentParser(description='Similarity Explanations Example')
parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                    help='input batch size for training (default: 256)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--polyvore_split', default='nondisjoint', type=str,
                    help='specifies the split of the polyvore data (either disjoint or nondisjoint)')
parser.add_argument('--dataset', default='polyvore_outfits', type=str, choices=['polyvore_outfits', 'awa'],
                    help='which dataset to run: polyvore_outfits/awa')
parser.add_argument('--datadir', default='data', type=str,
                    help='directory of the polyvore outfits dataset (default: data)')
parser.add_argument('--split', type=str, default='test',
                    help='split to run on, if training it caches heatmaps rather than tests')
parser.add_argument('--dim_embed', type=int, default=64, metavar='N',
                    help='how many dimensions in embedding (default: 64)')
parser.add_argument('--method', type=str, default='rise', choices=['rise', 'fong', 'slide'],
                    help='type of explanations: rise/fong/slide')
parser.add_argument('--fixed_ref', action='store_true', default=False,
                    help='use the whole image when computing saliency')
parser.add_argument('--mask_size', type=int, default=8, metavar='N',
                    help='resolution of the input mask (default: 8)')
parser.add_argument('--num_mask', type=int, default=2000, metavar='N',
                    help='number of random masks to use with rise (default: 2000)')
parser.add_argument('--mask_prob', type=float, default=0.5, metavar='N',
                    help='portion of the random mask to be set to 1 (default: 0.5)')
parser.add_argument('--num_heatmaps', type=int, default=20, metavar='N',
                    help='number of saliency maps used to supervise the attribute activation maps (default: 20)')

def pdist(x1, x2):
    """
        x1: Tensor of shape (h1, w)
        x2: Tensor of shape (h2, w)
        Return pairwise distance for each row vector in x1, x2 as
        a Tensor of shape (h1, h2)
    """
    return torch.mm(x1, x2.transpose(0, 1))

def evaluate_saliency(model, test_loader):
    cached_embedding_fn = os.path.join(args.datadir, 'cache', args.dataset, 'embeddings.pkl')
    if os.path.isfile(cached_embedding_fn):
        embeddings = torch.load(cached_embedding_fn)
        if args.cuda:
            embeddings = embeddings.cuda()
    else:
        embeddings = []
        for images in tqdm(test_loader,
                           desc='caching fixed embeddings',
                           total=len(test_loader)):
            if args.cuda:
                images = images.cuda()
            images = Variable(images)
            embeddings.append(model.embeddingnet(images).data)

        embeddings = torch.cat(embeddings)
        torch.save(embeddings, cached_embedding_fn)

    insert_auc, delete_auc = test_loader.dataset.test(model, embeddings)
    print('\n{} set for {} using {}: Insert AUC: {:.1f}\t Delete AUC: {:.1f}\n'.format(
        test_loader.dataset.split, args.dataset, args.method,
        round(insert_auc * 100, 1), round(delete_auc * 100, 1)))

def main():
    global args
    args = parser.parse_args()
    assert(args.method in ['slide', 'rise', 'mask', 'lime', 'random'])
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform = transforms.Compose([
        transforms.Resize(112),
        transforms.CenterCrop(112),
        transforms.ToTensor(),
        normalize,
    ])

    text_feat_dim = 6000
    kwargs = {'num_workers': 8, 'pin_memory': True} if args.cuda else {}
    if args.dataset == 'polyvore_outfits':
        fn = os.path.join(args.datadir, args.dataset, 'polyvore_item_metadata.json')
        meta_data = json.load(open(fn, 'r'))
        data_loader = torch.utils.data.DataLoader(
            OutfitsLoader(args, args.split, meta_data, transform=transform),
            batch_size=args.batch_size, shuffle=False, **kwargs)

        feat_cnn = nets.Resnet_18.resnet18(pretrained=True, embedding_size=args.dim_embed)
        tsn = TypeSpecificNet(args, feat_cnn, len(data_loader.dataset.typespaces))
        embeddingnet = Tripletnet(args, tsn, text_feat_dim)
    elif args.dataset == 'awa':
        data_loader = torch.utils.data.DataLoader(
            AwALoader(args, args.split, transform=transform),
            batch_size=args.batch_size, shuffle=False, **kwargs)

        tsn = nets.Resnet_18.resnet18(pretrained=True, embedding_size=args.dim_embed, norm_output=True)
        embeddingnet = Tripletnet(args, AwAWrapper(tsn), text_feat_dim)
    else:
        raise ValueError('Unrecognized dataset: ' + args.dataset)

    if args.cuda:
        embeddingnet.cuda()

    resume = os.path.join('image_similarity_models', args.dataset + '_model.pth.tar')
    print("=> loading checkpoint '{}'".format(resume))
    checkpoint = torch.load(resume, encoding='latin1')
    embeddingnet.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (epoch {})".format(resume, checkpoint['epoch']))

    # switch to evaluation mode
    embeddingnet.eval()
    model = SaliencyModel(tsn)

    if args.split == 'train':
        data_loader.dataset.compute_train_saliency_maps(model)
    else:
        evaluate_saliency(model, data_loader)

if __name__ == '__main__':
    main()
