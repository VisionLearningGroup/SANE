from __future__ import print_function
import _init_paths
import argparse
import os
import sys
import shutil
import json

import numpy as np

import scipy
import torch
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

import pickle
import nets.Resnet_18
from nets.type_specific_network import TypeSpecificNet
from nets.awa_net_wrapper import AwAWrapper
from nets.tripletnet import Tripletnet

import torchvision.models as models
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression as LR
from misc.fix_blocks import fix_resnet_blocks
#from nets.latent_attribute import LatentHeatmapModel
from nets.supervised_attribute import SupervisedHeatmapModel
from datasets.attributes_polyvore_outfits_cache import AttributePOutfitsLoader
from datasets.attributes_awa_cache import AttributeAwALoader

# Training settings
parser = argparse.ArgumentParser(description='Fashion Compatibility Example')
parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                    help='input batch size for training (default: 256)')
parser.add_argument('--epochs', type=int, default=300, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--start_epoch', type=int, default=1, metavar='N',
                    help='number of start epoch (default: 1)')
parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                    help='learning rate (default: 5e-4)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--log-interval', type=int, default=150, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='Type_Specific_Fashion_Compatibility', type=str,
                    help='name of experiment')
parser.add_argument('--polyvore_split', default='nondisjoint', type=str,
                    help='specifies the split of the polyvore data (either disjoint or nondisjoint)')
parser.add_argument('--dataset', default='polyvore_outfits', type=str,
                    help='which dataset to run: polyvore_outfits/awa')
parser.add_argument('--attr_model', default='classifier', type=str,
                    help='type of attribute model (classifier, latent, supervised)')
parser.add_argument('--method', type=str, default='rise',
                    help='type of explanations: rise/fong/slide')
parser.add_argument('--fixed_ref', action='store_true', default=False,
                    help='use the whole image when computing saliency')
parser.add_argument('--mask_size', type=int, default=4, metavar='N',
                    help='resolution of the input mask (default: 4)')
parser.add_argument('--fix_blocks', type=int, default=0, metavar='N',
                    help='number of resnet blocks to fix during training (default: 0)')
parser.add_argument('--datadir', default='data', type=str,
                    help='directory of the polyvore outfits dataset (default: data)')
parser.add_argument('--heat_loss', type=float, default=5e-3,
                    help='weight for heatmap loss (default: 5e-6)')
parser.add_argument('--attr_weight', type=float, default=1,
                    help='tradeoff between attribute and heatmap matching scores (default: 1)')
parser.add_argument('--test', dest='test', action='store_true', default=False,
                    help='To only run inference on test set')
parser.add_argument('--train_val', dest='train_val', action='store_true', default=False,
                    help='train using the training and validation sets (only awa)')
parser.add_argument('--dim_embed', type=int, default=64, metavar='N',
                    help='how many dimensions in embedding (default: 64)')

def main():
    global args
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    assert(args.attr_weight <= 1 and args.attr_weight >= 0)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    kwargs = {'num_workers': 8, 'pin_memory': True} if args.cuda else {}
    if args.dataset == 'polyvore_outfits':
        dataset_class = AttributePOutfitsLoader
    elif args.dataset == 'awa':
        dataset_class = AttributeAwALoader
    else:
        raise ValueError('Unrecognized dataset: ' + args.dataset)

    test_loader = torch.utils.data.DataLoader(
        dataset_class(args, 'test',
                      transform=transforms.Compose([
                          transforms.Resize(112),
                          transforms.CenterCrop(112),
                          transforms.ToTensor(),
                          normalize,
                      ])),
        batch_size=args.batch_size // 2, shuffle=False, **kwargs)

    model = AttributePredictor(test_loader.dataset.num_attr)
    if args.cuda:
        model.cuda()

    best_acc = 0
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                    .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    calibration_params =  'data/%s_calibration_sigmoid.pkl' % args.dataset
    if os.path.exists(calibration_params):
        calibration = pickle.load(open(calibration_params, 'rb'))
    else:
        assert(False)
        attr2im = [[] for _ in range(test_loader.dataset.num_attr)]
        finished = np.zeros(test_loader.dataset.num_attr, np.int32)
        num_items = 200
        for imfn in test_loader.dataset.images:
            for attr_id in test_loader.dataset.attribute_data[imfn]:
                attr2im[attr_id].append(imfn)

        for imfn in val_loader.dataset.images:
            for attr_id in val_loader.dataset.attribute_data[imfn]:
                attr2im[attr_id].append(imfn)

        for imfn in train_loader.dataset.images:
            for attr_id in train_loader.dataset.attribute_data[imfn]:
                if len(attr2im[attr_id]) < num_items:
                    attr2im[attr_id].append(imfn)

        all_imfns = set()
        for attr_id, imfns in enumerate(attr2im):
            #attr2im[attr_id] = set(imfns[:num_items])
            all_imfns.update(attr2im[attr_id])

        all_imfns = list(all_imfns)
        im2idx = dict(zip(all_imfns, range(len(all_imfns))))
        labels = np.zeros((len(all_imfns), len(attr2im)), np.float32)
        for attr_id, imfns in enumerate(attr2im):
            for imfn in imfns:
                labels[im2idx[imfn], attr_id] = 1

        images = []
        for imfn in tqdm(all_imfns, desc='converting images', total=len(all_imfns)):
            img = test_loader.dataset.loader(os.path.join(test_loader.dataset.impath, imfn + '.jpg'))
            if test_loader.dataset.transform is not None:
                img = test_loader.dataset.transform(img)
                 
            images.append(img)

        images = torch.stack(images)
        model.eval()
        scores = []
        num_batches = int(np.ceil(len(images) / float(args.batch_size)))
        for i in tqdm(range(0, len(images), args.batch_size), desc='scoring', total=num_batches):
            batch_scores, _ = model(Variable(images[i:i+args.batch_size].cuda()))
            scores.append(batch_scores.data.cpu())

        scores = torch.cat(scores, 0).numpy()
        calibration = []
        neg_mult = 5
        for attr_id, imfns in tqdm(enumerate(attr2im), desc='calibrating', total=len(attr2im)):
            pos_scores = scores[np.where(labels[:, attr_id] > 0)[0], attr_id]
            calibration.append(np.median(pos_scores))

        pickle.dump(calibration, open(calibration_params, 'wb'))

    test_acc = test(test_loader, model, calibration)

def test(test_loader, model, calibration):
    attr_fn = 'data/val_%s_attr_scores_sigmoid.pkl' % args.dataset
    if os.path.exists(attr_fn):
        data = pickle.load(open(attr_fn, 'rb'))
        scores, heatmaps, labels = data['scores'], data['heatmaps'], data['labels']
    else:
        attr_fn = 'data/%s_attr_scores_sigmoid.pkl' % args.dataset
        data = pickle.load(open(attr_fn, 'rb'))
        scores, heatmaps, labels = data['scores'], data['heatmaps'], data['labels']

        # switch to evaluation mode
        model.eval()
        #scores = []
        heatmaps = []
        labels = []
        # for test/val data we get images only from the data loader
        for batch_idx, (images, im_labels, _, _) in enumerate(test_loader):
            if args.cuda:
                images = images.cuda()

            images = Variable(images)
            labels.append(im_labels)
            if args.attr_model == 'classifier':
                pred = model(images)
            else:
                if args.attr_model == 'latent':
                    pred, heatmap, _ = model(images)
                else:
                    pred, heatmap = model(images)

                heatmaps.append(heatmap.data)

            print(pred.data)
            assert False
            scores.append(pred.data)

        labels = torch.cat(labels).numpy()
        scores = torch.cat(scores).cpu().numpy()
        heatmaps = torch.cat(heatmaps).view(len(scores), scores.shape[1], -1)
        pickle.dump({'scores' : scores,
                     'heatmaps' : heatmaps.cpu().numpy(),
                     'labels' : labels}, open(attr_fn, 'wb'))

    if args.dataset == 'polyvore_outfits':
        for i, thresh in enumerate(calibration):
            labels[:, i] = ((scores[:, i] > thresh) + labels[:, i]) > 0
    else:
        labels = labels > 0

    scores = torch.from_numpy(scores)
    test_loader.dataset.set_labels_scores(labels, scores)
    removal_scores = []
    for i, (removes) in tqdm(enumerate(test_loader),
                                       desc='%s attribute caching' % args.dataset,
                                       total=len(test_loader)):
        removal_scores.append(removes)
        
    removal_scores = np.vstack(removal_scores)
    print(removal_scores.shape)

    np.save('data/%s_orcale_removal_sigmoid.npy' % args.dataset, removal_scores)

if __name__ == '__main__':
    main()    
