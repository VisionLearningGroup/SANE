from __future__ import print_function
import _init_paths
import argparse
import os
import sys
import shutil

import numpy as np
import sklearn.metrics as skm

import torch
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from tqdm import tqdm

from nets.attribute_predictor import AttributePredictor
from datasets.attributes.polyvore_outfits import OutfitsLoader
from datasets.attributes.awa import AwALoader

# Training settings
parser = argparse.ArgumentParser(description='SANE Attribute Classifier Example')
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
parser.add_argument('--name', default='SANE', type=str,
                    help='name of experiment')
parser.add_argument('--polyvore_split', default='nondisjoint', type=str,
                    help='specifies the split of the polyvore data (either disjoint or nondisjoint)')
parser.add_argument('--dataset', default='polyvore_outfits', type=str,
                    help='which dataset to run: polyvore_outfits/awa')
parser.add_argument('--method', type=str, default='rise',
                    help='type of explanations: rise/fong/slide')
parser.add_argument('--fixed_ref', action='store_true', default=False,
                    help='use the whole image when computing saliency')
parser.add_argument('--mask_size', type=int, default=4, metavar='N',
                    help='resolution of the input mask (default: 4)')
parser.add_argument('--num_heatmaps', type=int, default=20, metavar='N',
                    help='number of saliency maps used to supervise the attribute activation maps (default: 20)')
parser.add_argument('--fix_blocks', type=int, default=0, metavar='N',
                    help='number of resnet blocks to fix during training (default: 0)')
parser.add_argument('--datadir', default='data', type=str,
                    help='directory of the polyvore outfits dataset (default: data)')
parser.add_argument('--heat_loss', type=float, default=5e-3,
                    help='weight for heatmap loss (default: 5e-6)')
parser.add_argument('--test', dest='test', action='store_true', default=False,
                    help='To only run inference on test set')
parser.add_argument('--train_val', dest='train_val', action='store_true', default=False,
                    help='train using the training and validation sets (only awa)')


def main():
    global args
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    kwargs = {'num_workers': 8, 'pin_memory': True} if args.cuda else {}
    if args.dataset == 'polyvore_outfits':
        dataset_class = OutfitsLoader
        image_size = 112
        crop_size = 112
    elif args.dataset == 'awa':
        dataset_class = AwALoader
        image_size = 112
        crop_size = 112
    else:
        raise ValueError('Unrecognized dataset: ' + args.dataset)


    test_loader = torch.utils.data.DataLoader(
        dataset_class(args, 'test',
                      transform=transforms.Compose([
                          transforms.Resize(image_size),
                          transforms.CenterCrop(crop_size),
                          transforms.ToTensor(),
                          normalize,
                      ])),
        batch_size=args.batch_size // 2, shuffle=False, **kwargs)
    train_loader = torch.utils.data.DataLoader(
        dataset_class(args, 'train',
                      transform=transforms.Compose([
                          transforms.Resize(image_size),
                          transforms.CenterCrop(crop_size),
                          #transforms.RandomHorizontalFlip(),
                          transforms.ToTensor(),
                          normalize,
                      ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(
        dataset_class(args, 'valid',
                      transform=transforms.Compose([
                          transforms.Resize(image_size),
                          transforms.CenterCrop(crop_size),
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
    if args.test:
        test_acc = test(test_loader, model)
        sys.exit()

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(parameters, lr=args.lr)
    n_parameters = sum([p.data.nelement() for p in model.parameters()])
    print('  + Number of params: {}'.format(n_parameters))

    for epoch in range(args.start_epoch, args.epochs + 1):
        # update learning rate
        adjust_learning_rate(optimizer, epoch)
        # train for one epoch
        train(train_loader, model, optimizer, epoch)
        # evaluate on validation set
        acc = test(val_loader, model)

        # remember best acc and save checkpoint
        is_best = acc > best_acc
        best_acc = max(acc, best_acc)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_acc,
        }, is_best)

    checkpoint = torch.load(os.path.join('runs', args.dataset, args.name, 'model_best.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])
    test_acc = test(test_loader, model)

def train(train_loader, model, optimizer, epoch):
    losses = AverageMeter()
    losses_heat = AverageMeter()
    criterion = torch.nn.SmoothL1Loss()
    
    # switch to train mode
    model.train()
    for batch_idx, (images, labels, gt_heatmap, heatmap_mask) in enumerate(train_loader):
        if args.cuda:
            images, labels, gt_heatmap, heatmap_mask = images.cuda(), labels.cuda(), gt_heatmap.cuda(), heatmap_mask.cuda()
        images, labels, gt_heatmap, heatmap_mask = Variable(images), Variable(labels), Variable(gt_heatmap), Variable(heatmap_mask)
        scores, supp = model(images)

        cls_loss = criterion(torch.nn.functional.softmax(scores, 1), labels)
        
        num_items = len(images)
        losses.update(cls_loss.item(), num_items)
        gt_heatmap = torch.nn.functional.softmax(gt_heatmap, 2).unsqueeze(2)
        heatmap = supp.unsqueeze(1).expand(num_items, 
                                           train_loader.dataset.max_num_heatmaps, 
                                           train_loader.dataset.num_attr, 
                                           args.mask_size * args.mask_size)

        diff = heatmap - gt_heatmap.expand_as(heatmap)
        norms = 1 - diff.view(num_items, train_loader.dataset.max_num_heatmaps, 
                              train_loader.dataset.num_attr, -1).norm(2, 3)

        labels = (labels.unsqueeze(1) > 0).float()
        norms, _ = (norms * labels).max(2)
        num_heatmap_items = heatmap_mask.sum()
        heat_loss = ((1 - norms) * heatmap_mask).sum() / num_heatmap_items
        losses_heat.update(heat_loss.item(), num_heatmap_items.int())

        loss = cls_loss + heat_loss * args.heat_loss

        # compute gradient and do optimizer step
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{}]\t'
                  'Loss: {:.4f} ({:.4f})\t'
                  'heatmap: {:.4f} ({:.4f})'.format(
                  epoch, batch_idx * num_items, len(train_loader.dataset),
                      losses.val * 100, losses.avg * 100,
                      losses_heat.val, losses_heat.avg))

def test(test_loader, model):
    model.eval()
    scores = []
    labels = []
    # for test/val data we get images only from the data loader
    for batch_idx, (images, im_labels) in tqdm(enumerate(test_loader), desc='testing...', total=len(test_loader)):
        if args.cuda:
            images = images.cuda()
        images = Variable(images)
        labels.append(im_labels)
        pred, _ = model(images)

        scores.append(pred.data.cpu())

    labels = torch.cat(labels).t().numpy() > 0
    scores = torch.cat(scores).t().numpy()
    ap = []
    for im_labels, im_scores in zip(labels, scores):
        if np.sum(im_labels) > 0:
            ap.append(skm.average_precision_score(im_labels, im_scores))

    mAP = np.mean(ap)

    print('\n{} set: mAP {:.1f}\n'.format(
        test_loader.dataset.split, round(mAP * 100, 1)))
    return mAP

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    directory = os.path.join("runs", args.dataset, args.name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = os.path.join(directory, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(directory, 'model_best.pth.tar'))

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * ((1 - 0.015) ** epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == '__main__':
    main()    
