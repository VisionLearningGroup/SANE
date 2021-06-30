from PIL import Image
import os
import numpy as np
import sklearn.metrics as skm
import torch
from torch.autograd import Variable
from torchvision import transforms
from tqdm import tqdm

from lime import lime_image
from misc.saliency_maps import *
from datasets.data_loader import DataLoader

class SaliencyDataLoader(DataLoader):
    def __init__(self, args, split, transform=None):
        super(SaliencyDataLoader, self).__init__(args, split, transform)
        self.mask_prob = args.mask_prob
        self.num_mask = args.num_mask
        self.max_num_heatmaps = args.num_heatmaps
        self._background_pixel_value = (255, 255, 255)
        self.all_thresh = np.arange(0, 1.01, 0.01)

        if args.method == 'rise':
            num_mask = self.num_mask
            if not self.fixed_ref:
                num_mask += 30

            self._masks = Variable(torch.from_numpy(generate_masks(num_mask, args.mask_size, args.mask_prob).astype(np.float32)))
            if args.cuda:
                self._masks = self._masks.cuda()

        elif args.method == 'slide':
            n = 25
            ff = np.zeros((n * n, 1, 112 + 16, 112 + 16))
            for i in range(n):
                for j in range(n):
                    ff[i * n + j, 0, i * 4:i * 4 + 32, j * 4:j * 4 + 32] = np.ones((32, 32))
            filters = 1 - ff[:, :, 8:-8, 8:-8]
            self._sliding_masks = Variable(torch.from_numpy(filters).float())
            if args.cuda:
                self._sliding_masks = self._sliding_masks.cuda()

            if not self.fixed_ref:
                n = 6
                ff = np.zeros((n * n, 1, 112, 112))
                for i in range(n):
                    for j in range(n):
                        ff[i * n + j, 0, i * 16:i * 16 + 32, j * 16:j * 16 + 32] = np.ones((32, 32))

                filters = 1 - ff
                self._reference_sliding_masks = Variable(torch.from_numpy(filters).float())
                if args.cuda:
                    self._reference_sliding_masks = self._reference_sliding_masks.cuda()
        elif args.method == 'lime':
            self._preprocess_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
            self._lime_explainer = lime_image.LimeImageExplainer()

            def score_fn(model, embed1, condition, images):
                batch = torch.stack(tuple(self._preprocess_transform(i) for i in images), dim=0)
                batch = batch.cuda()
                probs = model.run_on_batch(batch, embed1, condition).data
                probs = probs.detach().cpu().numpy()
                probs = np.expand_dims(probs, -1)
                return probs
            self._score_fn = score_fn
            self.N_lime = 1000
            

    def load_image_data(self, im, for_heatmap=False):
        img1_blur = None
        background = None
        img1 = self.loader(os.path.join(self.impath, im + '.jpg'))
        image_size = img1.size
        width = image_size[0]
        height = image_size[1]

        if for_heatmap and self.method == 'mask':
            original_img = np.array(img1)[:, :, ::-1].copy()
            blurred_img2 = cv2.medianBlur(original_img, 11)
            img1_blur = Image.fromarray(cv2.cvtColor(blurred_img2, cv2.COLOR_BGR2RGB))
        elif not for_heatmap:
            background = Image.new('RGB', (width, height), self._background_pixel_value)

        if self.transform is not None:
            img1 = self.transform(img1)
            if background is not None:
                background = self.transform(background)

            if img1_blur is not None:
                img1_blur = self.transform(img1_blur)

        if self.cuda:
            img1 = img1.cuda()
        img1 = Variable(torch.unsqueeze(img1, 0))

        if img1_blur is not None:
            if self.cuda:
                img1_blur = img1_blur.cuda()
            img1_blur = Variable(torch.unsqueeze(img1_blur, 0))

        if background is not None:
            if self.cuda:
                background = background.cuda()
            background = Variable(background.unsqueeze(0))

        return img1, img1_blur, background

    def sample_positive_pairs(self, index):
        raise NotImplementedError

    def compute_train_saliency_maps(self, model):
        if self.fixed_ref:
            ref_type = 'fixed'
        else:
            ref_type = 'var'

        all_pairs = {}
        for index, query in tqdm(enumerate(self.images),
                                           desc='%s %s %s train-caching' % (self.dataset, ref_type, self.method),
                                           total=len(self)):
            outfn = self.get_heatmap_cachefn(query)
            if os.path.exists(outfn):
                continue

            batch_pairs, pair_ims = self.sample_positive_pairs(index)
            all_heatmaps = []
            num_batches = int(np.ceil(len(batch_pairs) / 100.))
            for x in range(num_batches):
                pairs = batch_pairs[x * 100: (x+1) * 100]
                reference_images = torch.stack([self.__getitem__(i) for i in pairs])
                if self.cuda:
                    reference_images = reference_images.cuda()
                reference_images = Variable(reference_images)
                embeddings = model.embeddingnet(reference_images).data
                heatmap = torch.stack([self.compute_heatmap(model, query, self.images[i], embeds, False) for i, embeds in zip(pairs, embeddings)]).unsqueeze(1)
                heatmap = torch.nn.functional.interpolate(heatmap, size=(28, 28), mode='bilinear', align_corners=True).squeeze().cpu().numpy()
                all_heatmaps.append(heatmap)

            heatmap = np.concatenate(all_heatmaps, axis=0)
            np.save(outfn, heatmap)

    def compute_heatmap(self, model, query, reference, embeddings, all_embeddings = True):
        condition = self.get_typespace(query, reference)
        img1, img1_blur, _ = self.load_image_data(query, for_heatmap=True)
        if self.fixed_ref:
            if all_embeddings:
                reference_embedding = embeddings[self.image2index[reference]]
            else:
                reference_embedding = embeddings

            if condition is not None:
                reference_embedding = reference_embedding[condition]

            reference_embedding = reference_embedding.unsqueeze(0)
            if self.method == 'mask':
                heatmap = mask_explain(model, img1, img1_blur, reference_embedding, self.mask_size, condition)
            elif self.method == 'rise':
                heatmap = rise_explain(model, img1, reference_embedding, self._masks, self.num_mask, self.mask_prob, condition)
            elif self.method == 'slide':
                heatmap = slide_explain(model, img1, reference_embedding, self._sliding_masks, condition)
            elif self.method == 'lime':
                heatmap = lime_explain(model, img1, reference_embedding, condition, self._lime_explainer, self._score_fn, self._preprocess_transform, self.N_lime)
            else:
                raise ValueError('Unrecognized saliency method: ' + self.method)
        else:
            img2, img2_blur, _ = self.load_image_data(reference, for_heatmap=True)
            if self.method == 'mask':
                heatmap = mask_explain_var(model, img1, img1_blur, img2, img2_blur, self.mask_size, condition)
            elif self.method == 'rise':
                heatmap = rise_explain_var(model, img1, img2, self._masks, self.num_mask, self.mask_prob, condition)
            elif self.method == 'slide':
                heatmap = slide_explain_var(model, img1, img2, self._sliding_masks, self._reference_sliding_masks, condition)
            elif self.method == 'lime':
                raise NotImplementedError('lime_var is not implemented')
            else:
                raise ValueError('Unrecognized saliency method: ' + self.method)

        return heatmap

    def process_heatmap(self, heatmap):
        if heatmap is None:
            return heatmap

        if self.method in ['slide', 'mask']:
            w, h = heatmap.size()[-2:]
            heatmap = heatmap.view(-1, 1, w, h)
            heatmap = torch.nn.functional.interpolate(heatmap, size=(112, 112), mode='bilinear', align_corners=True)

        return heatmap.view(-1)

    def load_heatmap(self, img1, img2, model, embeddings):
        heatmap = super(SaliencyDataLoader, self).load_heatmap(img1, img2, model, embeddings)
        if heatmap is None:
            heatmap = self.compute_heatmap(model, img1, img2, embeddings)
            if self.method == 'mask' and not self.fixed_ref:
                heatmap1, heatmap2 = heatmap[:, :, :, :self.mask_size], heatmap[:, :, :, self.mask_size:]
                heatmap = heatmap1
                np.save(self.get_heatmap_cachefn(img2, img1), heatmap2.cpu().numpy())

            np.save(self.get_heatmap_cachefn(img1, img2), heatmap.cpu().numpy())
            heatmap = self.process_heatmap(heatmap)
           
        return heatmap

    def get_masked_embeddings(self, model, img1, heatmap, background, condition):
        all_img1 = {'insert' : [], 'delete' : []}
        img1_blur = img1
        for heatmap_thresh in self.all_thresh:
            n_items = max(int(round(heatmap.size(0) * heatmap_thresh)), 1)
            for eval_type in ['insert', 'delete']:
                if eval_type == 'insert':
                    _, indices = torch.topk(heatmap, n_items)
                    new_heatmap = torch.zeros(heatmap.size(0))
                    new_heatmap[indices] = 1
                    metric_img = img1_blur
                else:
                    _, indices = torch.topk(heatmap, n_items)
                    new_heatmap = torch.ones(heatmap.size(0))
                    new_heatmap[indices] = 0
                    metric_img = img1
               
                if self.cuda:
                    new_heatmap = new_heatmap.cuda()
                new_heatmap = Variable(new_heatmap.view(112, 112))
                masked = metric_img * new_heatmap.expand_as(metric_img) + background * (new_heatmap == 0).float().expand_as(metric_img)
                all_img1[eval_type].append(masked)

        all_img1 = torch.stack(all_img1['insert'] + all_img1['delete']).squeeze()
        if condition is None:
            masked_embed = model.embeddingnet(all_img1).data
        else:
            condition = Variable(torch.from_numpy(np.ones(len(all_img1), np.int64) * condition))
            if self.cuda:
                condition = condition.cuda()
            masked_embed = model.embeddingnet(all_img1, condition).data

        return masked_embed

    def test(self, model, embeddings):
        n_thresh = len(self.all_thresh)
        delete_scores = np.zeros((len(self.pairs), n_thresh), np.float32)
        insert_scores = np.zeros_like(delete_scores)
        if self.fixed_ref:
            ref_type = 'fixed'
        else:
            ref_type = 'var'

        for i, (img1, img2, label) in tqdm(enumerate(self.pairs),
                                           desc='%s %s %s test' % (self.dataset, ref_type, self.method),
                                           total=len(self.pairs)):
            heatmap1 = self.load_heatmap(img1, img2, model, embeddings)
            query, _, background = self.load_image_data(img1)
            reference, _, _ = self.load_image_data(img2)
            condition = self.get_typespace(img1, img2)
            embed1 = self.get_masked_embeddings(model, query, heatmap1, background, condition)
            embed2 = embeddings[self.image2index[img2]]
            if condition is not None:
                embed2 = embed2[condition]

            embed2 = embed2.unsqueeze(0)
            sim = torch.nn.functional.cosine_similarity(embed1, embed2.expand_as(embed1)).cpu().numpy()
            sim_insert = sim[:n_thresh]
            sim_insert -= min(sim_insert)
            insert_scores[i, :] = sim_insert / max(sim_insert)

            sim_delete = sim[n_thresh:]
            sim_delete -= min(sim_delete)
            delete_scores[i, :] = sim_delete / max(sim_delete)

        insert_auc = skm.auc(self.all_thresh, np.mean(insert_scores, 0))
        delete_auc = skm.auc(self.all_thresh, np.mean(delete_scores, 0))
        return insert_auc, delete_auc

