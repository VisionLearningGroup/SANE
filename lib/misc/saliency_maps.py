import cv2
import torch
import numpy as np
from torch.autograd import Variable
from tqdm import tqdm
from skimage.transform import resize
from functools import partial

class SaliencyModel():
    def __init__(self, model):
        self.embeddingnet = model
        self.input_size = (112, 112)

    def run_on_batch(self, x, e, condition):
        if condition is None:
            m = self.embeddingnet(x)
        else:
            if isinstance(condition, int):
                condition = Variable(torch.from_numpy(np.ones(len(x), np.int64) * condition))
                if x.is_cuda:
                    condition = condition.cuda()

            m = self.embeddingnet(x, condition)

        sim = torch.nn.functional.cosine_similarity(m.data, e.expand_as(m)).squeeze()
        return sim#[1:] - sim[0]

def tv_norm(input, tv_beta):
    img = input[0, 0, :]
    row_grad = torch.mean(torch.abs((img[:-1 , :] - img[1 :, :])).pow(tv_beta))
    col_grad = torch.mean(torch.abs((img[: , :-1] - img[: , 1 :])).pow(tv_beta))
    return row_grad + col_grad

def mask_inference(upsampled_mask, img, img1_blur, model, condition):
    # The single channel mask is used with an RGB image,
    # so the mask is duplicated to have 3 channel,
    upsampled_mask = upsampled_mask.expand(1, 3, upsampled_mask.size(2),
                                           upsampled_mask.size(3))

    # Use the mask to perturbated the input image.
    perturbated_input = img.mul(upsampled_mask) + \
                        img1_blur.mul(1-upsampled_mask)

    noise = np.zeros((1, 3, 112, 112), dtype = np.float32)
    cv2.randn(noise, 0, 0.2)
    noise = Variable(torch.from_numpy(noise).cuda())
    perturbated_input = perturbated_input + noise
    if condition is None:
        e = model.embeddingnet(perturbated_input).squeeze().unsqueeze(0)
    else:
        e = model.embeddingnet(perturbated_input, condition).squeeze().unsqueeze(0)

    return e

def mask_explain_var(model, img, img1_blur, img2, img2_blur, mask_size, condition):
    tv_beta = 3
    learning_rate = 0.1
    max_iterations = 500
    l1_coeff = 0.01
    tv_coeff = 0.2
    mask_init = np.ones((1, 1, mask_size, mask_size * 2), dtype = np.float32)
    mask_init = torch.from_numpy(mask_init)
    if img.is_cuda:
        mask_init = mask_init.cuda()

    mask = Variable(mask_init, requires_grad=True)
    optimizer = torch.optim.Adam([mask], lr=learning_rate)
    if condition is not None:
        condition = Variable(torch.from_numpy(np.array([condition], np.int64)))
        if img.is_cuda:
            condition = condition.cuda()
    
    for i in range(max_iterations):
        upsampled_mask = torch.nn.functional.interpolate(mask, size=(112, 112 * 2), mode='bilinear', align_corners=True)
        e = mask_inference(upsampled_mask[:, :, :, :112], img, img1_blur, model, condition)
        embed1 = mask_inference(upsampled_mask[:, :, :, 112:], img2, img2_blur, model, condition)
        outputs = torch.nn.functional.cosine_similarity(embed1, e).squeeze()
        loss = l1_coeff*torch.mean(torch.abs(1 - mask)) + \
               tv_coeff*tv_norm(mask, tv_beta) + outputs

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Optional: clamping seems to give better results
        mask.data.clamp_(0, 1)

    return 1 - mask.data

def mask_explain(model, img, img1_blur, embed1, mask_size, condition):
    tv_beta = 3
    learning_rate = 0.1
    max_iterations = 500
    l1_coeff = 0.01
    tv_coeff = 0.2
    mask_init = np.ones((1, 1, mask_size, mask_size), dtype = np.float32)
    mask_init = torch.from_numpy(mask_init)
    if img.is_cuda:
        mask_init = mask_init.cuda()

    mask = Variable(mask_init, requires_grad=True)
    optimizer = torch.optim.Adam([mask], lr=learning_rate)
    if condition is not None:
        condition = Variable(torch.from_numpy(np.array([condition], np.int64)))
        if img.is_cuda:
            condition = condition.cuda()
  
    for i in range(max_iterations):
        upsampled_mask = torch.nn.functional.interpolate(mask, size=(112, 112), mode='bilinear', align_corners=True)
        # The single channel mask is used with an RGB image,
        # so the mask is duplicated to have 3 channel,
        e = mask_inference(upsampled_mask, img, img1_blur, model, condition)
        outputs = torch.nn.functional.cosine_similarity(embed1, e).squeeze()
        loss = l1_coeff*torch.mean(torch.abs(1 - mask)) + \
               tv_coeff*tv_norm(mask, tv_beta) + outputs

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Optional: clamping seems to give better results
        mask.data.clamp_(0, 1)

    return 1 - mask.data

def generate_masks(N, s, p1):
    input_size = (112, 112)
    cell_size = np.ceil(np.array(input_size) / s)
    up_size = (s + 1) * cell_size

    grid = np.random.rand(N, s, s) < p1
    grid = grid.astype('float32')

    masks = np.empty((N, 112, 112))
    for i in tqdm(range(N), desc='Generating masks'):
        # Random shifts
        x = np.random.randint(0, cell_size[0])
        y = np.random.randint(0, cell_size[1])
        # Linear upsampling and cropping
        masks[i, :, :] = resize(grid[i], up_size, order=1, mode='reflect',
                                anti_aliasing=False)[x:x + input_size[0], y:y + input_size[1]]

    masks = masks.reshape(-1, 1, 112, 112)
    return masks

def rise_explain(model, im, embed1, masks, N, p1, condition):
    preds = []
    # Make sure multiplication is being done for correct axes
    batch_size = 500
    inp = im.repeat(N, 1, 1, 1)
    masked = inp * masks.expand_as(inp)
    for i in range(0, N, batch_size):
        masked_input = torch.cat((im, masked[i:min(i+batch_size, N)]), 0)
        #preds.append(model.run_on_batch(masked_input, embed1, condition).data)
        preds.append(model.run_on_batch(masked[i:min(i+batch_size, N)], embed1, condition).data)

    preds = torch.cat(preds).view(-1, 1)
    sal = torch.sum(preds * masks.reshape(N, -1), dim=0).reshape(112, 112)
    times_location_sampled = masks.squeeze().mean(0)
    sal /= times_location_sampled
    #sal = sal / N / p1
    return sal

def lime_explain(model, img_t, embed1, condition, lime_explainer, score_fn, preprocess_transform, N_lime=1000):
    # Denormalize image
    cur_img = ((img_t[0] + 1) / 2 * 255.).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    batch_predict = partial(score_fn, model, embed1, condition)
    explanation = lime_explainer.explain_instance(cur_img, batch_predict, top_labels=1,
                                                  hide_color=0, num_samples=N_lime)
    le = sorted(explanation.local_exp[0], key=lambda x: x[0]) # Sort (region_id, region_saliency) by id
    assign_saliency = np.vectorize(lambda x: le[x][1])        # Function that returns region saliency given id
    sal = assign_saliency(explanation.segments)               # Assign saliency to each pixel
    sal = torch.Tensor(sal)
    return sal

def slide_explain(model, im, embed1, masks, condition):
    preds = []
    # Make sure multiplication is being done for correct axes
    batch_size = 500
    N = len(masks)
    side = int(N ** 0.5)
    assert N == side**2
    inp = im.repeat(N, 1, 1, 1)
    masked = inp * masks.expand_as(inp)
    for i in range(0, N, batch_size):
        masked_input = torch.cat((im, masked[i:min(i+batch_size, N)]), 0)
        preds.append(model.run_on_batch(masked_input, embed1, condition).data)
    sal = 1 - torch.cat(preds).view(side, side)
    return sal


def rise_explain_var(model, im, im2, masks, N, p1, condition):
    # Make sure multiplication is being done for correct axes
    batch_size = 500

    # typically K < batch_size
    K = len(masks) - N
    query_condition = None
    if condition is not None:
        reference_condition = torch.from_numpy(np.ones(K, np.int64) * condition)
        query_condition = torch.from_numpy(np.ones(batch_size, np.int64) * condition)
        if im.is_cuda:
            reference_condition, query_condition = reference_condition.cuda(), query_condition.cuda()
        reference_condition, query_condition = Variable(reference_condition), Variable(query_condition)

    im2_ref = im2.repeat(K, 1, 1, 1)
    im2_masked = im2_ref * masks[:K].expand_as(im2_ref)
    if condition is None:
        all_embed = model.embeddingnet(im2_masked).data
    else:
        all_embed = model.embeddingnet(im2_masked, reference_condition).data

    masks = masks[K:]
    inp = im.repeat(N, 1, 1, 1)
    masked = inp * masks.expand_as(inp)
    all_sal = []
    times_location_sampled = masks.squeeze().mean(0)
    for j, embed1 in enumerate(all_embed):
        embed1 = embed1.unsqueeze(0)
        preds = []
        for i in range(0, N, batch_size):
            preds.append(model.run_on_batch(masked[i:min(i+batch_size, N)], embed1, query_condition).data)

        preds = torch.cat(preds).view(-1, 1)
        sal = torch.sum(preds * masks.reshape(N, -1), dim=0).reshape(112, 112)
        sal /= times_location_sampled
        #sal = sal / N / p1
        all_sal.append(sal)

    all_sal = torch.stack(all_sal).mean(0)
    return all_sal

def slide_explain_var(model, im, im2, masks, ref_masks, condition):
    # Make sure multiplication is being done for correct axes
    batch_size = 500
    N = len(masks)
    side = int(N ** 0.5)
    assert N == side**2
    inp = im.repeat(N, 1, 1, 1)
    K = len(ref_masks)
    if condition is not None:
        reference_condition = torch.from_numpy(np.ones(K, np.int64) * condition)
        query_condition = torch.from_numpy(np.ones(batch_size, np.int64) * condition)
        if im.is_cuda:
            reference_condition, query_condition = reference_condition.cuda(), query_condition.cuda()
        reference_condition, query_condition = Variable(reference_condition), Variable(query_condition)

    im2_ref = im2.repeat(K, 1, 1, 1)
    im2_masked = im2_ref * ref_masks.expand_as(im2_ref)
    if condition is None:
        all_embed = model.embeddingnet(im2_masked).data
    else:
        all_embed = model.embeddingnet(im2_masked, reference_condition).data
    all_sal = []
    masked = inp * masks.expand_as(inp)
    for j, embed1 in enumerate(all_embed):
        embed1 = embed1.unsqueeze(0)
        preds = []
        for i in range(0, N, batch_size):
            A = min(i+batch_size, N)
            batch_condition = None
            if condition is not None:
                batch_condition = query_condition[:(A - i)]
            preds.append(model.run_on_batch(masked[i:A], embed1, batch_condition).data)
        sal = 1 - torch.cat(preds).view(side, side)
        all_sal.append(sal)

    all_sal = torch.stack(all_sal).mean(0)
    return all_sal
