# The ema model and the domain-mixing are based on:
# https://github.com/vikolss/DACS

import math
import os
import random
from copy import deepcopy

import mmcv
import numpy as np
import torch
from matplotlib import pyplot as plt
from timm.models.layers import DropPath
from torch.nn.modules.dropout import _DropoutNd
import torch.nn.functional as F

from mmseg.core import add_prefix
from mmseg.models import UDA, build_segmentor
from mmseg.models.uda.uda_decorator import UDADecorator, get_module
from mmseg.models.utils.dacs_transforms import (denorm, get_class_masks,
                                                get_mean_std, strong_transform)
from mmseg.models.utils.visualization import subplotimg
from mmseg.utils.utils import downscale_label_ratio


def _params_equal(ema_model, model):
    for ema_param, param in zip(ema_model.named_parameters(),
                                model.named_parameters()):
        if not torch.equal(ema_param[1].data, param[1].data):
            # print("Difference in", ema_param[0])
            return False
    return True


def calc_grad_magnitude(grads, norm_type=2.0):
    norm_type = float(norm_type)
    if norm_type == math.inf:
        norm = max(p.abs().max() for p in grads)
    else:
        norm = torch.norm(
            torch.stack([torch.norm(p, norm_type) for p in grads]), norm_type)

    return norm


@UDA.register_module()
class DACS_CDA(UDADecorator):

    def __init__(self, **cfg):
        super(DACS_CDA, self).__init__(**cfg)
        self.local_iter = 0
        self.max_iters = cfg['max_iters']
        self.alpha = cfg['alpha']
        self.pseudo_threshold = cfg['pseudo_threshold']
        self.psweight_ignore_top = cfg['pseudo_weight_ignore_top']
        self.psweight_ignore_bottom = cfg['pseudo_weight_ignore_bottom']
        self.fdist_lambda = cfg['imnet_feature_dist_lambda']
        self.fdist_classes = cfg['imnet_feature_dist_classes']
        self.fdist_scale_min_ratio = cfg['imnet_feature_dist_scale_min_ratio']
        self.enable_fdist = self.fdist_lambda > 0
        self.blur = cfg['blur']
        self.color_jitter_s = cfg['color_jitter_strength']
        self.color_jitter_p = cfg['color_jitter_probability']
        self.debug_img_interval = cfg['debug_img_interval']
        self.print_grad_magnitude = cfg['print_grad_magnitude']
        self.cda_src_lambda = eval(cfg['cda_src_lambda'])
        self.cda_tgt_lambda = eval(cfg['cda_tgt_lambda'])
        self.cda_s2t_lambda = eval(cfg['cda_s2t_lambda'])
        self.cda_t2s_lambda = eval(cfg['cda_t2s_lambda'])
        self.attn_lambda = cfg['attn_lambda']
        self.debug_gt_rescale = None
        self.debug_fdist_mask = None

        self.class_probs = {}
        ema_cfg = deepcopy(cfg['model'])
        self.ema_model = build_segmentor(ema_cfg)
        if self.enable_fdist:
            self.imnet_model = build_segmentor(deepcopy(cfg['model']))
        else:
            self.imnet_model = None
        self.valid_masking = cfg['valid_masking']
        self.src_attn_tea = cfg['src_attn_tea']

    def get_ema_model(self):
        return get_module(self.ema_model)

    def get_imnet_model(self):
        return get_module(self.imnet_model)

    def _init_ema_weights(self):
        for param in self.get_ema_model().parameters():
            param.detach_()
        mp = list(self.get_model().parameters())
        mcp = list(self.get_ema_model().parameters())
        for i in range(0, len(mp)):
            if not mcp[i].data.shape:  # scalar tensor
                mcp[i].data = mp[i].data.clone()
            else:
                mcp[i].data[:] = mp[i].data[:].clone()

    def _update_ema(self, iter):
        alpha_teacher = min(1 - 1 / (iter + 1), self.alpha)
        for ema_param, param in zip(self.get_ema_model().parameters(),
                                    self.get_model().parameters()):
            if not param.data.shape:  # scalar tensor
                ema_param.data = \
                    alpha_teacher * ema_param.data + \
                    (1 - alpha_teacher) * param.data
            else:
                ema_param.data[:] = \
                    alpha_teacher * ema_param[:].data[:] + \
                    (1 - alpha_teacher) * param[:].data[:]

    def train_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """

        optimizer.zero_grad()
        log_vars = self(**data_batch)
        optimizer.step()

        log_vars.pop('loss', None)  # remove the unnecessary 'loss'
        outputs = dict(
            log_vars=log_vars, num_samples=len(data_batch['img_metas']))
        return outputs

    def masked_feat_dist(self, f1, f2, mask=None):
        feat_diff = f1 - f2
        # mmcv.print_log(f'fdiff: {feat_diff.shape}', 'mmseg')
        pw_feat_dist = torch.norm(feat_diff, dim=1, p=2)
        # mmcv.print_log(f'pw_fdist: {pw_feat_dist.shape}', 'mmseg')
        if mask is not None:
            # mmcv.print_log(f'fd mask: {mask.shape}', 'mmseg')
            pw_feat_dist = pw_feat_dist[mask.squeeze(1)]
            # mmcv.print_log(f'fd masked: {pw_feat_dist.shape}', 'mmseg')
        return torch.mean(pw_feat_dist)

    def compute_attn_loss(self, src_attns, mix_attns, tgt_attns, mix_masks, valid_masking=False):
        mix_masks = torch.cat(mix_masks)
        B = mix_masks.shape[0]
        mix_masks_cache = {}
        for src_attn in src_attns:
            for _len in [src_attn.shape[2], src_attn.shape[3]]:
                if _len not in mix_masks_cache:
                    target_len = int(_len ** 0.5)
                    mix_masks_cache[_len] = F.interpolate(mix_masks.float(), size=(target_len, target_len), mode='nearest')

        losses = []
        for index in range(len(mix_attns)): # 1 src, 0 tgt
            B, K, N, N_prime = mix_attns[index].shape
            _mix_masks = mix_masks_cache[N].reshape(B, 1, -1, 1)  # B, 1, N, 1

            mix_attns_sup = src_attns[index] * _mix_masks + tgt_attns[index] * (1 - _mix_masks) # B, nHead, N, N'
            if valid_masking:
                loss = F.kl_div(mix_attns[index].log(), mix_attns_sup, reduction='none')
                small_mask = mix_masks_cache[N_prime].reshape(B, 1, 1, -1)  # B, 1, 1, N'
                weight = _mix_masks * small_mask + (1 - _mix_masks) * (1 - small_mask) # (B, 1, N, N')  
                loss = (loss * weight).mean() * self.attn_lambda
            else:
                loss = F.kl_div(mix_attns[index].log(), mix_attns_sup, reduction='mean') * self.attn_lambda
    
            losses.append(loss)
        attn_loss, attn_log = self._parse_losses(
            {'loss_attn': sum(losses)/len(losses)})
        attn_log.pop('loss', None)
        return attn_loss, attn_log

    def calc_feat_dist(self, img, gt, feat=None):
        assert self.enable_fdist
        with torch.no_grad():
            self.get_imnet_model().eval()
            feat_imnet = self.get_imnet_model().extract_feat(img)
            feat_imnet = [f.detach() for f in feat_imnet]
        lay = -1
        if self.fdist_classes is not None:
            fdclasses = torch.tensor(self.fdist_classes, device=gt.device)
            scale_factor = gt.shape[-1] // feat[lay].shape[-1]
            gt_rescaled = downscale_label_ratio(gt, scale_factor,
                                                self.fdist_scale_min_ratio,
                                                self.num_classes,
                                                255).long().detach()
            fdist_mask = torch.any(gt_rescaled[..., None] == fdclasses, -1)
            feat_dist = self.masked_feat_dist(feat[lay], feat_imnet[lay],
                                              fdist_mask)
            self.debug_fdist_mask = fdist_mask
            self.debug_gt_rescale = gt_rescaled
        else:
            feat_dist = self.masked_feat_dist(feat[lay], feat_imnet[lay])
        feat_dist = self.fdist_lambda * feat_dist
        feat_loss, feat_log = self._parse_losses(
            {'loss_imnet_feat_dist': feat_dist})
        feat_log.pop('loss', None)
        return feat_loss, feat_log

    def forward_train(self, img, img_metas, gt_semantic_seg, target_img,
                      target_img_metas):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        log_vars = {}
        batch_size = img.shape[0]
        dev = img.device

        # Init/update ema model
        if self.local_iter == 0:
            self._init_ema_weights()
            # assert _params_equal(self.get_ema_model(), self.get_model())

        if self.local_iter > 0:
            self._update_ema(self.local_iter)
            # assert not _params_equal(self.get_ema_model(), self.get_model())
            # assert self.get_ema_model().training

        means, stds = get_mean_std(img_metas, dev)
        strong_parameters = {
            'color_jitter': random.uniform(0, 1),
            'color_jitter_s': self.color_jitter_s,
            'color_jitter_p': self.color_jitter_p,
            'blur': random.uniform(0, 1) if self.blur else 0,
            'mean': means[0].unsqueeze(0),  # assume same normalization
            'std': stds[0].unsqueeze(0)
        }

        require_attn = self.attn_lambda > 0
        # Generate pseudo-label
        for m in self.get_ema_model().modules():
            if isinstance(m, _DropoutNd):
                m.training = False
            if isinstance(m, DropPath):
                m.training = False
        with torch.no_grad():
            ema_logits = self.get_ema_model().encode_decode(
                target_img, target_img_metas, require_attn=require_attn)
            if require_attn:
                ema_logits, tgt_attns = ema_logits
                # tgt_attns = [i.detach() for i in tgt_attns]
                if self.src_attn_tea:
                    ema_logits_src = self.get_ema_model().encode_decode(
                        img, img_metas, require_attn=require_attn)
                    ema_logits_src, src_attns = ema_logits_src

        ema_softmax = torch.softmax(ema_logits, dim=1)
        pseudo_prob, pseudo_label = torch.max(ema_softmax, dim=1)
        ps_large_p = pseudo_prob.ge(self.pseudo_threshold).long() == 1
        ps_size = np.size(np.array(pseudo_label.cpu()))
        pseudo_weight_tgt = torch.sum(ps_large_p).item() / ps_size * torch.ones(pseudo_prob.shape, device=dev)
        pseudo_weight_t2s = torch.sum(ps_large_p).item() / ps_size * torch.ones(pseudo_prob.shape, device=dev)            
        pseudo_weight_s2t = None

        if self.psweight_ignore_top > 0:
            # Don't trust pseudo-labels in regions with potential
            # rectification artifacts. This can lead to a pseudo-label
            # drift from sky towards building or traffic light.
            pseudo_weight_tgt[:, :self.psweight_ignore_top, :] = 0
            pseudo_weight_t2s[:, :self.psweight_ignore_top, :] = 0
        if self.psweight_ignore_bottom > 0:
            pseudo_weight_tgt[:, -self.psweight_ignore_bottom:, :] = 0
            pseudo_weight_t2s[:, -self.psweight_ignore_bottom:, :] = 0

        gt_pixel_weight = torch.ones((pseudo_weight_tgt.shape), device=dev)

        # Apply mixing
        mixed_img, mixed_lbl = [None] * batch_size, [None] * batch_size
        mix_masks = get_class_masks(gt_semantic_seg)

        for i in range(batch_size):
            strong_parameters['mix'] = mix_masks[i]
            mixed_img[i], mixed_lbl[i] = strong_transform(
                strong_parameters,
                data=torch.stack((img[i], target_img[i])),
                target=torch.stack((gt_semantic_seg[i][0], pseudo_label[i])))
            _, pseudo_weight_tgt[i] = strong_transform(
                strong_parameters,
                target=torch.stack((gt_pixel_weight[i], pseudo_weight_tgt[i])))
            _, pseudo_weight_t2s[i] = strong_transform(
                strong_parameters,
                target=torch.stack((gt_pixel_weight[i], pseudo_weight_t2s[i])))

        mixed_img = torch.cat(mixed_img)
        mixed_lbl = torch.cat(mixed_lbl)

        tgt_gt = mixed_lbl

        cda_losses = self.get_model().forward_train_four_branches(
            src=img, src_metas=img_metas, src_gt=gt_semantic_seg, 
            src_seg_weight=None, s2t_seg_weight=pseudo_weight_s2t,
            tgt=mixed_img, tgt_metas=target_img_metas, tgt_gt=tgt_gt,
            tgt_seg_weight=pseudo_weight_tgt, t2s_seg_weight=pseudo_weight_t2s, 
            src_lambda=self.cda_src_lambda(self.local_iter, self.max_iters), 
            tgt_lambda=self.cda_tgt_lambda(self.local_iter, self.max_iters), 
            s2t_lambda=self.cda_s2t_lambda(self.local_iter, self.max_iters), 
            t2s_lambda=self.cda_t2s_lambda(self.local_iter, self.max_iters), 
            return_feat=True, require_attn=require_attn)

        src_feat = cda_losses.pop('src_features')
        tgt_feat = cda_losses.pop('tgt_features')
        s2t_feat = cda_losses.pop('s2t_features')
        t2s_feat = cda_losses.pop('t2s_features')
        if require_attn:
            if self.src_attn_tea:
                cda_losses.pop('src_attns')
            else:
                src_attns = cda_losses.pop('src_attns')
                src_attns = [i.detach() for i in src_attns]
            mix_attns = cda_losses.pop('mix_attns')

        # wgt_s2t = pseudo_weight_s2t.mean() if pseudo_weight_s2t is not None else torch.tensor(1.)
        # wgt_t2s = pseudo_weight_t2s.mean() if pseudo_weight_t2s is not None else torch.tensor(1.)
        # wgt_tgt = pseudo_weight_tgt.mean() if pseudo_weight_tgt is not None else torch.tensor(1.)

        # wgt_debug = {"wgt_s2t": wgt_s2t, "wgt_t2s": wgt_t2s, "wgt_tgt": wgt_tgt}
        # cda_losses.update(wgt_debug)
    
        cda_losses = add_prefix(cda_losses, 'cda')
        cda_loss, cda_log_vars = self._parse_losses(cda_losses)
        log_vars.update(cda_log_vars)

        cda_loss.backward(retain_graph=(self.enable_fdist or require_attn))
        
        if require_attn:
            attn_loss, attn_log = self.compute_attn_loss(src_attns, mix_attns, tgt_attns, mix_masks, self.valid_masking)
            attn_loss.backward(retain_graph=self.enable_fdist)
            log_vars.update(add_prefix(attn_log, 'attn'))

        # ImageNet feature distance
        if self.enable_fdist:
            feat_loss, feat_log = self.calc_feat_dist(img, gt_semantic_seg,
                                                      src_feat)
            feat_loss.backward()
            log_vars.update(add_prefix(feat_log, 'src'))
            if self.print_grad_magnitude:
                params = self.get_model().backbone.parameters()
                fd_grads = [
                    p.grad.detach() for p in params if p.grad is not None
                ]
                fd_grads = [g2 - g1 for g1, g2 in zip(seg_grads, fd_grads)]
                grad_mag = calc_grad_magnitude(fd_grads)
                mmcv.print_log(f'Fdist Grad.: {grad_mag}', 'mmseg')

        if self.local_iter % self.debug_img_interval == 0:
            out_dir = os.path.join(self.train_cfg['work_dir'],
                                   'class_mix_debug')
            os.makedirs(out_dir, exist_ok=True)
            vis_img = torch.clamp(denorm(img, means, stds), 0, 1)
            vis_trg_img = torch.clamp(denorm(target_img, means, stds), 0, 1)
            vis_aug_img = torch.clamp(denorm(mixed_img, means, stds), 0, 1)
            for j in range(batch_size):
                rows, cols = 2, 4
                fig, axs = plt.subplots(
                    rows,
                    cols,
                    figsize=(3 * cols, 3 * rows),
                    gridspec_kw={
                        'hspace': 0.1,
                        'wspace': 0,
                        'top': 0.95,
                        'bottom': 0,
                        'right': 1,
                        'left': 0
                    },
                )
                subplotimg(axs[0][0], vis_img[j], 'Source Image')
                subplotimg(axs[1][0], vis_trg_img[j], 'Target Image')
                subplotimg(
                    axs[0][1],
                    gt_semantic_seg[j],
                    'Source Seg GT',
                    cmap='cityscapes')
                subplotimg(
                    axs[1][1],
                    pseudo_label[j],
                    'Target Seg (Pseudo) GT',
                    cmap='cityscapes')
                # subplotimg(axs[0][2], vis_mixed_img[j], 'Mixed Image')
                # subplotimg(
                #     axs[1][2], mix_masks[j][0], 'Domain Mask', cmap='gray')
                # subplotimg(axs[0][3], pred_u_s[j], "Seg Pred",
                #            cmap="cityscapes")
                # subplotimg(
                #     axs[1][3], mixed_lbl[j], 'Seg Targ', cmap='cityscapes')
                subplotimg(
                    axs[0][2],
                    vis_aug_img[j],
                    'Mixed Image')
                subplotimg(
                    axs[1][2], mix_masks[j][0], 'Domain Mask', cmap='gray')
                subplotimg(
                    axs[0][3], mixed_lbl[j], 'Seg Targ', cmap='cityscapes')
                subplotimg(
                    axs[1][3], 
                    pseudo_weight_tgt[j], 
                    'Pseudo W.', 
                    vmin=0, vmax=1)

                for ax in axs.flat:
                    ax.axis('off')
                plt.savefig(
                    os.path.join(out_dir,
                                 f'{(self.local_iter + 1):06d}_{j}.png'))
                plt.close()

            out_dir_4_branch = os.path.join(self.train_cfg['work_dir'],
                                   '4_branch_debug')
            os.makedirs(out_dir_4_branch, exist_ok=True)
            vis_img = torch.clamp(denorm(img, means, stds), 0, 1)
            vis_trg_img = torch.clamp(denorm(target_img, means, stds), 0, 1)
            vis_aug_img = torch.clamp(denorm(mixed_img, means, stds), 0, 1)
            for j in range(batch_size):
                rows, cols = 4, 4
                fig, axs = plt.subplots(
                    rows,
                    cols,
                    figsize=(3 * cols, 3 * rows),
                    gridspec_kw={
                        'hspace': 0.1,
                        'wspace': 0,
                        'top': 0.95,
                        'bottom': 0,
                        'right': 1,
                        'left': 0
                    },
                )
                vis_src_feat_b1 = src_feat[0][j].mean(0) # C, H, W
                vis_src_feat_b2 = src_feat[1][j].mean(0) # C, H, W
                vis_src_feat_b3 = src_feat[2][j].mean(0) # C, H, W
                vis_src_feat_b4 = src_feat[3][j].mean(0) # C, H, W
                vis_tgt_feat_b1 = tgt_feat[0][j].mean(0) # C, H, W
                vis_tgt_feat_b2 = tgt_feat[1][j].mean(0) # C, H, W
                vis_tgt_feat_b3 = tgt_feat[2][j].mean(0) # C, H, W
                vis_tgt_feat_b4 = tgt_feat[3][j].mean(0) # C, H, W
                vis_s2t_feat_b1 = s2t_feat[0][j].mean(0) # C, H, W
                vis_s2t_feat_b2 = s2t_feat[1][j].mean(0) # C, H, W
                vis_s2t_feat_b3 = s2t_feat[2][j].mean(0) # C, H, W
                vis_s2t_feat_b4 = s2t_feat[3][j].mean(0) # C, H, W
                vis_t2s_feat_b1 = t2s_feat[0][j].mean(0) # C, H, W
                vis_t2s_feat_b2 = t2s_feat[1][j].mean(0) # C, H, W
                vis_t2s_feat_b3 = t2s_feat[2][j].mean(0) # C, H, W
                vis_t2s_feat_b4 = t2s_feat[3][j].mean(0) # C, H, W
                b1_min = min(vis_src_feat_b1.min(), vis_tgt_feat_b1.min(), vis_s2t_feat_b1.min(), vis_t2s_feat_b1.min())
                b1_max = max(vis_src_feat_b1.max(), vis_tgt_feat_b1.max(), vis_s2t_feat_b1.max(), vis_t2s_feat_b1.max())
                b2_min = min(vis_src_feat_b2.min(), vis_tgt_feat_b2.min(), vis_s2t_feat_b2.min(), vis_t2s_feat_b2.min())
                b2_max = max(vis_src_feat_b2.max(), vis_tgt_feat_b2.max(), vis_s2t_feat_b2.max(), vis_t2s_feat_b2.max())
                b3_min = min(vis_src_feat_b3.min(), vis_tgt_feat_b3.min(), vis_s2t_feat_b3.min(), vis_t2s_feat_b3.min())
                b3_max = max(vis_src_feat_b3.max(), vis_tgt_feat_b3.max(), vis_s2t_feat_b3.max(), vis_t2s_feat_b3.max())
                b4_min = min(vis_src_feat_b4.min(), vis_tgt_feat_b4.min(), vis_s2t_feat_b4.min(), vis_t2s_feat_b4.min())
                b4_max = max(vis_src_feat_b4.max(), vis_tgt_feat_b4.max(), vis_s2t_feat_b4.max(), vis_t2s_feat_b4.max())
                subplotimg(axs[0][0], (vis_src_feat_b1 - b1_min)/(b1_max - b1_min), 'Source block 1')
                subplotimg(axs[0][1], (vis_src_feat_b2 - b2_min)/(b2_max - b2_min), 'Source block 2')
                subplotimg(axs[0][2], (vis_src_feat_b3 - b3_min)/(b3_max - b3_min), 'Source block 3')
                subplotimg(axs[0][3], (vis_src_feat_b4 - b4_min)/(b4_max - b4_min), 'Source block 4')
                subplotimg(axs[1][0], (vis_tgt_feat_b1 - b1_min)/(b1_max - b1_min), 'Target block 1')
                subplotimg(axs[1][1], (vis_tgt_feat_b2 - b2_min)/(b2_max - b2_min), 'Target block 2')
                subplotimg(axs[1][2], (vis_tgt_feat_b3 - b3_min)/(b3_max - b3_min), 'Target block 3')
                subplotimg(axs[1][3], (vis_tgt_feat_b4 - b4_min)/(b4_max - b4_min), 'Target block 4')
                subplotimg(axs[2][0], (vis_s2t_feat_b1 - b1_min)/(b1_max - b1_min), 'S2T block 1')
                subplotimg(axs[2][1], (vis_s2t_feat_b2 - b2_min)/(b2_max - b2_min), 'S2T block 2')
                subplotimg(axs[2][2], (vis_s2t_feat_b3 - b3_min)/(b3_max - b3_min), 'S2T block 3')
                subplotimg(axs[2][3], (vis_s2t_feat_b4 - b4_min)/(b4_max - b4_min), 'S2T block 4')
                subplotimg(axs[3][0], (vis_t2s_feat_b1 - b1_min)/(b1_max - b1_min), 'T2S block 1')
                subplotimg(axs[3][1], (vis_t2s_feat_b2 - b2_min)/(b2_max - b2_min), 'T2S block 2')
                subplotimg(axs[3][2], (vis_t2s_feat_b3 - b3_min)/(b3_max - b3_min), 'T2S block 3')
                subplotimg(axs[3][3], (vis_t2s_feat_b4 - b4_min)/(b4_max - b4_min), 'T2S block 4')

                for ax in axs.flat:
                    ax.axis('off')
                plt.savefig(
                    os.path.join(out_dir_4_branch,
                                 f'{(self.local_iter + 1):06d}_{j}.png'))
                plt.close()

        self.local_iter += 1

        return log_vars
