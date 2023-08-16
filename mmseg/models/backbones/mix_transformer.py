# Obtained from: https://github.com/NVlabs/SegFormer
# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
# A copy of the license is available at resources/license_segformer

import math
import warnings
from functools import partial

import torch
import torch.nn as nn
from mmcv.runner import BaseModule, _load_checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from mmseg.models.builder import BACKBONES
from mmseg.utils import get_root_logger

import torch.nn.functional as F
import numpy as np

class Mlp(nn.Module):

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.,
                 CB=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.CB = CB

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        if self.CB:
            x = (x + x.mean(dim=1, keepdim=True)) * 0.5
        return x


class Attention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 sr_ratio=1,
                 attn_noise=0):
        super().__init__()
        assert dim % num_heads == 0, f'dim {dim} should be divided by ' \
                                     f'num_heads {num_heads}.'

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        self.attn_noise = attn_noise
        if sr_ratio > 1:
            self.sr = nn.Conv2d(
                dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W, x_cd=None, t2s=None, s2t=None, require_attn=False, switch=True, detach=False, switch_element="q"):
        B, N, C = x.shape
        q_x = self.q(x).reshape(B, N, self.num_heads,
                              C // self.num_heads).permute(0, 2, 1, 3).contiguous() # (B, N, C) -> (B, Head, N, dim_each_head)

        if self.sr_ratio > 1:
            x_ori = x.permute(0, 2, 1).contiguous().reshape(B, C, H, W)
            x_ori_sr = self.sr(x_ori).reshape(B, C, -1).permute(0, 2, 1).contiguous() # (B, C, H, W) -> (B, N', C)
            x_ori_sr = self.norm(x_ori_sr)
            kv_x = self.kv(x_ori_sr).reshape(B, -1, 2, self.num_heads,
                                     C // self.num_heads).permute(
                                         2, 0, 3, 1, 4).contiguous() # (B, N', C) -> (B, N', 2, n_head, C/n_head) -> (2, B, n_head, N', C/n_head)
        else:
            kv_x = self.kv(x).reshape(B, -1, 2, self.num_heads,
                                    C // self.num_heads).permute(
                                        2, 0, 3, 1, 4).contiguous()
        k_x, v_x = kv_x[0], kv_x[1]

        attn_x = (q_x @ k_x.transpose(-2, -1).contiguous()) * self.scale
        attn_x = attn_x.softmax(dim=-1)
        if self.attn_noise > 0 and self.training and np.random.rand() < self.attn_noise:
            assert self.attn_noise <= 1 # attn shape: (B, nHead, N, dim_each_head)
            _B, _nHead, _N, _dim_each_head = attn_x.shape
            _l = int(_dim_each_head ** 0.5)
            _stride = np.random.choice([2,4,8])
            noise_mat = torch.randn(_B, _nHead * _N, _l//_stride, _l//_stride, device=attn_x.device)
            noise_mat = F.interpolate(noise_mat, size=(_l, _l), mode='bilinear')
            noise_mat = noise_mat.view(_B, _nHead * _N, -1).softmax(dim=-1).view(_B, _nHead, _N, _dim_each_head)
            attn_x = (attn_x + noise_mat) / 2

        attn_x_drop = self.attn_drop(attn_x)

        x_out = (attn_x_drop @ v_x).transpose(1, 2).contiguous().reshape(B, N, C)
        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)

        if x_cd is None:
            return x_out, attn_x

        ####

        q_x_cd = self.q(x_cd).reshape(B, N, self.num_heads,
                              C // self.num_heads).permute(0, 2, 1, 3).contiguous()

        if self.sr_ratio > 1:
            x_cd_ori = x_cd.permute(0, 2, 1).contiguous().reshape(B, C, H, W)
            x_cd_ori_sr = self.sr(x_cd_ori).reshape(B, C, -1).permute(0, 2, 1).contiguous()
            x_cd_ori_sr = self.norm(x_cd_ori_sr)
            kv_x_cd = self.kv(x_cd_ori_sr).reshape(B, -1, 2, self.num_heads,
                                     C // self.num_heads).permute(
                                         2, 0, 3, 1, 4).contiguous()
        else:
            kv_x_cd = self.kv(x_cd).reshape(B, -1, 2, self.num_heads,
                                    C // self.num_heads).permute(
                                        2, 0, 3, 1, 4).contiguous() # 

        k_x_cd, v_x_cd = kv_x_cd[0], kv_x_cd[1]


        attn_x_cd = (q_x_cd @ k_x_cd.transpose(-2, -1).contiguous()) * self.scale
        attn_x_cd = attn_x_cd.softmax(dim=-1)
        attn_x_cd_drop = self.attn_drop(attn_x_cd)

        x_cd_out = (attn_x_cd_drop @ v_x_cd).transpose(1, 2).contiguous().reshape(B, N, C)
        x_cd_out = self.proj(x_cd_out)
        x_cd_out = self.proj_drop(x_cd_out)

        if "q" in switch_element.lower():
            q_t2s, q_s2t = q_x_cd, q_x
        else:
            q_t2s, q_s2t = q_x, q_x_cd

        if "k" in switch_element.lower():
            k_t2s, k_s2t = k_x_cd, k_x
        else:
            k_t2s, k_s2t = k_x, k_x_cd

        if "v" in switch_element.lower():
            v_t2s, v_s2t = v_x_cd, v_x
        else:
            v_t2s, v_s2t = v_x, v_x_cd

        if detach:
            q_t2s, q_s2t = q_t2s.detach(), q_s2t.detach()

        attn_t2s = (q_t2s @ k_t2s.transpose(-2, -1).contiguous()) * self.scale
        attn_t2s = attn_t2s.softmax(dim=-1)
        attn_t2s_drop = self.attn_drop(attn_t2s)
        t2s_out = (attn_t2s_drop @ v_t2s).transpose(1, 2).contiguous().reshape(B, N, C)
        t2s_out = self.proj(t2s_out)
        t2s_out = self.proj_drop(t2s_out)

        attn_s2t = (q_s2t @ k_s2t.transpose(-2, -1).contiguous()) * self.scale
        attn_s2t = attn_s2t.softmax(dim=-1)
        attn_s2t_drop = self.attn_drop(attn_s2t)
        s2t_out = (attn_s2t_drop @ v_s2t).transpose(1, 2).contiguous().reshape(B, N, C)
        s2t_out = self.proj(s2t_out)
        s2t_out = self.proj_drop(s2t_out)

        return x_out, x_cd_out, t2s_out, s2t_out, attn_x, attn_x_cd, attn_t2s, attn_s2t


class Block(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 sr_ratio=1,
                 CB=False,
                 attn_noise=0):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,
            attn_noise=attn_noise)
        # NOTE: drop path for stochastic depth, we shall see if this is better
        # than dropout here
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            CB=CB)

    def forward(self, x, H, W, require_attn=False):
        out, attn = self.attn(self.norm1(x), H, W)
        x = x + self.drop_path(out)
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return [x, attn] if require_attn else x

    def forward_four_branches(self, src, tgt, t2s, s2t, H, W, require_attn=False, detach=False, switch_element="q"):
        src_out, tgt_out, t2s_out, s2t_out, attn_src, attn_tgt, _, _ = self.attn(self.norm1(src), H, W, x_cd=self.norm1(tgt), t2s=self.norm1(t2s), s2t=self.norm1(s2t), detach=detach, switch_element=switch_element)
        src = src + self.drop_path(src_out)
        tgt = tgt + self.drop_path(tgt_out)
        t2s = t2s + self.drop_path(t2s_out)
        s2t = s2t + self.drop_path(s2t_out)

        src = src + self.drop_path(self.mlp(self.norm2(src), H, W))
        tgt = tgt + self.drop_path(self.mlp(self.norm2(tgt), H, W))
        t2s = t2s + self.drop_path(self.mlp(self.norm2(t2s), H, W))
        s2t = s2t + self.drop_path(self.mlp(self.norm2(s2t), H, W))

        return src, tgt, t2s, s2t, [attn_src, attn_tgt] if require_attn else None

class OverlapPatchEmbed(nn.Module):
    """Image to Patch Embedding."""

    def __init__(self,
                 img_size=224,
                 patch_size=7,
                 stride=4,
                 in_chans=3,
                 embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[
            1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.norm(x)

        return x, H, W


@BACKBONES.register_module()
class MixVisionTransformer(BaseModule):

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 num_classes=1000,
                 embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8],
                 mlp_ratios=[4, 4, 4, 4],
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3],
                 sr_ratios=[8, 4, 2, 1],
                 style=None,
                 pretrained=None,
                 init_cfg=None,
                 freeze_patch_embed=False,
                 detach=False,
                 switch_element="q",
                 CB=False,
                 attn_noise=0,
                 cda_prob=0,
                 hflip=0):
        super().__init__(init_cfg)

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be setting at the same time'
        if isinstance(pretrained, str) or pretrained is None:
            warnings.warn('DeprecationWarning: pretrained is a deprecated, '
                          'please use "init_cfg" instead')
        else:
            raise TypeError('pretrained must be a str or None')

        self.num_classes = num_classes
        self.depths = depths
        self.pretrained = pretrained
        self.init_cfg = init_cfg

        # cda module
        self.detach = detach
        self.switch_element = switch_element
        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(
            img_size=img_size,
            patch_size=7,
            stride=4,
            in_chans=in_chans,
            embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(
            img_size=img_size // 4,
            patch_size=3,
            stride=2,
            in_chans=embed_dims[0],
            embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(
            img_size=img_size // 8,
            patch_size=3,
            stride=2,
            in_chans=embed_dims[1],
            embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(
            img_size=img_size // 16,
            patch_size=3,
            stride=2,
            in_chans=embed_dims[2],
            embed_dim=embed_dims[3])
        if freeze_patch_embed:
            self.freeze_patch_emb()

        # transformer encoder
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList([
            Block(
                dim=embed_dims[0],
                num_heads=num_heads[0],
                mlp_ratio=mlp_ratios[0],
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[cur + i],
                norm_layer=norm_layer,
                sr_ratio=sr_ratios[0],
                CB=CB,
                attn_noise=attn_noise) for i in range(depths[0])
        ])
        self.norm1 = norm_layer(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList([
            Block(
                dim=embed_dims[1],
                num_heads=num_heads[1],
                mlp_ratio=mlp_ratios[1],
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[cur + i],
                norm_layer=norm_layer,
                sr_ratio=sr_ratios[1],
                CB=CB,
                attn_noise=attn_noise) for i in range(depths[1])
        ])
        self.norm2 = norm_layer(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList([
            Block(
                dim=embed_dims[2],
                num_heads=num_heads[2],
                mlp_ratio=mlp_ratios[2],
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[cur + i],
                norm_layer=norm_layer,
                sr_ratio=sr_ratios[2],
                CB=CB,
                attn_noise=attn_noise) for i in range(depths[2])
        ])
        self.norm3 = norm_layer(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList([
            Block(
                dim=embed_dims[3],
                num_heads=num_heads[3],
                mlp_ratio=mlp_ratios[3],
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[cur + i],
                norm_layer=norm_layer,
                sr_ratio=sr_ratios[3],
                CB=CB,
                attn_noise=attn_noise) for i in range(depths[3])
        ])
        self.norm4 = norm_layer(embed_dims[3])

        # classification head
        # self.head = nn.Linear(embed_dims[3], num_classes) \
        #     if num_classes > 0 else nn.Identity()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def init_weights(self):
        logger = get_root_logger()
        if self.pretrained is None:
            logger.info('Init mit from scratch.')
            for m in self.modules():
                self._init_weights(m)
        elif isinstance(self.pretrained, str):
            logger.info('Load mit checkpoint.')
            checkpoint = _load_checkpoint(
                self.pretrained, logger=logger, map_location='cpu')
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            self.load_state_dict(state_dict, False)

    def reset_drop_path(self, drop_path_rate):
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(self.depths))
        ]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'
        }  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(
            self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, require_attn=False):
        B = x.shape[0]
        outs = []
        if require_attn:
            attns = []
        # stage 1
        x, H, W = self.patch_embed1(x)
        for i, blk in enumerate(self.block1):
            x = blk(x, H, W, require_attn=require_attn)
            if require_attn:
                x, attn = x
                attns.append(attn)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 2
        x, H, W = self.patch_embed2(x)
        for i, blk in enumerate(self.block2):
            x = blk(x, H, W, require_attn=require_attn)
            if require_attn:
                x, attn = x
                attns.append(attn)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 3
        x, H, W = self.patch_embed3(x)
        for i, blk in enumerate(self.block3):
            x = blk(x, H, W, require_attn=require_attn)
            if require_attn:
                x, attn = x
                attns.append(attn)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 4
        x, H, W = self.patch_embed4(x)
        for i, blk in enumerate(self.block4):
            x = blk(x, H, W, require_attn=require_attn)
            if require_attn:
                x, attn = x
                attns.append(attn)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        return [outs, attns] if require_attn else outs


    def forward_features_four_branches(self, src, tgt, require_attn=False):
        assert src.shape == tgt.shape
        B = src.shape[0]
        src_outs = []
        tgt_outs = []
        t2s_outs = []
        s2t_outs = []
        src_attns = []
        mix_attns = []

        # stage 1
        t2s, H, W = self.patch_embed1(src)
        s2t, _, _ = self.patch_embed1(tgt)
        src, _, _ = self.patch_embed1(src)
        tgt, _, _ = self.patch_embed1(tgt)

        for i, blk in enumerate(self.block1):
            src, tgt, t2s, s2t, attns = blk.forward_four_branches(src, tgt, t2s, s2t, H, W, require_attn, self.detach, self.switch_element)
            if require_attn:
                src_attn, mix_attn = attns
                src_attns.append(src_attn)
                mix_attns.append(mix_attn)
        src, tgt, t2s, s2t = self.norm1(src), self.norm1(tgt), self.norm1(t2s), self.norm1(s2t)
        src = src.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        tgt = tgt.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t2s = t2s.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        s2t = s2t.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        src_outs.append(src)
        tgt_outs.append(tgt)
        t2s_outs.append(t2s)
        s2t_outs.append(s2t)

        # stage 2
        src, H, W = self.patch_embed2(src)
        tgt, _, _ = self.patch_embed2(tgt)
        t2s, _, _ = self.patch_embed2(t2s)
        s2t, _, _ = self.patch_embed2(s2t)

        for i, blk in enumerate(self.block2):
            src, tgt, t2s, s2t, attns = blk.forward_four_branches(src, tgt, t2s, s2t, H, W, require_attn, self.detach, self.switch_element)
            if require_attn:
                src_attn, mix_attn = attns
                src_attns.append(src_attn)
                mix_attns.append(mix_attn)
        src, tgt, t2s, s2t = self.norm2(src), self.norm2(tgt), self.norm2(t2s), self.norm2(s2t)
        src = src.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        tgt = tgt.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t2s = t2s.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        s2t = s2t.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        src_outs.append(src)
        tgt_outs.append(tgt)
        t2s_outs.append(t2s)
        s2t_outs.append(s2t)

        # stage 3
        src, H, W = self.patch_embed3(src)
        tgt, _, _ = self.patch_embed3(tgt)
        t2s, _, _ = self.patch_embed3(t2s)
        s2t, _, _ = self.patch_embed3(s2t)
        for i, blk in enumerate(self.block3):
            src, tgt, t2s, s2t, attns = blk.forward_four_branches(src, tgt, t2s, s2t, H, W, require_attn, self.detach, self.switch_element)
            if require_attn:
                src_attn, mix_attn = attns
                src_attns.append(src_attn)
                mix_attns.append(mix_attn)
        src, tgt, t2s, s2t = self.norm3(src), self.norm3(tgt), self.norm3(t2s), self.norm3(s2t)
        src = src.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        tgt = tgt.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t2s = t2s.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        s2t = s2t.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        src_outs.append(src)
        tgt_outs.append(tgt)
        t2s_outs.append(t2s)
        s2t_outs.append(s2t)

        # stage 4
        src, H, W = self.patch_embed4(src)
        tgt, _, _ = self.patch_embed4(tgt)
        t2s, _, _ = self.patch_embed4(t2s)
        s2t, _, _ = self.patch_embed4(s2t)
        for i, blk in enumerate(self.block4):
            src, tgt, t2s, s2t, attns = blk.forward_four_branches(src, tgt, t2s, s2t, H, W, require_attn, self.detach, self.switch_element)
            if require_attn:
                src_attn, mix_attn = attns
                src_attns.append(src_attn)
                mix_attns.append(mix_attn)
        src, tgt, t2s, s2t = self.norm4(src), self.norm4(tgt), self.norm4(t2s), self.norm4(s2t)
        src = src.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        tgt = tgt.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t2s = t2s.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        s2t = s2t.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        src_outs.append(src)
        tgt_outs.append(tgt)
        t2s_outs.append(t2s)
        s2t_outs.append(s2t)

        return src_outs, tgt_outs, t2s_outs, s2t_outs, [], src_attns, mix_attns

    def forward(self, x, require_attn=False):
        x = self.forward_features(x, require_attn)
        # x = self.head(x)

        return x


class DWConv(nn.Module):

    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2).contiguous()

        return x


@BACKBONES.register_module()
class mit_b0(MixVisionTransformer):

    def __init__(self, **kwargs):
        super(mit_b0, self).__init__(
            patch_size=4,
            embed_dims=[32, 64, 160, 256],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[2, 2, 2, 2],
            sr_ratios=[8, 4, 2, 1],
            **kwargs)


@BACKBONES.register_module()
class mit_b1(MixVisionTransformer):

    def __init__(self, **kwargs):
        super(mit_b1, self).__init__(
            patch_size=4,
            embed_dims=[64, 128, 320, 512],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[2, 2, 2, 2],
            sr_ratios=[8, 4, 2, 1],
            **kwargs)


@BACKBONES.register_module()
class mit_b2(MixVisionTransformer):

    def __init__(self, **kwargs):
        super(mit_b2, self).__init__(
            patch_size=4,
            embed_dims=[64, 128, 320, 512],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[3, 4, 6, 3],
            sr_ratios=[8, 4, 2, 1],
            **kwargs)


@BACKBONES.register_module()
class mit_b3(MixVisionTransformer):

    def __init__(self, **kwargs):
        super(mit_b3, self).__init__(
            patch_size=4,
            embed_dims=[64, 128, 320, 512],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[3, 4, 18, 3],
            sr_ratios=[8, 4, 2, 1],
            **kwargs)


@BACKBONES.register_module()
class mit_b4(MixVisionTransformer):

    def __init__(self, **kwargs):
        super(mit_b4, self).__init__(
            patch_size=4,
            embed_dims=[64, 128, 320, 512],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[3, 8, 27, 3],
            sr_ratios=[8, 4, 2, 1],
            **kwargs)


@BACKBONES.register_module()
class mit_b5(MixVisionTransformer):

    def __init__(self, **kwargs):
        super(mit_b5, self).__init__(
            patch_size=4,
            embed_dims=[64, 128, 320, 512],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[3, 6, 40, 3],
            sr_ratios=[8, 4, 2, 1],
            **kwargs)
