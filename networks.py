import copy

import torch
import torch.nn as nn
from torch import optim
from torch.nn import functional as F
import pytorch_lightning as pl

from models.vit import VisionTransformer, PatchEmbed, Block,resolve_pretrained_cfg, build_model_with_cfg, checkpoint_filter_fn
from models.convit import ClassAttention
from models.convit import Block as ConBlock


class ViT_KPrompts(VisionTransformer):
    def __init__(
            self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, global_pool='token',
            embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None,
            drop_rate=0., attn_drop_rate=0., drop_path_rate=0., weight_init='', init_values=None,
            embed_layer=PatchEmbed, norm_layer=None, act_layer=None, block_fn=Block):

        super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, num_classes=num_classes, global_pool=global_pool,
            embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, representation_size=representation_size,
            drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, weight_init=weight_init, init_values=init_values,
            embed_layer=embed_layer, norm_layer=norm_layer, act_layer=act_layer, block_fn=block_fn)

    def forward(self, x, instance_tokens=None, returnbeforepool=False, **kwargs):
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)

        if instance_tokens is not None:
            instance_tokens = instance_tokens.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)

        x = x + self.pos_embed.to(x.dtype)
        if instance_tokens is not None:
            x = torch.cat([x[:,:1,:], instance_tokens, x[:,1:,:]], dim=1)
        x = self.pos_drop(x)
        x = self.blocks(x)
        if returnbeforepool == True:
            return x
        x = self.norm(x)
        if self.global_pool:
            x = x[:, 1:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.fc_norm(x)
        return x

def _create_vision_transformer(variant, pretrained=False, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    # NOTE this extra code to support handling of repr size for in21k pretrained models
    pretrained_cfg = resolve_pretrained_cfg(variant, pretrained_cfg=kwargs.pop('pretrained_cfg', None))
    default_num_classes = pretrained_cfg['num_classes']
    num_classes = kwargs.get('num_classes', default_num_classes)
    repr_size = kwargs.pop('representation_size', None)
    if repr_size is not None and num_classes != default_num_classes:
        repr_size = None

    model = build_model_with_cfg(
        ViT_KPrompts, variant, pretrained,
        pretrained_cfg=pretrained_cfg,
        representation_size=repr_size,
        pretrained_filter_fn=checkpoint_filter_fn,
        pretrained_custom_load='npz' in pretrained_cfg['url'],
        **kwargs)
    return model



class incremental_vitood(pl.LightningModule):
    def __init__(self, num_cls, lr, max_epoch, weight_decay, known_classes, freezep, using_prompt, anchor_energy=-10,
                 lamda=0.1, energy_beta=1):
        super().__init__()
        self.save_hyperparameters()

        model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12)
        self.image_encoder =_create_vision_transformer('vit_base_patch16_224', pretrained=True, **model_kwargs)

        self.classifiers = nn.Linear(self.image_encoder.embed_dim, self.hparams.num_cls, bias=True)
        self.tabs = ConBlock(dim=self.image_encoder.embed_dim, num_heads=12, mlp_ratio=0.5, qkv_bias=True,
                        qk_scale=None, drop=0.,attn_drop=0., norm_layer=nn.LayerNorm, attention_type=ClassAttention)
        self.task_tokens = copy.deepcopy(self.image_encoder.cls_token)
        self.vitprompt = nn.Linear(self.image_encoder.embed_dim, 100, bias=False)
        self.pre_vitprompt = None

        for name, param in self.image_encoder.named_parameters():
            param.requires_grad_(False)

        if self.hparams.freezep:
            for name, param in self.vitprompt.named_parameters():
                param.requires_grad_(False)

    def forward(self, image):
        if self.hparams.using_prompt:
            image_features = self.image_encoder(image, instance_tokens=self.vitprompt, returnbeforepool=True, )
        else:
            image_features = self.image_encoder(image, returnbeforepool=True)

        B = image_features.shape[0]
        task_token = self.task_tokens.expand(B, -1, -1)
        task_token, attn, v = self.tabs(torch.cat((task_token, image_features), dim=1), mask_heads=None)
        logits = self.classifiers(task_token[:, 0])

        return logits

    def configure_optimizers(self):
        optparams = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = optim.SGD(optparams, momentum=0.9,lr=self.hparams.lr,weight_decay=self.hparams.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=self.hparams.max_epoch)
        return [optimizer], [scheduler]

    def _calculate_loss(self, batch, mode='train'):
        _, images, labels = batch
        labels = labels-self.hparams.known_classes

        if self.hparams.using_prompt:
            image_features = self.image_encoder(images, instance_tokens=self.vitprompt.weight, returnbeforepool=True, )
        else:
            image_features = self.image_encoder(images, returnbeforepool=True)
        B = image_features.shape[0]
        task_token = self.task_tokens.expand(B, -1, -1)
        task_token, attn, v = self.tabs(torch.cat((task_token, image_features), dim=1), mask_heads=None)
        logits = self.classifiers(task_token[:, 0])
        loss = F.cross_entropy(logits, labels)

        output_div_t = -1.0 * self.hparams.energy_beta * logits
        output_logsumexp = torch.logsumexp(output_div_t, dim=1, keepdim=False)
        free_energy = -1.0 * output_logsumexp / self.hparams.energy_beta
        align_loss = self.hparams.lamda * ((free_energy - self.hparams.anchor_energy) ** 2).mean()

        if self.pre_vitprompt is not None:
            pre_feature = self.image_encoder(images, instance_tokens=self.pre_vitprompt.weight, returnbeforepool=True, )
            kdloss = nn.MSELoss()(pre_feature.detach(), image_features)
        else:
            kdloss = 0

        loss = loss+align_loss+kdloss

        acc = (logits.argmax(dim=-1) == labels).float().mean()
        self.log("%s_loss" % mode, loss)
        self.log("%s_acc" % mode, acc)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode='val')

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode='train')
        return loss

    def val_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode='val')

    def test_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode='test')

