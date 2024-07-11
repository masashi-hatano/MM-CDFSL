import logging

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from models.modeling_finetune import Block, PatchEmbed, get_sinusoid_encoding_table
from models.modeling_pretrain import PretrainVisionTransformerEncoder, trunc_normal_

logger = logging.getLogger(__name__)


class ViTEncoder(nn.Module):
    """Vision Transformer with support for patch or hybrid CNN input stage"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=0,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_values=None,
        tubelet_size=2,
        use_checkpoint=False,
        use_learnable_pos_emb=False,
        use_mmtoken=False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = (
            embed_dim  # num_features for consistency with other models
        )
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            tubelet_size=tubelet_size,
        )
        num_patches = self.patch_embed.num_patches
        self.use_checkpoint = use_checkpoint
        self.use_mmtoken = use_mmtoken

        # TODO: Add the cls token
        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        else:
            # sine-cosine positional embeddings
            self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    init_values=init_values,
                )
                for i in range(depth)
            ]
        )
        if use_mmtoken:
            self.mmtoken = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.norm = norm_layer(embed_dim)
        self.head = (
            nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=0.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=""):
        self.num_classes = num_classes
        self.head = (
            nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

    def forward(self, x, mask=None):
        _, _, T, _, _ = x.shape
        x = self.patch_embed(x)

        x = x + self.pos_embed.type_as(x).to(x.device).clone().detach()

        if self.use_mmtoken:
            x = torch.cat([x, self.mmtoken], dim=1)

        B, _, C = x.shape

        if mask is not None:
            x = x[~mask].reshape(B, -1, C)  # ~mask means visible

        if self.use_checkpoint:
            for blk in self.blocks:
                x = checkpoint.checkpoint(blk, x)
        else:
            for blk in self.blocks:
                x = blk(x)

        x = self.norm(x)
        return x


class VideoMAEClassifier(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        encoder_in_chans=3,
        encoder_num_classes=0,
        encoder_embed_dim=384,
        encoder_depth=12,
        encoder_num_heads=6,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_values=0.0,
        use_learnable_pos_emb=False,
        use_checkpoint=False,
        tubelet_size=2,
        num_classes_action=204,  # default: ego4d train num verb classes
        num_classes_verb=33,  # default: ego4d train num verb classes
        num_classes_noun=66,  # default: ego4d train num noun classes
        in_chans=0,  # avoid the error from create_fn in timm
        fc_drop_rate=0.5,
        use_mean_pooling=False,
    ):
        super(VideoMAEClassifier, self).__init__()
        self.encoder = PretrainVisionTransformerEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=encoder_in_chans,
            num_classes=0,  # should be zero to avoid the error
            embed_dim=encoder_embed_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            tubelet_size=tubelet_size,
            use_checkpoint=use_checkpoint,
            use_learnable_pos_emb=use_learnable_pos_emb,
        )

        self.fc_dropout = (
            nn.Dropout(p=fc_drop_rate) if fc_drop_rate > 0 else nn.Identity()
        )
        self.fc_norm = norm_layer(encoder_embed_dim) if use_mean_pooling else None
        self.head_action = nn.Linear(encoder_embed_dim, num_classes_action)
        self.head_verb = nn.Linear(encoder_embed_dim, num_classes_verb)
        self.head_noun = nn.Linear(encoder_embed_dim, num_classes_noun)

        # # mask
        # self.num_patches_per_frame = 196
        # self.num_masks_per_frame = int(0.75*self.num_patches_per_frame)
        # mask_per_frame = np.hstack(
        #     [
        #         np.zeros(self.num_patches_per_frame - self.num_masks_per_frame),
        #         np.ones(self.num_masks_per_frame),
        #     ]
        # )
        # np.random.shuffle(mask_per_frame)
        # self.mask = np.tile(mask_per_frame, (8, 1)).flatten()
        # self.mask = np.expand_dims(self.mask, 0)
        # self.mask = torch.from_numpy(self.mask).to(torch.bool)

    def forward(self, x, mask=None):
        x = self.encoder(x, mask)
        # [B, N, C] -> [B, C]
        if self.fc_norm is not None:
            x_encoded = self.fc_norm(x.mean(1))
        # else:
        #     x_encoded = x[:, 0]
        x = self.head_action(self.fc_dropout(x_encoded))
        return x_encoded, x


def videomae_classifier_small_patch16_224(
    ckpt_pth=None,
    img_size=224,
    patch_size=16,
    in_chans=3,
    num_classes_action=204,
    use_mean_pooling=False,
    **kwargs,
):
    model = VideoMAEClassifier(
        img_size=img_size,
        patch_size=patch_size,
        encoder_in_chans=in_chans,
        encoder_embed_dim=384,
        encoder_depth=12,
        encoder_num_heads=6,
        encoder_num_classes=0,
        mlp_ratio=4,
        qkv_bias=True,
        num_classes_action=num_classes_action,
        use_mean_pooling=use_mean_pooling,
        **kwargs,
    )
    if ckpt_pth is not None:
        p = torch.load(ckpt_pth)
        od = p["module"]
        pretrained_dict = {}
        for k, v in od.items():
            if "model." in k and "encoder." in k:
                k = k.replace("_forward_module.model.", "")
                pretrained_dict[k] = v
            elif "student_rgb." in k and "encoder." in k:
                k = k.replace("_forward_module.student_rgb.", "")
                pretrained_dict[k] = v
        # print(pretrained_dict.keys())
        model_dict = model.state_dict()
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        logger.info(f"Model is initialized from {ckpt_pth}")
    else:
        raise Exception("2nd phase training is running from scratch!")
    return model
