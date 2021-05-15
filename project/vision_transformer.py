from typing import TypeVar, Union, Tuple, Optional
from enum import Enum

from torch.utils.data import DataLoader, Dataset, random_split

from einops import rearrange, repeat

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


MIN_NUM_PATCHES = 16


class Embedding_mode(Enum):

    linear='linear'
    resnet='resnet'
    conv='conv'

    def __str__(self):
        return self.value


class AddPositionEmbs(nn.Module):
    """Adds (optionally learned) positional embeddings to the inputs."""

    def __init__(self, num_patches: int, dim: int, dropout_rate: float = .0):
        super(AddPositionEmbs, self).__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.dropout = nn.Dropout(dropout_rate, inplace=True) if dropout_rate > 0 else None

    def forward(self, x):
        x = x + self.pos_embedding
        return self.dropout(x) if self.dropout else x


class MlpBlock(nn.Module):
    """Transformer MLP / feed-forward block."""

    def __init__(self, in_dim: int, mlp_dim: int, out_dim: int, dropout_rate: float = 0.1):
        super(MlpBlock, self).__init__()
        self.fc1 = self.__init_weights(nn.Linear(in_dim, mlp_dim))
        self.fc2 = self.__init_weights(nn.Linear(mlp_dim, in_dim))
        self.act = nn.GELU()
        self.dropout1, self.dropout2 = (nn.Dropout(dropout_rate, inplace=True), nn.Dropout(
            dropout_rate, inplace=True)) if dropout_rate > 0 else (None, None)

    def forward(self, x):
        """Applies Transformer MlpBlock module."""
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout1(x) if self.dropout1 else x
        x = self.fc2(x)
        x = self.dropout2(x) if self.dropout1 else x
        return x

    def __init_weights(self, layer):
        nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.normal_(layer.bias, std=1e-6)
        return layer


class MultiHeadSelfAttention(nn.Module):
    """Some Information about MultiHeadSelfAttention"""

    def __init__(self, dim: int, num_heads: int, dropout_rate: float = 0., d_h:int=None):
        super(MultiHeadSelfAttention, self).__init__()
        if not d_h:
            d_h = dim // num_heads

        self.num_heads = num_heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(in_features=dim, out_features=self.num_heads * d_h * 3)

        self.out = nn.Linear(in_features=self.num_heads * d_h, out_features=dim)

        self.dropout = nn.Dropout(dropout_rate, inplace=True) if dropout_rate > 0 else None

    def forward(self, x, mask=None, need_weights=False):
        x = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), x)
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        mask_value = -torch.finfo(dots.dtype).max
        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        out = self.out(out)
        out = self.dropout(out) if self.dropout else out
        if need_weights:
            return out, attn
        return out


class Encoder1DBlock(nn.Module):
    """Transformer encoder layer."""

    def __init__(self, in_dim: int, mlp_dim: int, num_heads: int, dropout_rate: float = 0.1, attn_dropout_rate: float = 0.1):
        super(Encoder1DBlock, self).__init__()
        self.norm1 = nn.LayerNorm(in_dim)
        self.attn = MultiHeadSelfAttention(dim=in_dim, num_heads=num_heads, dropout_rate=attn_dropout_rate)
        self.dropout = nn.Dropout(dropout_rate, inplace=True) if dropout_rate > 0 else None
        self.norm2 = nn.LayerNorm(in_dim)
        self.mlp = MlpBlock(in_dim=in_dim, mlp_dim=mlp_dim, out_dim=in_dim, dropout_rate=dropout_rate)

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.attn(x)
        x = self.dropout(x) if self.dropout else x
        x += residual

        residual = x

        x = self.norm2(x)
        x = self.mlp(x)
        x += residual
        return x


class Encoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""

    def __init__(self, num_patches: int, emb_dim: int, mlp_dim: int, num_layers: int = 12, num_heads: int = 12, dropout_rate: float = 0.1, attn_dropout_rate: float = 0.0):
        super(Encoder, self).__init__()
        self.pos_embedding = AddPositionEmbs(num_patches, emb_dim, dropout_rate)
        in_dim = emb_dim
        self.encoder_layers = nn.Sequential()

        for i in range(num_layers):
            layer = Encoder1DBlock(
                in_dim=in_dim,
                mlp_dim=mlp_dim,
                num_heads=num_heads,
                dropout_rate=dropout_rate,
                attn_dropout_rate=attn_dropout_rate
            )
            self.encoder_layers.add_module(name=f"Encoder1DBlock_{i}", module=layer)

        self.norm = nn.LayerNorm(in_dim)

    def forward(self, x):
        x = self.pos_embedding(x)
        x = self.encoder_layers(x)
        x = self.norm(x)
        return x


class VisionTransformer(nn.Module):
    """VisionTransformer."""

    def __init__(self,
                image_size: Union[int, Tuple[int, int]] = (224, 224),
                channels: int = 3,
                patch_size: Union[int, Tuple[int, int]] = (16, 16),
                emb_dim: int = 1024,
                mlp_dim: int = 2048,
                num_heads: int = 16,
                num_layers: int = 24,
                num_classes: int = 10,
                attn_dropout_rate: float = 0.0,
                dropout_rate: float = 0.1,
                embedding_mode: Embedding_mode = Embedding_mode.linear,
                **kwargs,
    ):
        """Vision Transformer https://arxiv.org/abs/2010.11929

        Args:
            image_size (Tuple[int, int], optional): Image Size. Defaults to (224, 224).
            patch_size (Tuple[int, int], optional): Patch size. Defaults to (16, 16).
            emb_dim (int, optional): Embedding size of each patch. Defaults to 1024.
            mlp_dim (int, optional): Transformer MLP / feed-forward block dimension. Defaults to 2048.
            num_heads (int, optional): Number of heads on MultiHeadSelfAttention. Defaults to 16.
            num_layers (int, optional): Number of Transformer encoder layer in our model. Defaults to 24.
            num_classes (int, optional): Number of class in out. Defaults to 10.
            attn_dropout_rate (float, optional): Attention dropout. Defaults to 0.0.
            dropout_rate (float, optional): dropout rate of all layers and Embedding. Defaults to 0.1.
            embedding_mode (str, optional): `linear`, `resnet`, `conv`. Default: `linear`
        """

        super(VisionTransformer, self).__init__()
        if type(image_size) is tuple:
            h, w = image_size
        else:
            h, w = image_size, image_size

        if type(patch_size) is tuple:
            self.fh, self.fw = patch_size
        else:
            self.fh, self.fw = patch_size, patch_size


        assert h % self.fh == 0, 'Image H must be divisible by the patch H size.'
        assert w % self.fw == 0, 'Image W must be divisible by the patch W size.'
        gh, gw = h // self.fh, w // self.fw
        num_patches = gh * gw
        assert num_patches > MIN_NUM_PATCHES, f'your number of patches ({num_patches}) is way too small for attention to be effective (at least 16). Try decreasing your patch size'

        self.embedding_mode = embedding_mode
        self.embedding = nn.Linear(channels * self.fh * self.fw, emb_dim)

        if self.embedding_mode == Embedding_mode.resnet:
            resnet18 = torchvision.models.resnet18(pretrained=True)
            self.embedding = nn.Sequential(
                resnet18.conv1,
                resnet18.bn1,
                resnet18.relu,
                resnet18.maxpool,
                resnet18.layer1,
                resnet18.layer2,
                resnet18.layer3,
                resnet18.layer4,
            )
            for param in self.embedding.parameters():
                param.requires_grad = False

            num_patches = 7 * 7
            emb_dim = 512

        if self.embedding_mode == Embedding_mode.conv:
            self.embedding = nn.Conv2d( 3, emb_dim, kernel_size=(self.fh, self.fw), stride=(self.fh, self.fw))

        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.transformer = Encoder(
            num_patches=num_patches,
            emb_dim=emb_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            attn_dropout_rate=attn_dropout_rate
        )

        self.classifier = nn.Linear(emb_dim, num_classes)

    def forward(self, x):
        if self.embedding_mode == Embedding_mode.conv or self.embedding_mode == Embedding_mode.resnet:
            x = self.embedding(x) # x (batch_size, in_channel, img_heigh, img_width)
            x = rearrange( x, 'b c gh gw -> b (gh gw) c') # x (batch_size, num_patch, emb_dim)

        if self.embedding_mode == Embedding_mode.linear:
            x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.fh, p2=self.fw)
            x = self.embedding(x)

        # cls_token ==> shape (batch_size, 1, emb_dim)
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=x.shape[0])

        x = torch.cat([cls_tokens, x], dim=1)  # x (batch_size, num_patch+1, emb_dim)

        x = self.transformer(x)

        x = self.classifier(x[:, 0])  # x (batch_size, num_classes)

        return x
