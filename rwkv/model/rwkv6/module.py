########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import math, gc
from typing import Dict, Optional
import torch

import torch.nn as nn
from torch.nn import Module, functional as F
from rwkv.model.rwkv6.time_mixing import TimeMixing, TimeMixingState
from rwkv.model.rwkv6.channel_mixing import ChannelMixing


class Block(nn.Module):
    def __init__(
        self,
        layer_id: int,
        embedding_num: int,
        tiny_ctx_len: int,
        layer_num: int,
        ffn_dim: int,
        head_size: int,
        head_size_divisor: int,
        attn_dim: int,
        pre_ffn: bool = False,
        tiny_attn_dim: int = 0,
        tiny_attn_layer: int = 0,
        pos_embedding_dim: int = 0,
        attn_layer: int = 0,
        trainable_state: bool = False,
        dropout_p: float = 0,
    ):
        super().__init__()
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(embedding_num)
        self.ln2 = nn.LayerNorm(embedding_num)
        self.pos_embedding_dim = pos_embedding_dim

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(embedding_num)
            if pos_embedding_dim > 0:
                self.pos_emb_x = nn.Parameter(torch.zeros((1, pos_embedding_dim, embedding_num)))
                self.pos_emb_y = nn.Parameter(torch.zeros((pos_embedding_dim, 1, embedding_num)))

        if self.layer_id == 0 and pre_ffn > 0:
            self.pre_ffn = ChannelMixing(
                layer_id=layer_id, layer_num=layer_num, embedding_num=embedding_num, ffn_dim=ffn_dim
            )
        else:
            if trainable_state:
                self.att = TimeMixingState(
                    layer_id=layer_id,
                    head_size=head_size,
                    head_size_divisor=head_size_divisor,
                    attn_dim=attn_dim,
                    layer_num=layer_num,
                    embedding_num=embedding_num,
                    tiny_ctx_len=tiny_ctx_len,
                )
            else:
                self.att = TimeMixing(
                    layer_id=layer_id,
                    head_size=head_size,
                    head_size_divisor=head_size_divisor,
                    attn_dim=attn_dim,
                    layer_num=layer_num,
                    embedding_num=embedding_num,
                    tiny_ctx_len=tiny_ctx_len,
                )
        self.ffn = ChannelMixing(layer_id=layer_id, layer_num=layer_num, embedding_num=embedding_num, ffn_dim=ffn_dim)
        if tiny_attn_dim > 0 and self.layer_id == attn_layer:
            self.tiny_ln = nn.LayerNorm(embedding_num)
            self.tiny_q = nn.Linear(embedding_num, tiny_attn_dim, bias=False)
            self.tiny_k = nn.Linear(embedding_num, tiny_attn_dim, bias=False)
            self.tiny_v = nn.Linear(embedding_num, embedding_num, bias=False)
            self.register_buffer("tiny_mask", torch.tril(torch.ones(tiny_ctx_len, tiny_ctx_len)))
        if dropout_p > 0:
            self.drop0 = nn.Dropout(p=dropout_p)
            self.drop1 = nn.Dropout(p=dropout_p)
        self.layer_id = layer_id
        self.dropout_p = dropout_p
        self.tiny_attn_dim = tiny_attn_dim
        self.tiny_attn_layer = tiny_attn_layer

    def forward(self, x: torch.Tensor, x_emb_for_tinyattn: Optional[torch.Tensor] = None):
        B, T, C = x.size()
        if self.layer_id == 0:
            x = self.ln0(x)
            if self.pos_embedding_dim > 0:
                pos_emb = (self.pos_emb_x + self.pos_emb_y).reshape(T + 1, -1)[:-1, :]
                x = x + pos_emb

        if self.dropoutp == 0:
            if self.layer_id == 0 and "pre_ffn" in dir(self):
                x = x + self.pre_ffn(self.ln1(x))
            else:
                x = x + self.att(self.ln1(x))
            x = x + self.ffn(self.ln2(x))
        else:
            if self.layer_id == 0 and "pre_ffn" in dir(self):
                x = self.drop0(x + self.pre_ffn(self.ln1(x)))
            else:
                x = self.drop0(x + self.att(self.ln1(x)))
            x = self.drop1(x + self.ffn(self.ln2(x)))

        if self.tiny_attn_dim > 0 and self.layer_id == self.tiny_att_layer:
            xx = self.tiny_ln(x)
            q = self.tiny_q(xx)[:, :T, :]
            k = self.tiny_k(xx)[:, :T, :]
            c = (q @ k.transpose(-2, -1)) * (self.tiny_attn_dim ** (-0.5))
            c = c.masked_fill(self.tiny_mask[:T, :T] == 0, 0)
            x = x + c @ self.tiny_v(x_emb_for_tinyattn)
        return x


class RWKV(Module):
    def __init__(
        self,
        attn_dim: int = 2048,
        embedding_num: int = 2048,
        vocab_size: int = 65536,
        head_size: int = 64,
        head_size_divisor: int = 8,
        layer_num: int = 24,
        head_qk_dim: int = 0,
        pre_ffn: bool = False,
        ffn_dim: int = 0,
        pos_embedding_dim: int = 0,
        trainable_state: bool = False,
        dropout_p: float = 0,
        tiny_attn_dim: int = 0,
        tiny_attn_layer: int = 0,
        tiny_ctx_len: int = 0,
    ):
        super().__init__()
        if ffn_dim == 0:
            ffn_dim = int((embedding_num * 3.5) // 32 * 32)
        if tiny_attn_dim == 0:
            tiny_attn_dim = -1
        if tiny_attn_layer == 0:
            tiny_attn_layer = -1
        assert embedding_num % 32 == 0
        assert attn_dim % 32 == 0
        assert ffn_dim % 32 == 0

        self.emb = nn.Embedding(vocab_size, embedding_num)

        self.blocks = nn.ModuleList(
            [
                Block(
                    layer_id=i,
                    embedding_num=embedding_num,
                    tiny_ctx_len=tiny_ctx_len,
                    layer_num=layer_num,
                    ffn_dim=ffn_dim,
                    head_size=head_size,
                    head_size_divisor=head_size_divisor,
                    attn_dim=attn_dim,
                    pre_ffn=pre_ffn,
                    tiny_attn_dim=tiny_attn_dim,
                    tiny_attn_layer=tiny_attn_dim,
                    pos_embedding_dim=pos_embedding_dim,
                    attn_layer=attn_dim,
                    trainable_state=trainable_state,
                    dropout_p=dropout_p,
                )
                for i in range(layer_num)
            ]
        )

        self.ln_out = nn.LayerNorm(embedding_num)
        self.head = nn.Linear(embedding_num, vocab_size, bias=False)

        if head_qk_dim > 0:
            self.head_q = nn.Linear(embedding_num, head_qk_dim, bias=False)
            self.head_k = nn.Linear(embedding_num, head_qk_dim, bias=False)
            self.register_buffer("copy_mask", torch.tril(torch.ones(tiny_ctx_len, tiny_ctx_len)))
        if dropout_p > 0:
            self.drop0 = nn.Dropout(p=dropout_p)
        self.tiny_ctx_len = tiny_ctx_len
        self.dropout_p = dropout_p
        self.head_qk_dim = head_qk_dim
        self.vocab_size = vocab_size
        self.layer_num = layer_num
        self.embedding_num = embedding_num

    def forward(self, idx: torch.Tensor):
        B, T = idx.size()
        assert T <= self.tiny_ctx_len, "Cannot forward, model tiny_ctx_len is exhausted."

        x = self.emb(idx)
        x_emb = x

        if self.dropout_p > 0:
            x = self.drop0(x)

        for block in self.blocks:
            x = block(x, x_emb)

        x = self.ln_out(x)

        if self.head_qk_dim > 0:
            q = self.head_q(x)[:, :T, :]
            k = self.head_k(x)[:, :T, :]
            c = (q @ k.transpose(-2, -1)) * (1.0 / self.head_qk_dim)
            c = c.masked_fill(self.copy_mask[:T, :T] == 0, 0)
            c = c @ F.one_hot(idx, num_classes=self.vocab_size)

            x = self.head(x) + c
        else:
            x = self.head(x)

        return x

    def init_weight(self, dtype: Optional[torch.dtype]):
        params: Dict[str, torch.Tensor] = {}
        for layer_name, layer_param in self.state_dict().items():
            shape = layer_param.shape

            scale = 1.0
            if (
                "ln_" in layer_name
                or ".ln" in layer_name
                or "time_" in layer_name
                or "_mask" in layer_name
                or "pos_emb" in layer_name
                or ".mask." in layer_name
                or layer_name.endswith("_w")
                or layer_name.endswith("_w1")
                or layer_name.endswith("_w2")
                or layer_name.endswith("_bias")
            ):
                if "ln_x.weight" in layer_name:
                    layer_scale = (1 + int(layer_name.split(".")[1])) / self.layer_num
                    params[layer_name] = (layer_param * 0.0) + (layer_scale**0.7)
                else:
                    params[layer_name] = layer_param
            elif layer_name == "emb.weight":
                params[layer_name] = layer_param
                scale = -1e-4
                nn.init.uniform_(params[layer_name], a=scale, b=-scale)
            elif layer_name == "head.weight":
                params[layer_name] = layer_param
                if self.vocab_size > self.embedding_num:
                    scale = 0.5 * math.sqrt(self.vocab_size / self.embedding_num)
                else:
                    scale = 0.5
                nn.init.orthogonal_(params[layer_name], gain=scale)
            else:
                assert layer_name.endswith(".weight")

                zero = [
                    ".att.output.",
                    ".ffn.value.",
                    ".ffn.receptance.",
                    ".ffnPre.value.",
                    ".ffnPre.receptance.",
                    "head_q.",
                    ".oo.",
                    ".rr.",
                ]

                for kk in zero:
                    if kk in layer_name:
                        scale = 0
                if "head_k." in layer_name:
                    scale = 0.1
                if "head_q." in layer_name:
                    scale = 0

                for kk in [".att.key."]:
                    if kk in layer_name:
                        scale = 0.1
                for kk in [".att.gate."]:
                    if kk in layer_name:
                        scale = 0.1

                params[layer_name] = torch.empty((shape[0], shape[1]))

                if scale == 0:
                    nn.init.zeros_(params[layer_name])
                elif scale < 0:
                    nn.init.uniform_(params[layer_name], a=scale, b=-scale)
                else:
                    nn.init.orthogonal_(params[layer_name], gain=scale)

                if dtype is not None:
                    params[layer_name] = params[layer_name].type(dtype)

        gc.collect()
        torch.cuda.empty_cache()
        return params
