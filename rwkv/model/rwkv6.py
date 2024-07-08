########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import math, gc
from typing import Dict, Optional
import torch

import torch.nn as nn
from torch.nn import Module, functional as F
from torch.utils.cpp_extension import load


def init_state_autograd(head_size: int, ctx_len: int):
    wkv6state_cuda = load(
        name="wkv6state",
        sources=["wkv6state_op.cpp", f"wkv6state_cuda.cu"],
        verbose=True,
        extra_cuda_cflags=[
            "-res-usage",
            "--use_fast_math",
            "-O3",
            "-Xptxas -O3",
            "--extra-device-vectorization",
            f"-D_N_={head_size}",
            f"-D_T_={ctx_len}",
        ],
    )

    class StateCuda(torch.autograd.Function):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.head_size = head_size

        def forward(self, ctx, B, T, C, H, r, k, v, w, u, s):
            with torch.no_grad():
                assert r.dtype == torch.bfloat16
                assert k.dtype == torch.bfloat16
                assert v.dtype == torch.bfloat16
                assert w.dtype == torch.bfloat16
                assert u.dtype == torch.bfloat16
                assert s.dtype == torch.bfloat16
                assert self.head_size == C // H
                ctx.B = B
                ctx.T = T
                ctx.C = C
                ctx.H = H
                assert r.is_contiguous()
                assert k.is_contiguous()
                assert v.is_contiguous()
                assert w.is_contiguous()
                assert u.is_contiguous()
                assert s.is_contiguous()
                ctx.save_for_backward(r, k, v, w, u, s)
                y = torch.empty((B, T, C), device=r.device, dtype=torch.bfloat16, memory_format=torch.contiguous_format)
                wkv6state_cuda.forward(B, T, C, H, r, k, v, w, u, s, y)  # type: ignore
                return y

        def backward(self, ctx, gy):
            with torch.no_grad():
                assert gy.dtype == torch.bfloat16
                B = ctx.B
                T = ctx.T
                C = ctx.C
                H = ctx.H
                assert gy.is_contiguous()
                r, k, v, w, u, s = ctx.saved_tensors
                gr = torch.empty(
                    (B, T, C),
                    device=gy.device,
                    requires_grad=False,
                    dtype=torch.bfloat16,
                    memory_format=torch.contiguous_format,
                )
                gk = torch.empty(
                    (B, T, C),
                    device=gy.device,
                    requires_grad=False,
                    dtype=torch.bfloat16,
                    memory_format=torch.contiguous_format,
                )
                gv = torch.empty(
                    (B, T, C),
                    device=gy.device,
                    requires_grad=False,
                    dtype=torch.bfloat16,
                    memory_format=torch.contiguous_format,
                )
                gw = torch.empty(
                    (B, T, C),
                    device=gy.device,
                    requires_grad=False,
                    dtype=torch.bfloat16,
                    memory_format=torch.contiguous_format,
                )
                gu = torch.empty(
                    (B, C),
                    device=gy.device,
                    requires_grad=False,
                    dtype=torch.bfloat16,
                    memory_format=torch.contiguous_format,
                )
                gs = torch.empty(
                    (B, H, C // H, C // H),
                    device=gy.device,
                    requires_grad=False,
                    dtype=torch.bfloat16,
                    memory_format=torch.contiguous_format,
                )
                wkv6state_cuda.backward(B, T, C, H, r, k, v, w, u, s, gy, gr, gk, gv, gw, gu, gs)  # type: ignore
                gu = torch.sum(gu, 0).view(H, C // H)
                gs = torch.sum(gs, 0).view(H, C // H, C // H)

                return (None, None, None, None, gr, gk, gv, gw, gu, gs)

    return StateCuda


def init_rwkv_cuda(head_size: int, ctx_len: int):
    class RWKVCuda(torch.autograd.Function):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.wkv6_cuda = load(
                name="wkv6",
                sources=["wkv6_op.cpp", f"wkv6_cuda.cu"],
                verbose=True,
                extra_cuda_cflags=[
                    "-res-usage",
                    "--use_fast_math",
                    "-O3",
                    "-Xptxas -O3",
                    "--extra-device-vectorization",
                    f"-D_N_={head_size}",
                    f"-D_T_={ctx_len}",
                ],
            )

        def forward(self, ctx, B, T, C, H, r, k, v, w, u):
            with torch.no_grad():
                assert r.dtype == torch.bfloat16
                assert k.dtype == torch.bfloat16
                assert v.dtype == torch.bfloat16
                assert w.dtype == torch.bfloat16
                assert u.dtype == torch.bfloat16
                assert head_size == C // H
                ctx.B = B
                ctx.T = T
                ctx.C = C
                ctx.H = H
                assert r.is_contiguous()
                assert k.is_contiguous()
                assert v.is_contiguous()
                assert w.is_contiguous()
                assert u.is_contiguous()
                ctx.save_for_backward(r, k, v, w, u)
                y = torch.empty((B, T, C), device=r.device, dtype=torch.bfloat16, memory_format=torch.contiguous_format)
                self.wkv6_cuda.forward(B, T, C, H, r, k, v, w, u, y)
                return y

        def backward(self, ctx, gy):
            with torch.no_grad():
                assert gy.dtype == torch.bfloat16
                B = ctx.B
                T = ctx.T
                C = ctx.C
                H = ctx.H
                assert gy.is_contiguous()
                r, k, v, w, u = ctx.saved_tensors
                gr = torch.empty(
                    (B, T, C),
                    device=gy.device,
                    requires_grad=False,
                    dtype=torch.bfloat16,
                    memory_format=torch.contiguous_format,
                )
                gk = torch.empty(
                    (B, T, C),
                    device=gy.device,
                    requires_grad=False,
                    dtype=torch.bfloat16,
                    memory_format=torch.contiguous_format,
                )
                gv = torch.empty(
                    (B, T, C),
                    device=gy.device,
                    requires_grad=False,
                    dtype=torch.bfloat16,
                    memory_format=torch.contiguous_format,
                )
                gw = torch.empty(
                    (B, T, C),
                    device=gy.device,
                    requires_grad=False,
                    dtype=torch.bfloat16,
                    memory_format=torch.contiguous_format,
                )
                gu = torch.empty(
                    (B, C),
                    device=gy.device,
                    requires_grad=False,
                    dtype=torch.bfloat16,
                    memory_format=torch.contiguous_format,
                )
                self.wkv6_cuda.backward(B, T, C, H, r, k, v, w, u, gy, gr, gk, gv, gw, gu)
                gu = torch.sum(gu, 0).view(H, C // H)
                return (None, None, None, None, gr, gk, gv, gw, gu)

    return RWKVCuda


class TimeMixing(Module):
    def __init__(
        self, layer_id: int, head_size: int, head_size_divisor: int, attn_dim: int, layer_num: int, embedding_num: int
    ):
        super().__init__()
        self.layer_id = layer_id

        self.head_size = head_size
        self.n_head = attn_dim // self.head_size
        assert attn_dim % self.n_head == 0

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (layer_num - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / layer_num)  # 1 to ~0
            ddd = torch.ones(1, 1, embedding_num)
            for i in range(embedding_num):
                ddd[0, 0, i] = i / embedding_num

            # fancy time_mix
            self.time_maa_x = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_w = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_v = nn.Parameter(1.0 - (torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1))
            self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))
            self.time_maa_g = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))

            D_MIX_LORA = 32  # generate TIME_MIX for w,k,v,r,g
            self.time_maa_w1 = nn.Parameter(torch.zeros(embedding_num, D_MIX_LORA * 5))
            self.time_maa_w2 = nn.Parameter(torch.zeros(5, D_MIX_LORA, embedding_num).uniform_(-0.01, 0.01))

            # fancy time_decay
            decay_speed = torch.ones(attn_dim)
            for n in range(attn_dim):
                decay_speed[n] = -6 + 5 * (n / (attn_dim - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed.reshape(1, 1, attn_dim))

            D_DECAY_LORA = 64
            self.time_decay_w1 = nn.Parameter(torch.zeros(embedding_num, D_DECAY_LORA))
            self.time_decay_w2 = nn.Parameter(torch.zeros(D_DECAY_LORA, attn_dim).uniform_(-0.01, 0.01))

            tmp = torch.zeros()
            for n in range(attn_dim):
                zigzag = ((n + 1) % 3 - 1) * 0.1
                tmp[n] = ratio_0_to_1 * (1 - (n / (attn_dim - 1))) + zigzag

            self.time_faaaa = nn.Parameter(tmp.reshape(self.n_head, self.head_size))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.receptance = nn.Linear(embedding_num, attn_dim, bias=False)
        self.key = nn.Linear(embedding_num, attn_dim, bias=False)

        self.value = nn.Linear(embedding_num, attn_dim, bias=False)
        self.output = nn.Linear(attn_dim, embedding_num, bias=False)
        self.gate = nn.Linear(embedding_num, attn_dim, bias=False)
        self.ln_x = nn.GroupNorm(self.n_head, attn_dim, eps=(1e-5) * (head_size_divisor**2))
        self.rwkv_cuda = init_rwkv_cuda(self.head_size, self.ctx_len)

    def as_jit(self) -> "TimeMixing":
        self.forward_part1 = torch.jit.script(self.forward_part1)  # type: ignore
        self.forward_part2 = torch.jit.script(self.forward_part2)  # type: ignore
        return torch.jit.script(self)  # type: ignore

    def forward_part1(self, x):
        B, T, C = x.size()

        xx = self.time_shift(x) - x

        xxx = x + xx * self.time_maa_x
        xxx = torch.tanh(xxx @ self.time_maa_w1).view(B * T, 5, -1).transpose(0, 1)
        xxx = torch.bmm(xxx, self.time_maa_w2).view(5, B, T, -1)
        mw, mk, mv, mr, mg = xxx.unbind(dim=0)

        xw = x + xx * (self.time_maa_w + mw)
        xk = x + xx * (self.time_maa_k + mk)
        xv = x + xx * (self.time_maa_v + mv)
        xr = x + xx * (self.time_maa_r + mr)
        xg = x + xx * (self.time_maa_g + mg)

        r = self.receptance(xr)
        k = self.key(xk)
        v = self.value(xv)
        g = F.silu(self.gate(xg))

        ww = torch.tanh(xw @ self.time_decay_w1) @ self.time_decay_w2
        w = self.time_decay + ww

        return r, k, v, g, w

    def forward_part2(self, x, g):
        B, T, C = x.size()
        x = x.view(B * T, C)

        x = self.ln_x(x).view(B, T, C)
        x = self.output(x * g)
        return x

    def forward(self, x):
        B, T, C = x.size()
        H = self.n_head

        r, k, v, g, w = self.forward_part1(x)
        x = self.rwkv_cuda.apply(B, T, C, H, r, k, v, w, u=self.time_faaaa)

        return self.forward_part2(x, g)


class TimeMixingState(Module):
    def __init__(
        self, layer_id: int, head_size: int, head_size_divisor: int, attn_dim: int, layer_num: int, embedding_num: int
    ):
        super().__init__()
        self.layer_id = layer_id

        self.head_size = head_size
        self.n_head = attn_dim // self.head_size
        assert attn_dim % self.n_head == 0

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (layer_num - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / layer_num)  # 1 to ~0
            ddd = torch.ones(1, 1, embedding_num)
            for i in range(embedding_num):
                ddd[0, 0, i] = i / embedding_num

            # fancy time_mix
            self.time_maa_x = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_w = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_v = nn.Parameter(1.0 - (torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1))
            self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))
            self.time_maa_g = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))

            D_MIX_LORA = 32  # generate TIME_MIX for w,k,v,r,g
            self.time_maa_w1 = nn.Parameter(torch.zeros(embedding_num, D_MIX_LORA * 5))
            self.time_maa_w2 = nn.Parameter(torch.zeros(5, D_MIX_LORA, embedding_num).uniform_(-0.01, 0.01))

            # fancy time_decay
            decay_speed = torch.ones(attn_dim)
            for n in range(attn_dim):
                decay_speed[n] = -6 + 5 * (n / (attn_dim - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed.reshape(1, 1, attn_dim))

            D_DECAY_LORA = 64
            self.time_decay_w1 = nn.Parameter(torch.zeros(embedding_num, D_DECAY_LORA))
            self.time_decay_w2 = nn.Parameter(torch.zeros(D_DECAY_LORA, attn_dim).uniform_(-0.01, 0.01))

            tmp = torch.zeros(attn_dim)
            for n in range(attn_dim):
                zigzag = ((n + 1) % 3 - 1) * 0.1
                tmp[n] = ratio_0_to_1 * (1 - (n / (attn_dim - 1))) + zigzag

            self.time_faaaa = nn.Parameter(tmp.reshape(self.n_head, self.head_size))
            self.time_state = nn.Parameter(torch.zeros(self.n_head, self.head_size, self.head_size))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.receptance = nn.Linear(embedding_num, attn_dim, bias=False)
        self.key = nn.Linear(embedding_num, attn_dim, bias=False)

        self.value = nn.Linear(embedding_num, attn_dim, bias=False)
        self.output = nn.Linear(attn_dim, embedding_num, bias=False)
        self.gate = nn.Linear(embedding_num, attn_dim, bias=False)
        self.ln_x = nn.GroupNorm(self.n_head, attn_dim, eps=(1e-5) * (head_size_divisor**2))
        self.state_cuda = init_state_autograd(self.head_size, self.ctx_len)

    def forward_part1(self, x):
        """
        split forward function to 3 part: part1, cuda, part2
        """
        B, T, C = x.size()

        xx = self.time_shift(x) - x

        xxx = x + xx * self.time_maa_x
        xxx = torch.tanh(xxx @ self.time_maa_w1).view(B * T, 5, -1).transpose(0, 1)
        xxx = torch.bmm(xxx, self.time_maa_w2).view(5, B, T, -1)
        mw, mk, mv, mr, mg = xxx.unbind(dim=0)

        xw = x + xx * (self.time_maa_w + mw)
        xk = x + xx * (self.time_maa_k + mk)
        xv = x + xx * (self.time_maa_v + mv)
        xr = x + xx * (self.time_maa_r + mr)
        xg = x + xx * (self.time_maa_g + mg)

        r = self.receptance(xr)
        k = self.key(xk)
        v = self.value(xv)
        g = F.silu(self.gate(xg))

        ww = torch.tanh(xw @ self.time_decay_w1) @ self.time_decay_w2
        w = self.time_decay + ww

        return r, k, v, g, w

    def forward_part2(self, x, g):
        """
        split forward function to 3 part: part1, cuda, part2
        """
        B, T, C = x.size()
        x = x.view(B * T, C)

        x = self.ln_x(x).view(B, T, C)
        x = self.output(x * g)
        return x

    def as_jit(self) -> "TimeMixingState":
        self.forward_part1 = torch.jit.script(self.forward_part1)  # type: ignore
        self.forward_part2 = torch.jit.script(self.forward_part2)  # type: ignore
        return torch.jit.script(self)  # type: ignore

    def forward(self, x):
        B, T, C = x.size()
        H = self.n_head

        r, k, v, g, w = self.forward_part1(x)
        # TODO: is headsize currect??
        x = self.state_cuda.apply(B, T, C, H, r, k, v, w, u=self.time_faaaa, s=self.time_state)

        return self.forward_part2(x, g)


class ChannelMixing(Module):
    def __init__(self, layer_id: int, layer_num: int, embedding_num: int, ffn_dim: int):
        super().__init__()
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():  # fancy init of time_mix
            ratio_1_to_almost0 = 1.0 - (layer_id / layer_num)  # 1 to ~0
            ddd = torch.ones(1, 1, embedding_num)
            for i in range(embedding_num):
                ddd[0, 0, i] = i / embedding_num
            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))

        self.key = nn.Linear(embedding_num, ffn_dim, bias=False)
        self.receptance = nn.Linear(embedding_num, embedding_num, bias=False)
        self.value = nn.Linear(ffn_dim, embedding_num, bias=False)

    def forward(self, x: torch.Tensor):
        xx = self.time_shift(x) - x
        xk = x + xx * self.time_maa_k
        xr = x + xx * self.time_maa_r

        k = self.key(xk)
        k = torch.relu(k) ** 2
        kv = self.value(k)
        return torch.sigmoid(self.receptance(xr)) * kv


class Block(nn.Module):
    def __init__(
        self,
        layer_id: int,
        embedding_num: int,
        max_len: int,
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
                )
            else:
                self.att = TimeMixing(
                    layer_id=layer_id,
                    head_size=head_size,
                    head_size_divisor=head_size_divisor,
                    attn_dim=attn_dim,
                    layer_num=layer_num,
                    embedding_num=embedding_num,
                )
        self.ffn = ChannelMixing(layer_id=layer_id, layer_num=layer_num, embedding_num=embedding_num, ffn_dim=ffn_dim)
        if tiny_attn_dim > 0 and self.layer_id == attn_layer:
            self.tiny_ln = nn.LayerNorm(embedding_num)
            self.tiny_q = nn.Linear(embedding_num, tiny_attn_dim, bias=False)
            self.tiny_k = nn.Linear(embedding_num, tiny_attn_dim, bias=False)
            self.tiny_v = nn.Linear(embedding_num, embedding_num, bias=False)
            self.register_buffer("tiny_mask", torch.tril(torch.ones(max_len, max_len)))
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


class L2Wrap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, loss, y):
        ctx.save_for_backward(y)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        y = ctx.saved_tensors[0]
        # to encourage the logits to be close to 0
        factor = 1e-4 / (y.shape[0] * y.shape[1])
        maxx, ids = torch.max(y, -1, keepdim=True)
        gy = torch.zeros_like(y)
        gy.scatter_(-1, ids, maxx * factor)
        return (grad_output, gy)


class RWKV(Module):
    def __init__(
        self,
        attn_dim: int,
        tiny_attn_dim: int,
        tiny_attn_layer: int,
        embedding_num: int,
        vocab_size: int,
        head_size: int,
        head_size_divisor: int,
        layer_num: int,
        head_qk_dim: int,
        max_len: int,
        pre_ffn: bool = False,
        ffn_dim: int = 0,
        pos_embedding_dim: int = 0,
        trainable_state: bool = False,
        dropout_p: float = 0,
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
                    max_len=max_len,
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
            self.register_buffer("copy_mask", torch.tril(torch.ones(max_len, max_len)))
        if dropout_p > 0:
            self.drop0 = nn.Dropout(p=dropout_p)
        self.max_len = max_len
        self.dropout_p = dropout_p
        self.head_qk_dim = head_qk_dim
        self.vocab_size = vocab_size
        self.layer_num = layer_num
        self.embedding_num = embedding_num

    def forward(self, idx: torch.Tensor):
        B, T = idx.size()
        assert T <= self.max_len, "Cannot forward, model max_len is exhausted."

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
