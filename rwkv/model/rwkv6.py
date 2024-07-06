########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import os, math, gc
import torch

import torch.nn as nn
from torch.nn import Module, functional as F
import pytorch_lightning as pl
from pytorch_lightning.strategies import DeepSpeedStrategy
from torch.utils.cpp_extension import load

# if importlib.util.find_spec("deepspeed"):
#     import deepspeed
#     from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam


def init_state_autograd(head_size: int, ctx_len: int):
    wkv6state_cuda = load(
        name="wkv6state",
        sources=["cuda/wkv6state_op.cpp", f"cuda/wkv6state_cuda.cu"],
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
                sources=["cuda/wkv6_op.cpp", f"cuda/wkv6_cuda.cu"],
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
                y = torch.empty(
                    (B, T, C), device=r.device, dtype=torch.bfloat16, memory_format=torch.contiguous_format
                )  # .uniform_(-100, 100)
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
                )  # .uniform_(-100, 100)
                gk = torch.empty(
                    (B, T, C),
                    device=gy.device,
                    requires_grad=False,
                    dtype=torch.bfloat16,
                    memory_format=torch.contiguous_format,
                )  # .uniform_(-100, 100)
                gv = torch.empty(
                    (B, T, C),
                    device=gy.device,
                    requires_grad=False,
                    dtype=torch.bfloat16,
                    memory_format=torch.contiguous_format,
                )  # .uniform_(-100, 100)
                gw = torch.empty(
                    (B, T, C),
                    device=gy.device,
                    requires_grad=False,
                    dtype=torch.bfloat16,
                    memory_format=torch.contiguous_format,
                )  # .uniform_(-100, 100)
                gu = torch.empty(
                    (B, C),
                    device=gy.device,
                    requires_grad=False,
                    dtype=torch.bfloat16,
                    memory_format=torch.contiguous_format,
                )  # .uniform_(-100, 100)
                self.wkv6_cuda.backward(B, T, C, H, r, k, v, w, u, gy, gr, gk, gv, gw, gu)
                gu = torch.sum(gu, 0).view(H, C // H)
                return (None, None, None, None, gr, gk, gv, gw, gu)

    return RWKVCuda


class RWKV_Tmix_x060(Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.head_size = args.head_size_a
        self.n_head = args.dim_att // self.head_size
        assert args.dim_att % self.n_head == 0

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (args.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd

            # fancy time_mix
            self.time_maa_x = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_w = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_v = nn.Parameter(1.0 - (torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1))
            self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))
            self.time_maa_g = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))

            D_MIX_LORA = 32  # generate TIME_MIX for w,k,v,r,g
            self.time_maa_w1 = nn.Parameter(torch.zeros(args.n_embd, D_MIX_LORA * 5))
            self.time_maa_w2 = nn.Parameter(torch.zeros(5, D_MIX_LORA, args.n_embd).uniform_(-0.01, 0.01))

            # fancy time_decay
            decay_speed = torch.ones(args.dim_att)
            for n in range(args.dim_att):
                decay_speed[n] = -6 + 5 * (n / (args.dim_att - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed.reshape(1, 1, args.dim_att))

            D_DECAY_LORA = 64
            self.time_decay_w1 = nn.Parameter(torch.zeros(args.n_embd, D_DECAY_LORA))
            self.time_decay_w2 = nn.Parameter(torch.zeros(D_DECAY_LORA, args.dim_att).uniform_(-0.01, 0.01))

            tmp = torch.zeros(args.dim_att)
            for n in range(args.dim_att):
                zigzag = ((n + 1) % 3 - 1) * 0.1
                tmp[n] = ratio_0_to_1 * (1 - (n / (args.dim_att - 1))) + zigzag

            self.time_faaaa = nn.Parameter(tmp.reshape(self.n_head, self.head_size))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.receptance = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.key = nn.Linear(args.n_embd, args.dim_att, bias=False)

        self.value = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.output = nn.Linear(args.dim_att, args.n_embd, bias=False)
        self.gate = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.ln_x = nn.GroupNorm(self.n_head, args.dim_att, eps=(1e-5) * (args.head_size_divisor**2))
        # TODO: ctxlen
        self.rwkv_cuda = init_rwkv_cuda(
            self.head_size,
        )

    def as_jit(self) -> "TimeMixingState":
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


########################################################################################################


class TimeMixingState(Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.head_size = args.head_size_a
        self.n_head = args.dim_att // self.head_size
        assert args.dim_att % self.n_head == 0

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (args.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd

            # fancy time_mix
            self.time_maa_x = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_w = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_v = nn.Parameter(1.0 - (torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1))
            self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))
            self.time_maa_g = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))

            D_MIX_LORA = 32  # generate TIME_MIX for w,k,v,r,g
            self.time_maa_w1 = nn.Parameter(torch.zeros(args.n_embd, D_MIX_LORA * 5))
            self.time_maa_w2 = nn.Parameter(torch.zeros(5, D_MIX_LORA, args.n_embd).uniform_(-0.01, 0.01))

            # fancy time_decay
            decay_speed = torch.ones(args.dim_att)
            for n in range(args.dim_att):
                decay_speed[n] = -6 + 5 * (n / (args.dim_att - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed.reshape(1, 1, args.dim_att))

            D_DECAY_LORA = 64
            self.time_decay_w1 = nn.Parameter(torch.zeros(args.n_embd, D_DECAY_LORA))
            self.time_decay_w2 = nn.Parameter(torch.zeros(D_DECAY_LORA, args.dim_att).uniform_(-0.01, 0.01))

            tmp = torch.zeros(args.dim_att)
            for n in range(args.dim_att):
                zigzag = ((n + 1) % 3 - 1) * 0.1
                tmp[n] = ratio_0_to_1 * (1 - (n / (args.dim_att - 1))) + zigzag

            self.time_faaaa = nn.Parameter(tmp.reshape(self.n_head, self.head_size))
            self.time_state = nn.Parameter(torch.zeros(self.n_head, self.head_size, self.head_size))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.receptance = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.key = nn.Linear(args.n_embd, args.dim_att, bias=False)

        self.value = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.output = nn.Linear(args.dim_att, args.n_embd, bias=False)
        self.gate = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.ln_x = nn.GroupNorm(self.n_head, args.dim_att, eps=(1e-5) * (args.head_size_divisor**2))
        self.state_cuda = init_state_autograd(self.ctx_len, self.head_size)

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
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():  # fancy init of time_mix
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd
            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))

        self.key = nn.Linear(args.n_embd, args.dim_ffn, bias=False)
        self.receptance = nn.Linear(args.n_embd, args.n_embd, bias=False)
        self.value = nn.Linear(args.dim_ffn, args.n_embd, bias=False)

    def forward(self, x):
        xx = self.time_shift(x) - x
        xk = x + xx * self.time_maa_k
        xr = x + xx * self.time_maa_r

        k = self.key(xk)
        k = torch.relu(k) ** 2
        kv = self.value(k)
        return torch.sigmoid(self.receptance(xr)) * kv


class Block(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(args.n_embd)
        self.ln2 = nn.LayerNorm(args.n_embd)

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(args.n_embd)
            if args.my_pos_emb > 0:
                self.pos_emb_x = nn.Parameter(torch.zeros((1, args.my_pos_emb, args.n_embd)))
                self.pos_emb_y = nn.Parameter(torch.zeros((args.my_pos_emb, 1, args.n_embd)))

        if os.environ["RWKV_TRAIN_TYPE"] == "states":
            self.att = TimeMixingState(args, layer_id)
        else:
            self.att = RWKV_Tmix_x060(args, layer_id)
        self.ffn = ChannelMixing(args, layer_id)
        if args.tiny_att_dim > 0 and self.layer_id == args.tiny_att_layer:
            self.tiny_ln = nn.LayerNorm(args.n_embd)
            self.tiny_q = nn.Linear(args.n_embd, args.tiny_att_dim, bias=False)
            self.tiny_k = nn.Linear(args.n_embd, args.tiny_att_dim, bias=False)
            self.tiny_v = nn.Linear(args.n_embd, args.n_embd, bias=False)
            self.register_buffer("tiny_mask", torch.tril(torch.ones(args.ctx_len, args.ctx_len)))

        if args.dropout > 0:
            self.drop0 = nn.Dropout(p=args.dropout)
            self.drop1 = nn.Dropout(p=args.dropout)

    def forward(self, x, x_emb=None):
        args = self.args
        B, T, C = x.size()
        if self.layer_id == 0:
            x = self.ln0(x)
            if args.my_pos_emb > 0:
                pos_emb = (self.pos_emb_x + self.pos_emb_y).reshape(T + 1, -1)[:-1, :]
                x = x + pos_emb

        if self.args.dropout == 0:
            if self.layer_id == 0 and args.pre_ffn > 0:
                x = x + self.ffnPre(self.ln1(x))
            else:
                x = x + self.att(self.ln1(x))
            x = x + self.ffn(self.ln2(x))
        else:
            if self.layer_id == 0 and args.pre_ffn > 0:
                x = self.drop0(x + self.ffnPre(self.ln1(x)))
            else:
                x = self.drop0(x + self.att(self.ln1(x)))
            x = self.drop1(x + self.ffn(self.ln2(x)))

        if args.tiny_att_dim > 0 and self.layer_id == args.tiny_att_layer:
            xx = self.tiny_ln(x)
            q = self.tiny_q(xx)[:, :T, :]
            k = self.tiny_k(xx)[:, :T, :]
            c = (q @ k.transpose(-2, -1)) * (args.tiny_att_dim ** (-0.5))
            c = c.masked_fill(self.tiny_mask[:T, :T] == 0, 0)
            x = x + c @ self.tiny_v(x_emb)
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


class RWKV(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        if not hasattr(args, "dim_att"):
            args.dim_att = args.n_embd
        if not hasattr(args, "dim_ffn"):
            if "-f4" in os.environ["RWKV_MY_TESTING"]:
                args.dim_ffn = int((args.n_embd * 4) // 32 * 32)
            else:
                args.dim_ffn = int((args.n_embd * 3.5) // 32 * 32)  # default = 3.5x emb size
        if not hasattr(args, "tiny_att_layer"):
            args.tiny_att_layer = -1
        if not hasattr(args, "tiny_att_dim"):
            args.tiny_att_dim = -1
        assert args.n_embd % 32 == 0
        assert args.dim_att % 32 == 0
        assert args.dim_ffn % 32 == 0

        self.emb = nn.Embedding(args.vocab_size, args.n_embd)

        self.blocks = nn.ModuleList([Block(args, i) for i in range(args.n_layer)])

        self.ln_out = nn.LayerNorm(args.n_embd)
        self.head = nn.Linear(args.n_embd, args.vocab_size, bias=False)

        if args.head_qk > 0:
            self.head_q = nn.Linear(args.n_embd, args.head_qk, bias=False)
            self.head_k = nn.Linear(args.n_embd, args.head_qk, bias=False)
            self.register_buffer("copy_mask", torch.tril(torch.ones(args.ctx_len, args.ctx_len)))
        if args.dropout > 0:
            self.drop0 = nn.Dropout(p=args.dropout)

    def configure_optimizers(self):
        args = self.args

        lr_decay = set()
        lr_1x = set()
        lr_2x = set()
        lr_3x = set()
        for n, p in self.named_parameters():

            # if not p.requires_grad:
            #     continue
            if args.train_type == "states":
                if "time_sta" not in n:
                    continue

            if (("_w1" in n) or ("_w2" in n)) and (args.layerwise_lr > 0):
                lr_1x.add(n)
            elif ("time_sta" in n) and (args.weight_decay > 0):
                lr_decay.add(n)
            elif (("time_mix" in n) or ("time_maa" in n)) and (args.layerwise_lr > 0):
                if args.my_pile_stage == 2:
                    lr_2x.add(n)
                else:
                    lr_1x.add(n)
            elif (("time_decay" in n) or ("time_daaaa" in n)) and (args.layerwise_lr > 0):
                if args.my_pile_stage == 2:
                    lr_3x.add(n)
                else:
                    lr_2x.add(n)
            elif ("time_faaaa" in n) and (args.layerwise_lr > 0):
                if args.my_pile_stage == 2:
                    lr_2x.add(n)
                else:
                    lr_1x.add(n)
            elif ("time_first" in n) and (args.layerwise_lr > 0):
                lr_3x.add(n)
            elif (len(p.squeeze().shape) >= 2) and (args.weight_decay > 0):
                lr_decay.add(n)
            else:
                lr_1x.add(n)

        lr_decay = sorted(list(lr_decay))
        lr_1x = sorted(list(lr_1x))
        lr_2x = sorted(list(lr_2x))
        lr_3x = sorted(list(lr_3x))

        if self.trainer.is_global_zero:
            print("decay", lr_decay, "\n")
            print("1x", lr_1x, "\n")
            print("2x", lr_2x, "\n")
            print("3x", lr_3x, "\n")

        param_dict = {n: p for n, p in self.named_parameters()}

        if args.layerwise_lr > 0:
            if args.my_pile_stage == 2:
                optim_groups = [
                    {"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0},
                    {
                        "params": [param_dict[n] for n in lr_2x],
                        "weight_decay": 0.0,
                        "my_lr_scale": 5.0,
                    },  # test: 2e-3 / args.lr_init},
                    {
                        "params": [param_dict[n] for n in lr_3x],
                        "weight_decay": 0.0,
                        "my_lr_scale": 5.0,
                    },  # test: 3e-3 / args.lr_init},
                ]
            else:
                optim_groups = [
                    {"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0},
                    {"params": [param_dict[n] for n in lr_2x], "weight_decay": 0.0, "my_lr_scale": 2.0},
                    {"params": [param_dict[n] for n in lr_3x], "weight_decay": 0.0, "my_lr_scale": 3.0},
                ]
        else:
            optim_groups = [{"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0}]

        if args.weight_decay > 0:
            optim_groups += [
                {"params": [param_dict[n] for n in lr_decay], "weight_decay": args.weight_decay, "my_lr_scale": 1.0}
            ]
            if self.deepspeed_offload:
                return DeepSpeedCPUAdam(
                    optim_groups,
                    lr=self.args.lr_init,
                    betas=self.args.betas,
                    eps=self.args.adam_eps,
                    bias_correction=True,
                    adamw_mode=True,
                    amsgrad=False,
                )
            return FusedAdam(
                optim_groups,
                lr=self.args.lr_init,
                betas=self.args.betas,
                eps=self.args.adam_eps,
                bias_correction=True,
                adam_w_mode=True,
                amsgrad=False,
            )
        else:
            if self.deepspeed_offload:
                return DeepSpeedCPUAdam(
                    optim_groups,
                    lr=self.args.lr_init,
                    betas=self.args.betas,
                    eps=self.args.adam_eps,
                    bias_correction=True,
                    adamw_mode=False,
                    weight_decay=0,
                    amsgrad=False,
                )
            return FusedAdam(
                optim_groups,
                lr=self.args.lr_init,
                betas=self.args.betas,
                eps=self.args.adam_eps,
                bias_correction=True,
                adam_w_mode=False,
                weight_decay=0,
                amsgrad=False,
            )
        # return ZeroOneAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, weight_decay=0, amsgrad=False, cuda_aware=False)

    @property
    def deepspeed_offload(self) -> bool:
        strategy = self.trainer.strategy
        if isinstance(strategy, DeepSpeedStrategy):
            cfg = strategy.config["zero_optimization"]
            return cfg.get("offload_optimizer") or cfg.get("offload_param")
        return False

    def forward(self, idx):
        args = self.args
        B, T = idx.size()
        assert T <= args.ctx_len, "Cannot forward, model ctx_len is exhausted."

        x = self.emb(idx)
        x_emb = x

        if args.dropout > 0:
            x = self.drop0(x)
        if args.tiny_att_dim > 0:
            for block in self.blocks:
                if args.grad_cp == 1:
                    x = deepspeed.checkpointing.checkpoint(block, x, x_emb)
                else:
                    x = block(x, x_emb)
        else:
            for block in self.blocks:
                if args.grad_cp == 1:
                    x = deepspeed.checkpointing.checkpoint(block, x)
                else:
                    x = block(x)

        x = self.ln_out(x)

        if args.head_qk > 0:
            q = self.head_q(x)[:, :T, :]
            k = self.head_k(x)[:, :T, :]
            c = (q @ k.transpose(-2, -1)) * (1.0 / args.head_qk)
            c = c.masked_fill(self.copy_mask[:T, :T] == 0, 0)

            if "32" in os.environ["RWKV_FLOAT_MODE"]:
                c = c @ F.one_hot(idx, num_classes=args.vocab_size)
            elif os.environ["RWKV_FLOAT_MODE"] == "fp16":
                c = c @ F.one_hot(idx, num_classes=args.vocab_size).half()
            elif os.environ["RWKV_FLOAT_MODE"] == "bf16":
                c = c @ F.one_hot(idx, num_classes=args.vocab_size).bfloat16()

            x = self.head(x) + c
        else:
            x = self.head(x)

        return x

    def training_step(self, batch, batch_idx):
        args = self.args
        if args.my_qa_mask != 1:
            idx, targets = batch
            logits = self(idx)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            # if '0' in os.environ["RWKV_MY_TESTING"]:
            #     print('logits', logits)
            #     torch.set_printoptions(threshold=10000)
            #     print('idx', idx)
            #     exit(0)
        else:
            idx, targets, mask = batch
            mask = mask.view(-1)
            sum_mask = torch.sum(mask).item()
            # if sum_mask == 0:
            #     return torch.tensor([0.0], requires_grad=True)

            logits = self(idx)
            if sum_mask == mask.shape[0]:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
                # print('rank', self.global_rank, 'loss', loss.item())
            else:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction="none")
                # loss_raw = loss
                loss = torch.sum(loss * mask) / sum_mask

                # torch.set_printoptions(threshold=10000)
                # if True: #self.global_rank == 1:
                #     tmp = ''
                #     sss = 0
                #     ccc = 0
                #     for i in range(mask.shape[0]):
                #         if mask[i] > 0:
                #             tmp += str(idx.view(-1)[i].item()) + ','
                #             sss += loss_raw.view(-1)[i].float().item()
                #             ccc += 1
                #     print('rank', self.global_rank, 'loss', loss.item(), 'lavg', sss / ccc)#, 'tmp', tmp, 'input', idx)

        return L2Wrap.apply(loss, logits)

    def training_step_end(self, batch_parts):
        if pl.__version__[0] != "2":
            all = self.all_gather(batch_parts)
            if self.trainer.is_global_zero:
                self.trainer.my_loss_all = all

    def generate_init_weight(self):
        print(
            f"""
############################################################################
#
# Init model weight (slow for large models)...
#
############################################################################
"""
        )
        m = {}
        n_params = 0
        for n in self.state_dict():
            p = self.state_dict()[n]
            shape = p.shape

            s0 = str(shape[0]) if len(shape) > 0 else ""
            s1 = str(shape[1]) if len(shape) > 1 else ""
            s2 = str(shape[2]) if len(shape) > 2 else ""
            print(f"{s0.ljust(5)} {s1.ljust(5)} {s2.ljust(5)} {n}", end="")

            scale = 1.0
            if (
                "ln_" in n
                or ".ln" in n
                or "time_" in n
                or "_mask" in n
                or "pos_emb" in n
                or ".mask." in n
                or n.endswith("_w")
                or n.endswith("_w1")
                or n.endswith("_w2")
                or n.endswith("_bias")
            ):
                if "ln_x.weight" in n:
                    layer_scale = (1 + int(n.split(".")[1])) / self.args.n_layer
                    m[n] = (p * 0.0) + (layer_scale**0.7)
                else:
                    m[n] = p
                print()
            elif n == "emb.weight":
                m[n] = p
                scale = -1e-4
                nn.init.uniform_(m[n], a=scale, b=-scale)
                print(f" [scale {scale}]")
            elif n == "head.weight":
                m[n] = p
                if self.args.vocab_size > self.args.n_embd:
                    scale = 0.5 * math.sqrt(self.args.vocab_size / self.args.n_embd)
                else:
                    scale = 0.5
                nn.init.orthogonal_(m[n], gain=scale)
                print(f" [scale {scale}]")
            else:
                if "mamba" in os.environ["RWKV_MY_TESTING"]:
                    m[n] = p
                    if ".out_proj.weight" in n:
                        scale = 0
                        nn.init.zeros_(m[n])
                        print(f" [scale {scale}]")
                    elif ".bias" in n:
                        scale = 0
                        nn.init.zeros_(m[n])
                        print(f" [scale {scale}]")
                    else:
                        print()
                else:
                    assert n.endswith(".weight")  # should always be true

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
                        if kk in n:
                            scale = 0
                    if "head_k." in n:
                        scale = 0.1
                    if "head_q." in n:
                        scale = 0

                    for kk in [".att.key."]:
                        if kk in n:
                            scale = 0.1
                    for kk in [".att.gate."]:
                        if kk in n:
                            scale = 0.1

                    print(f" [scale {scale}]")

                    if self.args.accelerator.upper() == "GPU":
                        m[n] = torch.empty((shape[0], shape[1]), device="cuda")
                    else:
                        m[n] = torch.empty((shape[0], shape[1]))

                    if scale == 0:
                        nn.init.zeros_(m[n])
                    elif scale < 0:
                        nn.init.uniform_(m[n], a=scale, b=-scale)
                    else:
                        nn.init.orthogonal_(m[n], gain=scale)

            m[n] = m[n].cpu()
            if os.environ["RWKV_FLOAT_MODE"] == "fp16":
                m[n] = m[n].half()
            elif os.environ["RWKV_FLOAT_MODE"] == "bf16":
                m[n] = m[n].bfloat16()
            n_params += m[n].numel()

            # if n == "emb.weight":
            #     print(m[n])

        print("model params", n_params)
        gc.collect()
        torch.cuda.empty_cache()
        return m
