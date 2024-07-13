from torch import nn
import torch
from torch.utils.cpp_extension import load


def init_state_autograd(head_size: int, ctx_len: int):
    wkv6state_cuda = load(
        name="wkv6state",
        sources=["time_mixing_state_op.cpp", f"time_mixing_state_cuda.cu"],
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
                sources=["time_mixing_op.cpp", f"time_mixing_cuda.cu"],
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


class TimeMixing(nn.Module):
    def __init__(
        self,
        layer_id: int,
        head_size: int,
        head_size_divisor: int,
        attn_dim: int,
        layer_num: int,
        embedding_num: int,
        tiny_ctx_len: int,
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

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.receptance = nn.Linear(embedding_num, attn_dim, bias=False)
        self.key = nn.Linear(embedding_num, attn_dim, bias=False)

        self.value = nn.Linear(embedding_num, attn_dim, bias=False)
        self.output = nn.Linear(attn_dim, embedding_num, bias=False)
        self.gate = nn.Linear(embedding_num, attn_dim, bias=False)
        self.ln_x = nn.GroupNorm(self.n_head, attn_dim, eps=(1e-5) * (head_size_divisor**2))
        self.rwkv_cuda = init_rwkv_cuda(self.head_size, tiny_ctx_len)

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


class TimeMixingState(nn.Module):
    def __init__(
        self,
        layer_id: int,
        head_size: int,
        head_size_divisor: int,
        attn_dim: int,
        layer_num: int,
        embedding_num: int,
        tiny_ctx_len: int,
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
        self.state_cuda = init_state_autograd(self.head_size, tiny_ctx_len)

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
