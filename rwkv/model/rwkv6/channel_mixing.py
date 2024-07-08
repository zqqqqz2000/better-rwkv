from torch import nn
import torch


class ChannelMixing(nn.Module):
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
