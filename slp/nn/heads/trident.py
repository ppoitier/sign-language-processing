import torch
from torch import nn
import torch.nn.functional as F

from slp.core.registry import HEAD_REGISTRY


@HEAD_REGISTRY.register("trident")
class TridentHead(nn.Module):
    def __init__(self, n_bins: int):
        super().__init__()
        self.n_bins = n_bins
        # bin index b: distance (in instants) from the reference point t
        # shape (1, B+1, 1) to broadcast against (N, B+1, T)
        self.register_buffer(
            "bin_range",
            torch.arange(n_bins + 1, dtype=torch.float32).view(1, -1, 1),
        )

    def forward(self, x):
        """
        Args:
            x: (N, 2 + 2*(B+1), T) — channels are [F_s, F_e, F_c_start, F_c_end]
        Returns:
            (N, 2, T) — channels are [d_start, d_end] in units of instants
        """
        B = self.n_bins
        # split along channel dim
        f_s, f_e, fc_s, fc_e = torch.split(x, [1, 1, B + 1, B + 1], dim=1)
        f_s = f_s.squeeze(1)  # (N, T)
        f_e = f_e.squeeze(1)  # (N, T)

        # --- Start branch ---
        # Bin set for instant t is F_s[t-B : t+1]; bin b counts distance back from t,
        # so bin 0 must correspond to F_s[t] and bin B to F_s[t-B].
        f_s_pad = F.pad(f_s, (B, 0))  # (N, T+B)
        f_s_win = f_s_pad.unfold(1, B + 1, 1)  # (N, T, B+1), index 0 = t-B
        f_s_win = f_s_win.flip(-1)  # now index 0 = t
        f_s_win = f_s_win.transpose(1, 2)  # (N, B+1, T)
        start_probs = torch.softmax(f_s_win + fc_s, dim=1)  # softmax over bins
        d_start = (self.bin_range * start_probs).sum(dim=1, keepdim=True)  # (N, 1, T)

        # --- End branch ---
        # Bin set is F_e[t : t+B+1]; bin 0 = F_e[t], bin B = F_e[t+B]. No flip needed.
        f_e_pad = F.pad(f_e, (0, B))  # (N, T+B)
        f_e_win = f_e_pad.unfold(1, B + 1, 1)  # (N, T, B+1)
        f_e_win = f_e_win.transpose(1, 2)  # (N, B+1, T)
        end_probs = torch.softmax(f_e_win + fc_e, dim=1)
        d_end = (self.bin_range * end_probs).sum(dim=1, keepdim=True)  # (N, 1, T)

        return torch.cat([d_start, d_end], dim=1)  # (N, 2, T)


if __name__ == "__main__":
    model = TridentHead(16)
    x = torch.randn(3, 2 + 2 * 17, 128)
    print(model(x).shape)  # torch.Size([3, 2, 128])
