from collections import OrderedDict
from typing import Optional, cast
import torch as th
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention


class NoisyLayer(th.nn.Module):
    """
    Noisy Linear Layer
    """

    def __init__(self, input_dim: int, out_dim: int):
        super(NoisyLayer, self).__init__()
        # TODO: check initialization
        # Initialize mu parameters with uniform distribution
        mu_range = 1.0 / (input_dim**0.5)
        self.weight_mu = th.nn.Parameter(th.empty(out_dim, input_dim))
        th.nn.init.uniform_(self.weight_mu, -mu_range, mu_range)

        # Initialize sigma parameters with constant value
        sigma_init = 0.5 / (input_dim**0.5)
        self.weight_sigma = th.nn.Parameter(th.full((out_dim, input_dim), sigma_init))
        # Register epsilon buffers (initialized in reset_noise)
        self.weight_epsilon = th.nn.Buffer(th.FloatTensor(out_dim, input_dim))
        self.bias_mu = th.nn.Parameter(th.empty(out_dim))
        self.bias_sigma = th.nn.Parameter(th.full((out_dim,), sigma_init))
        th.nn.init.uniform_(self.bias_mu, -mu_range, mu_range)
        self.bias_epsilon = th.nn.Buffer(th.FloatTensor(out_dim))

        self.reset_noise()

    def forward(self, x):
        weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
        bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        return F.linear(x, weight, bias)

    def reset_noise(self):
        self.weight_epsilon.normal_(mean=0.0, std=1.0)
        self.bias_epsilon.normal_(mean=0.0, std=1.0)


class SwiGLU(th.nn.Module):
    def __init__(self, input_dim: int, out_dim: int, noisy=False):
        super(SwiGLU, self).__init__()
        # the eightthirds rule for SwiGLU activation function
        hidden_dim = round((input_dim * 8 / 3) / 32) * 32
        self.u = th.nn.Linear(input_dim, hidden_dim)
        self.gate = th.nn.Linear(input_dim, hidden_dim)
        if noisy:
            self.out = NoisyLayer(hidden_dim, out_dim)
        else:
            self.out = th.nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        return self.out(self.u(x) * F.silu(self.gate(x)))

    def reset_noise(self):
        if isinstance(self.out, NoisyLayer):
            self.out.reset_noise()


class AttentionLayer(th.nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int = 6):
        super(AttentionLayer, self).__init__()
        self.Q = th.nn.Linear(embedding_dim, embedding_dim)
        self.K = th.nn.Linear(embedding_dim, embedding_dim)
        self.V = th.nn.Linear(embedding_dim, embedding_dim)

        self.SwiGLU = SwiGLU(embedding_dim, embedding_dim)

        self.norm1 = th.nn.RMSNorm(embedding_dim)
        self.norm2 = th.nn.RMSNorm(embedding_dim)
        self.qnorm = th.nn.RMSNorm(embedding_dim)
        self.knorm = th.nn.RMSNorm(embedding_dim)

        def soft_cap(score, b, h, q_idx, kv_idx):
            softcap = 30  # Gemma 4
            score = score / softcap
            score = th.tanh(score)
            score = score * softcap
            return score

        self.MHA = th.compile(
            lambda q, k, v: flex_attention(  # QKNorm use scale=1.0
                q, k, v, score_mod=soft_cap, scale=1.0, enable_gqa=False
            ),
            mode="reduce-overhead",
        )

        self.num_heads = num_heads

    def forward(self, x):
        x_prime = self.norm1(x)
        # ensure x_prime is 3D tensor for flex_attention
        x_prime = x_prime.view(-1, x.size(-2), x.size(-1))
        batch_size, seq_len, embedding_dim = x_prime.shape

        q, k, v = self.Q(x_prime), self.K(x_prime), self.V(x_prime)
        q, k = self.qnorm(q), self.knorm(k)

        # DINOv3 ViT-S has 6 heads and MHA
        q = (
            q.view(batch_size, seq_len, self.num_heads, -1)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        k = (
            k.view(batch_size, seq_len, self.num_heads, -1)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        v = (
            v.view(batch_size, seq_len, self.num_heads, -1)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        x_prime = self.MHA(q, k, v)

        x_prime = (
            x_prime.permute(0, 2, 1, 3)
            .contiguous()
            .view(batch_size, seq_len, embedding_dim)
        )
        x_prime = self.norm2(x_prime)
        x_prime = self.SwiGLU(x_prime)
        # no dropout for RL
        return x + x_prime


class Encoder(th.nn.Module):
    def __init__(self, embedding_dim: int, num_layers: int = 2):
        super(Encoder, self).__init__()
        self.model = th.nn.Sequential(
            OrderedDict(
                {f"layer_{i}": AttentionLayer(embedding_dim) for i in range(num_layers)}
            )
        )

    def forward(self, x):
        return self.model(x)


class NoisyDuelingDistributionalNetwork(th.nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_actions: int,
        num_layers: int = 2,
        n_atoms: int = 51,
        v_max=10.0,
        v_min=None,
    ):
        super(NoisyDuelingDistributionalNetwork, self).__init__()
        self.feature_extractor = Encoder(
            embedding_dim=embedding_dim, num_layers=num_layers
        )
        if v_min is None:
            v_min = -v_max
        self.support = th.nn.Buffer(th.linspace(v_min, v_max, n_atoms))

        self.advantage = th.nn.Sequential(
            SwiGLU(embedding_dim, embedding_dim, noisy=True),
            NoisyLayer(
                embedding_dim, n_atoms
            ),  # each token (image tile) is a action and has a advantage value
        )
        self.value = th.nn.Sequential(
            SwiGLU(embedding_dim, embedding_dim, noisy=True),
            NoisyLayer(embedding_dim, n_atoms),
        )
        self.num_actions = num_actions

    def forward(self, x):
        """
        output: Q values and their distribution
        """
        x = self.feature_extractor(x)
        adv = self.advantage(x[:, -self.num_actions :, :])
        val = self.value(
            x[:, 0:5, :].mean(dim=1, keepdim=True)
        )  # mean pooling for the first 5 tokens (CLS + global registers)

        x = val + adv - adv.mean(dim=1, keepdim=True)
        dist = F.softmax(x, dim=-1)
        return th.sum(dist * self.support, dim=-1), dist

    def qf(self, obs, action: th.Tensor):
        """
        obs: [B, L, E]
        action: [B]
        output: Q value and it's distribution
        """
        x = self.feature_extractor(obs)
        # x: [B, L, E]
        adv = self.advantage(x[:, -self.num_actions :, :])
        # adv: [B, A, n_atoms]
        val = self.value(
            x[:, 0:5, :].mean(dim=1, keepdim=True)
        )  # mean pooling for the first 5 tokens (CLS + global registers)
        # val: [B, 1, n_atoms]
        x = (
            val
            + get_action_embed(adv, action).unsqueeze(1)
            - adv.mean(dim=1, keepdim=True)
        )
        dist = F.softmax(x, dim=-1)
        return th.sum(dist * self.support, dim=-1), x

    def max_qf(self, obs) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        output: max Q value and it's distribution
        """
        if len(obs.shape) < 3:
            obs = obs.unsqueeze(0)

        x = self.feature_extractor(obs)
        adv: th.Tensor = self.advantage(x[:, -self.num_actions :, :])
        # mean pooling for the first 5 tokens (CLS + global registers)
        val = self.value(x[:, 0:5, :].mean(dim=1, keepdim=True))

        x = val + adv - adv.mean(dim=1, keepdim=True)

        qv = th.sum(F.softmax(x, dim=-1) * self.support, dim=-1)
        max_qv, max_idx = qv.max(dim=1, keepdim=True)
        return (
            max_qv,
            get_action_embed(x, max_idx),
            max_idx,
        )

    def reset_noise(self):
        for module in self.modules():
            if isinstance(module, NoisyLayer):
                module.reset_noise()


def get_action_embed(obs: th.Tensor, action: th.Tensor):
    """
    obs: [B, L, E]
    action: total size of [B]
    return: [B, E]
    """
    act_embed = obs.gather(1, action.view(-1, 1, 1).expand(-1, -1, obs.size(2))).view(
        -1, obs.size(2)
    )  # [B, E]
    return act_embed


class DynamicTanh(th.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        # Learnable input scaling parameter (initialized to 1.0)
        self.alpha = th.nn.Parameter(th.ones(dim))

        # Optional learnable affine parameters (similar to LayerNorm weight/bias)
        self.gamma = th.nn.Parameter(th.ones(dim))
        self.beta = th.nn.Parameter(th.zeros(dim))

    def forward(self, x: th.Tensor) -> th.Tensor:
        # x shape typically: [Batch, Sequence, Dimension]
        # Dynamically scales the input range and applies tanh squashing
        scaled_x = x * self.alpha
        out = th.tanh(scaled_x)

        # Apply affine transformation (scale and shift)
        return out * self.gamma + self.beta


if __name__ == "__main__":
    embedding_dim = 318
    num_actions = 300
    model = NoisyDuelingDistributionalNetwork(embedding_dim, num_actions).to("cuda")
    x = th.randn(32, num_actions, embedding_dim).to("cuda")
    qv, dist = model(x)
    print("qv shape:", qv.shape)
    print("dist shape:", dist.shape)

    qv, pmf = model.qf(x, th.randint(0, num_actions, (32,)).to("cuda"))
    print("qv shape:", qv.shape)
    print("pmf shape:", pmf.shape)

    max_qv, max_act_embed, max_idx = model.max_qf(x)
    print("max_qv shape:", max_qv.shape)
    print("max_act_embed shape:", max_act_embed.shape)
    print("max_idx shape:", max_idx.shape)
