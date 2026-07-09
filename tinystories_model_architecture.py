from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 256
    block_size: int = 128
    d_model: int = 128
    num_layers: int = 4
    num_heads: int = 4
    mlp_hidden_size: int = 512
    dropout: float = 0.1


class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.d_model % config.num_heads != 0:
            raise ValueError("d_model must divide evenly by num_heads")

        self.num_heads = config.num_heads
        self.head_dim = config.d_model // config.num_heads

        self.q_proj = nn.Linear(config.d_model, config.d_model)
        self.k_proj = nn.Linear(config.d_model, config.d_model)
        self.v_proj = nn.Linear(config.d_model, config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)

        self.attention_dropout = nn.Dropout(config.dropout)
        self.output_dropout = nn.Dropout(config.dropout)

        mask = torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool))
        self.register_buffer("causal_mask", mask)

    def forward(self, x):
        batch, time, d_model = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(batch, time, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, time, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, time, self.num_heads, self.head_dim).transpose(1, 2)

        scores = q @ k.transpose(-2, -1)
        scores = scores / (self.head_dim ** 0.5)

        mask = self.causal_mask[:time, :time]
        scores = scores.masked_fill(mask == 0, float("-inf"))

        weights = F.softmax(scores, dim=-1)
        weights = self.attention_dropout(weights)
        self.last_attention_weights = weights.detach()

        out = weights @ v
        out = out.transpose(1, 2).contiguous().view(batch, time, d_model)
        out = self.out_proj(out)
        out = self.output_dropout(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = CausalMultiHeadSelfAttention(config)
        self.ln1 = nn.LayerNorm(config.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, config.mlp_hidden_size),
            nn.GELU(),
            nn.Linear(config.mlp_hidden_size, config.d_model),
            nn.Dropout(config.dropout),
        )
        self.ln2 = nn.LayerNorm(config.d_model)

    def forward(self, x):
        x = self.ln1(x + self.attention(x))
        x = self.ln2(x + self.mlp(x))
        return x


class TinyStoriesGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.block_size, config.d_model)
        self.blocks = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config.num_layers)]
        )
        self.final_ln = nn.LayerNorm(config.d_model)
        self.output_projection = nn.Linear(config.d_model, config.vocab_size)

    def forward(self, token_ids, targets=None):
        batch, time = token_ids.shape
        if time > self.config.block_size:
            raise ValueError("input sequence is longer than block_size")

        token_vectors = self.token_embedding(token_ids)
        position_ids = torch.arange(time, device=token_ids.device)
        position_vectors = self.position_embedding(position_ids)

        x = token_vectors + position_vectors
        x = self.blocks(x)
        x = self.final_ln(x)
        logits = self.output_projection(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(batch * time, self.config.vocab_size),
                targets.reshape(batch * time),
            )

        return logits, loss


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters())


def count_trainable_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def main():
    torch.manual_seed(1337)

    config = GPTConfig()
    model = TinyStoriesGPT(config)

    batch_size = 4
    dummy_input = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(batch_size, config.block_size),
    )
    dummy_targets = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(batch_size, config.block_size),
    )

    logits, loss = model(dummy_input, dummy_targets)

    print("model config:")
    print(f"vocab_size: {config.vocab_size}")
    print(f"block_size: {config.block_size}")
    print(f"d_model: {config.d_model}")
    print(f"num_layers: {config.num_layers}")
    print(f"num_heads: {config.num_heads}")
    print(f"head_dim: {config.d_model // config.num_heads}")
    print(f"mlp_hidden_size: {config.mlp_hidden_size}")
    print(f"dropout: {config.dropout}")
    print()

    print("shape check:")
    print(f"input token ids shape: {dummy_input.shape}")
    print(f"logits shape: {logits.shape}")
    print(f"loss: {loss.item():.4f}")
    print()

    print("parameter count:")
    print(f"total parameters: {count_parameters(model):,}")
    print(f"trainable parameters: {count_trainable_parameters(model):,}")
    print()

    print("oracle checks:")
    print(f"logits shape is batch, time, vocab: {logits.shape == (batch_size, config.block_size, config.vocab_size)}")
    print(f"d_model divides evenly by num_heads: {config.d_model % config.num_heads == 0}")


if __name__ == "__main__":
    main()
