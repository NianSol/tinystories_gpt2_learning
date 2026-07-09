import csv
import math
import os
import time
from pathlib import Path

import torch
import torch.nn.utils as nn_utils

from tinystories_model_architecture import (
    GPTConfig,
    TinyStoriesGPT,
    count_parameters,
)
from train_bpe_tinystories import (
    MIN_PAIR_COUNT as DEFAULT_MIN_PAIR_COUNT,
    TARGET_VOCAB_SIZE as DEFAULT_TARGET_VOCAB_SIZE,
    TRAIN_FRACTION,
    TRAIN_PATH,
    decode,
    encode,
    read_fraction,
    train_bpe,
)


def env_int(name, default):
    return int(os.environ.get(name, default))


def env_float(name, default):
    return float(os.environ.get(name, default))


def env_bool(name, default):
    value = os.environ.get(name)
    if value is None:
        return default

    return value.lower() not in ("0", "false", "no", "off", "")


RUN_DIR = Path(os.environ.get("RUN_DIR", "runs/tinystories_gpt"))
CHECKPOINT_DIR = RUN_DIR / "checkpoints"
LOG_PATH = RUN_DIR / "loss_log.csv"
CACHE_DIR = Path("data/tinystories/cache")

TOKENIZER_TRAIN_FRACTION = env_float("TRAIN_FRACTION", TRAIN_FRACTION)
TARGET_VOCAB_SIZE = env_int("TARGET_VOCAB_SIZE", DEFAULT_TARGET_VOCAB_SIZE)
MIN_PAIR_COUNT = env_int("MIN_PAIR_COUNT", DEFAULT_MIN_PAIR_COUNT)
TRAIN_SPLIT = 0.9
USE_TOKEN_CACHE = env_bool("TOKEN_CACHE", True)

BATCH_SIZE = env_int("BATCH_SIZE", 16)
BLOCK_SIZE = env_int("BLOCK_SIZE", 128)
D_MODEL = env_int("D_MODEL", 128)
NUM_LAYERS = env_int("NUM_LAYERS", 4)
NUM_HEADS = env_int("NUM_HEADS", 4)
MLP_HIDDEN_SIZE = env_int("MLP_HIDDEN_SIZE", 4 * D_MODEL)
DROPOUT = env_float("DROPOUT", 0.1)

MAX_STEPS = env_int("MAX_STEPS", 200)
GRAD_ACCUM_STEPS = env_int("GRAD_ACCUM_STEPS", 4)
MAX_LR = env_float("MAX_LR", 3e-4)
MIN_LR = env_float("MIN_LR", 3e-5)
WARMUP_STEPS = env_int("WARMUP_STEPS", 20)
WEIGHT_DECAY = env_float("WEIGHT_DECAY", 0.1)
GRAD_CLIP = env_float("GRAD_CLIP", 1.0)

EVAL_EVERY = env_int("EVAL_EVERY", 50)
EVAL_ITERS = env_int("EVAL_ITERS", 20)
LOG_EVERY = env_int("LOG_EVERY", 10)
CHECKPOINT_EVERY = env_int("CHECKPOINT_EVERY", 100)
SAMPLE_EVERY = env_int("SAMPLE_EVERY", 100)
SAMPLE_TOKENS = env_int("SAMPLE_TOKENS", 80)
TEMPERATURE = env_float("TEMPERATURE", 1.0)
SEED = env_int("SEED", 1337)

USE_MIXED_PRECISION = env_bool("MIXED_PRECISION", True)
USE_TORCH_COMPILE = env_bool("COMPILE", False)
COMPILE_MODE = os.environ.get("COMPILE_MODE", "default")
CACHE_VERSION = 1


def sync_if_cuda(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def fraction_label(fraction):
    label = f"{fraction:.8f}".rstrip("0").rstrip(".")
    return label.replace(".", "p")


def token_cache_path(file_size, byte_count):
    return CACHE_DIR / (
        f"tinystories_bpe_cache_v{CACHE_VERSION}_"
        f"vocab{TARGET_VOCAB_SIZE}_"
        f"min{MIN_PAIR_COUNT}_"
        f"frac{fraction_label(TOKENIZER_TRAIN_FRACTION)}_"
        f"bytes{byte_count}_"
        f"filesize{file_size}.pt"
    )


def build_vocab_maps(vocab_tokens):
    token_to_id = {token: token_id for token_id, token in enumerate(vocab_tokens)}
    id_to_token = {token_id: token for token, token_id in token_to_id.items()}
    return token_to_id, id_to_token


def load_torch_cache(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_or_build_token_cache():
    file_size = TRAIN_PATH.stat().st_size
    byte_count = int(file_size * TOKENIZER_TRAIN_FRACTION)
    cache_path = token_cache_path(file_size, byte_count)

    if USE_TOKEN_CACHE and cache_path.exists():
        cache = load_torch_cache(cache_path)
        vocab_tokens = cache["vocab_tokens"]
        token_to_id, id_to_token = build_vocab_maps(vocab_tokens)

        print("data:")
        print(f"training file: {TRAIN_PATH}")
        print(f"file size bytes: {cache['file_size']:,}")
        print(f"fraction read: {cache['train_fraction']:.2%}")
        print(f"bytes read: {cache['byte_count']:,}")
        print(f"characters read: {cache['char_count']:,}")
        print(f"loaded token cache: {cache_path}")
        print()

        return {
            "cache_path": cache_path,
            "loaded_from_cache": True,
            "file_size": cache["file_size"],
            "byte_count": cache["byte_count"],
            "char_count": cache["char_count"],
            "merge_rules": cache["merge_rules"],
            "vocab_tokens": vocab_tokens,
            "token_to_id": token_to_id,
            "id_to_token": id_to_token,
            "token_tensor": cache["token_ids"].long(),
            "round_trip_ok": cache["round_trip_ok"],
        }

    text, file_size, byte_count = read_fraction(TRAIN_PATH, TOKENIZER_TRAIN_FRACTION)

    print("data:")
    print(f"training file: {TRAIN_PATH}")
    print(f"file size bytes: {file_size:,}")
    print(f"fraction read: {TOKENIZER_TRAIN_FRACTION:.2%}")
    print(f"bytes read: {byte_count:,}")
    print(f"characters read: {len(text):,}")
    if USE_TOKEN_CACHE:
        print(f"token cache miss: {cache_path}")
    else:
        print("token cache disabled")
    print()

    print("building tokenizer cache:")
    print(f"target vocab size: {TARGET_VOCAB_SIZE}")
    print(f"minimum pair count: {MIN_PAIR_COUNT}")
    merge_rules, vocab_tokens, token_to_id, id_to_token = train_bpe(
        text,
        TARGET_VOCAB_SIZE,
        MIN_PAIR_COUNT,
        verbose=False,
    )
    _encoded_tokens, token_ids = encode(text, merge_rules, token_to_id)
    _, decoded_text = decode(token_ids, id_to_token)
    round_trip_ok = decoded_text == text
    assert round_trip_ok

    token_tensor = torch.tensor(token_ids, dtype=torch.long)

    if USE_TOKEN_CACHE:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "cache_version": CACHE_VERSION,
                "source_path": str(TRAIN_PATH),
                "file_size": file_size,
                "train_fraction": TOKENIZER_TRAIN_FRACTION,
                "byte_count": byte_count,
                "char_count": len(text),
                "target_vocab_size": TARGET_VOCAB_SIZE,
                "min_pair_count": MIN_PAIR_COUNT,
                "merge_rules": merge_rules,
                "vocab_tokens": vocab_tokens,
                "token_ids": token_tensor,
                "round_trip_ok": round_trip_ok,
            },
            cache_path,
        )
        print(f"saved token cache: {cache_path}")

    return {
        "cache_path": cache_path,
        "loaded_from_cache": False,
        "file_size": file_size,
        "byte_count": byte_count,
        "char_count": len(text),
        "merge_rules": merge_rules,
        "vocab_tokens": vocab_tokens,
        "token_to_id": token_to_id,
        "id_to_token": id_to_token,
        "token_tensor": token_tensor,
        "round_trip_ok": round_trip_ok,
    }


def get_lr(step):
    if step < WARMUP_STEPS:
        return MAX_LR * (step + 1) / WARMUP_STEPS

    if step >= MAX_STEPS:
        return MIN_LR

    decay_ratio = (step - WARMUP_STEPS) / max(1, MAX_STEPS - WARMUP_STEPS)
    cosine = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return MIN_LR + cosine * (MAX_LR - MIN_LR)


def get_batch(data, batch_size, block_size, device):
    max_start = len(data) - block_size - 1
    if max_start < 0:
        raise ValueError("not enough tokens to create one full x/y window")

    starts = torch.randint(0, max_start + 1, (batch_size,))
    x = torch.stack([data[start : start + block_size] for start in starts])
    y = torch.stack([data[start + 1 : start + block_size + 1] for start in starts])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, device, use_amp):
    model.eval()
    losses = {}

    for split_name, data in (("train", train_data), ("val", val_data)):
        split_losses = []
        for _ in range(EVAL_ITERS):
            xb, yb = get_batch(data, BATCH_SIZE, BLOCK_SIZE, device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=use_amp,
            ):
                _, loss = model(xb, yb)
            split_losses.append(loss.item())
        losses[split_name] = sum(split_losses) / len(split_losses)

    model.train()
    return losses


@torch.no_grad()
def sample(model, prompt, merge_rules, token_to_id, id_to_token, device, use_amp):
    model.eval()
    _, prompt_ids = encode(prompt, merge_rules, token_to_id)
    generated = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    for _ in range(SAMPLE_TOKENS):
        context = generated[:, -BLOCK_SIZE:]
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_amp,
        ):
            logits, _ = model(context)

        next_logits = logits[:, -1, :] / TEMPERATURE
        probabilities = torch.softmax(next_logits, dim=-1)
        next_token_id = torch.multinomial(probabilities, num_samples=1)
        generated = torch.cat([generated, next_token_id], dim=1)

    _, text = decode(generated[0].tolist(), id_to_token)
    model.train()
    return text


def save_checkpoint(path, model, optimizer, config, step, merge_rules, vocab_tokens):
    checkpoint = {
        "step": step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": config.__dict__,
        "merge_rules": merge_rules,
        "vocab_tokens": vocab_tokens,
    }
    torch.save(checkpoint, path)


def main():
    torch.manual_seed(SEED)

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = USE_MIXED_PRECISION and device.type == "cuda"

    token_cache = load_or_build_token_cache()
    merge_rules = token_cache["merge_rules"]
    vocab_tokens = token_cache["vocab_tokens"]
    token_to_id = token_cache["token_to_id"]
    id_to_token = token_cache["id_to_token"]
    token_tensor = token_cache["token_tensor"]
    split_index = int(len(token_tensor) * TRAIN_SPLIT)
    train_data = token_tensor[:split_index]
    val_data = token_tensor[split_index:]

    print("tokenizer:")
    print(f"target vocab size: {TARGET_VOCAB_SIZE}")
    print(f"minimum pair count: {MIN_PAIR_COUNT}")
    print(f"token cache enabled: {USE_TOKEN_CACHE}")
    print(f"loaded from cache: {token_cache['loaded_from_cache']}")
    print(f"token cache path: {token_cache['cache_path']}")
    print(f"merge rules learned: {len(merge_rules):,}")
    print(f"vocab size: {len(vocab_tokens):,}")
    print(f"bpe tokens: {len(token_tensor):,}")
    print(f"compression ratio tokens/chars: {len(token_tensor) / token_cache['char_count']:.4f}")
    print(f"train tokens: {len(train_data):,}")
    print(f"val tokens: {len(val_data):,}")
    print(f"decode(encode(text)) == text: {token_cache['round_trip_ok']}")
    print()

    config = GPTConfig(
        vocab_size=len(vocab_tokens),
        block_size=BLOCK_SIZE,
        d_model=D_MODEL,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        mlp_hidden_size=MLP_HIDDEN_SIZE,
        dropout=DROPOUT,
    )
    raw_model = TinyStoriesGPT(config).to(device)
    model = raw_model
    if USE_TORCH_COMPILE:
        model = torch.compile(raw_model, mode=COMPILE_MODE)

    optimizer = torch.optim.AdamW(
        raw_model.parameters(),
        lr=MAX_LR,
        weight_decay=WEIGHT_DECAY,
    )

    parameter_count = count_parameters(raw_model)
    baseline_loss = math.log(config.vocab_size)

    print("model:")
    print(f"device: {device}")
    print(f"mixed precision requested: {USE_MIXED_PRECISION}")
    print(f"mixed precision bf16 active: {use_amp}")
    print(f"torch.compile requested: {USE_TORCH_COMPILE}")
    print(f"torch.compile active: {USE_TORCH_COMPILE}")
    print(f"torch.compile mode: {COMPILE_MODE}")
    print(f"vocab_size: {config.vocab_size}")
    print(f"block_size: {config.block_size}")
    print(f"d_model: {config.d_model}")
    print(f"num_layers: {config.num_layers}")
    print(f"num_heads: {config.num_heads}")
    print(f"mlp_hidden_size: {config.mlp_hidden_size}")
    print(f"dropout: {config.dropout}")
    print(f"parameters: {parameter_count:,}")
    print(f"tokens per parameter: {len(train_data) / parameter_count:.2f}")
    print(f"ignorance baseline ln(vocab_size): {baseline_loss:.4f}")
    print()

    initial_losses = estimate_loss(model, train_data, val_data, device, use_amp)
    print("initial loss check:")
    print(f"train loss: {initial_losses['train']:.4f}")
    print(f"val loss: {initial_losses['val']:.4f}")
    print(f"baseline: {baseline_loss:.4f}")
    print()

    with LOG_PATH.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "step",
                "train_loss",
                "val_loss",
                "lr",
                "grad_norm",
                "tokens_per_second",
                "rolling_tokens_per_second",
                "average_tokens_per_second",
                "elapsed_seconds",
            ],
        )
        writer.writeheader()

    print("training loop:")
    print(f"max_steps: {MAX_STEPS}")
    print(f"batch_size: {BATCH_SIZE}")
    print(f"gradient_accumulation_steps: {GRAD_ACCUM_STEPS}")
    print(f"effective batch tokens: {BATCH_SIZE * BLOCK_SIZE * GRAD_ACCUM_STEPS:,}")
    print(f"max_lr: {MAX_LR}")
    print(f"min_lr: {MIN_LR}")
    print(f"warmup_steps: {WARMUP_STEPS}")
    print(f"weight_decay: {WEIGHT_DECAY}")
    print(f"grad_clip: {GRAD_CLIP}")
    print()

    last_train_loss = initial_losses["train"]
    last_val_loss = initial_losses["val"]
    train_start_time = time.time()
    train_tokens_per_step = BATCH_SIZE * BLOCK_SIZE * GRAD_ACCUM_STEPS
    total_train_tokens_seen = 0
    measured_train_seconds = 0.0
    log_window_tokens = 0
    log_window_train_seconds = 0.0
    best_tokens_per_second = 0.0

    for step in range(MAX_STEPS):
        sync_if_cuda(device)
        step_start_time = time.time()
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0

        for _ in range(GRAD_ACCUM_STEPS):
            xb, yb = get_batch(train_data, BATCH_SIZE, BLOCK_SIZE, device)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=use_amp,
            ):
                logits, loss = model(xb, yb)
                loss = loss / GRAD_ACCUM_STEPS

            loss.backward()
            accumulated_loss += loss.item()

        grad_norm = nn_utils.clip_grad_norm_(model.parameters(), GRAD_CLIP).item()
        optimizer.step()

        sync_if_cuda(device)
        elapsed_this_step = time.time() - step_start_time
        tokens_per_second = train_tokens_per_step / elapsed_this_step
        total_train_tokens_seen += train_tokens_per_step
        measured_train_seconds += elapsed_this_step
        log_window_tokens += train_tokens_per_step
        log_window_train_seconds += elapsed_this_step
        average_tokens_per_second = total_train_tokens_seen / measured_train_seconds
        best_tokens_per_second = max(best_tokens_per_second, tokens_per_second)

        if step % EVAL_EVERY == 0 or step == MAX_STEPS - 1:
            losses = estimate_loss(model, train_data, val_data, device, use_amp)
            last_train_loss = losses["train"]
            last_val_loss = losses["val"]

        if step % LOG_EVERY == 0 or step == MAX_STEPS - 1:
            total_elapsed = time.time() - train_start_time
            rolling_tokens_per_second = log_window_tokens / log_window_train_seconds
            print(
                f"step {step:5d} | "
                f"loss {accumulated_loss:.4f} | "
                f"train {last_train_loss:.4f} | "
                f"val {last_val_loss:.4f} | "
                f"lr {lr:.2e} | "
                f"grad {grad_norm:.2f} | "
                f"tok/s {tokens_per_second:,.0f} | "
                f"rolling tok/s {rolling_tokens_per_second:,.0f} | "
                f"avg tok/s {average_tokens_per_second:,.0f}"
            )

            with LOG_PATH.open("a", newline="") as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=[
                        "step",
                        "train_loss",
                        "val_loss",
                        "lr",
                        "grad_norm",
                        "tokens_per_second",
                        "rolling_tokens_per_second",
                        "average_tokens_per_second",
                        "elapsed_seconds",
                    ],
                )
                writer.writerow(
                    {
                        "step": step,
                        "train_loss": last_train_loss,
                        "val_loss": last_val_loss,
                        "lr": lr,
                        "grad_norm": grad_norm,
                        "tokens_per_second": tokens_per_second,
                        "rolling_tokens_per_second": rolling_tokens_per_second,
                        "average_tokens_per_second": average_tokens_per_second,
                        "elapsed_seconds": total_elapsed,
                    }
                )

            log_window_tokens = 0
            log_window_train_seconds = 0.0

        if step % SAMPLE_EVERY == 0 or step == MAX_STEPS - 1:
            generated_text = sample(
                model,
                "Once",
                merge_rules,
                token_to_id,
                id_to_token,
                device,
                use_amp,
            )
            print("sample:")
            print(repr(generated_text))

        if (step > 0 and step % CHECKPOINT_EVERY == 0) or step == MAX_STEPS - 1:
            checkpoint_path = CHECKPOINT_DIR / f"step_{step:06d}.pt"
            save_checkpoint(
                checkpoint_path,
                raw_model,
                optimizer,
                config,
                step,
                merge_rules,
                vocab_tokens,
            )
            print(f"saved checkpoint: {checkpoint_path}")

    print()
    print("done:")
    print(f"loss log: {LOG_PATH}")
    print(f"checkpoint dir: {CHECKPOINT_DIR}")
    print(f"final train loss: {last_train_loss:.4f}")
    print(f"final val loss: {last_val_loss:.4f}")
    print(f"baseline ln(vocab_size): {baseline_loss:.4f}")
    print(f"train tokens seen: {total_train_tokens_seen:,}")
    print(f"average train-only tokens/sec: {average_tokens_per_second:,.0f}")
    print(f"best single-step train-only tokens/sec: {best_tokens_per_second:,.0f}")


if __name__ == "__main__":
    main()
