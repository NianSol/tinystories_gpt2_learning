import random

from train_bpe_tinystories import (
    MIN_PAIR_COUNT,
    TARGET_VOCAB_SIZE,
    TRAIN_FRACTION,
    TRAIN_PATH,
    encode,
    print_check,
    read_fraction,
    train_bpe,
)


BLOCK_SIZE = 32
BATCH_SIZE = 4
RANDOM_SEED = 123


def decode_text(token_ids, id_to_token):
    return "".join(id_to_token[token_id] for token_id in token_ids)


def get_batch(token_ids, batch_size, block_size, rng):
    max_start = len(token_ids) - block_size - 1
    if max_start < 0:
        raise ValueError("not enough tokens to create one full x/y window")

    starts = [rng.randint(0, max_start) for _ in range(batch_size)]
    x = [token_ids[start : start + block_size] for start in starts]
    y = [token_ids[start + 1 : start + block_size + 1] for start in starts]
    return x, y, starts


text, file_size, byte_count = read_fraction(TRAIN_PATH, TRAIN_FRACTION)

print(f"training file: {TRAIN_PATH}")
print(f"file size bytes: {file_size:,}")
print(f"training fraction: {TRAIN_FRACTION:.2%}")
print(f"bytes read: {byte_count:,}")
print(f"decoded characters: {len(text):,}")
print(f"target vocab size: {TARGET_VOCAB_SIZE}")
print(f"minimum pair count: {MIN_PAIR_COUNT}")
print(f"block size: {BLOCK_SIZE}")
print(f"batch size: {BATCH_SIZE}")
print()

merge_rules, vocab_tokens, token_to_id, id_to_token = train_bpe(
    text,
    TARGET_VOCAB_SIZE,
    MIN_PAIR_COUNT,
)

encoded_tokens, token_ids = encode(text, merge_rules, token_to_id)

print()
print("token stream:")
print(f"raw characters: {len(text):,}")
print(f"bpe token ids: {len(token_ids):,}")
print(f"compression ratio tokens/chars: {len(token_ids) / len(text):.4f}")
print(f"vocab size: {len(vocab_tokens):,}")
print("first 60 tokens:")
print(encoded_tokens[:60])
print("first 60 token ids:")
print(token_ids[:60])
print()

rng = random.Random(RANDOM_SEED)
x, y, starts = get_batch(token_ids, BATCH_SIZE, BLOCK_SIZE, rng)

print("training batch:")
print(f"x shape: ({len(x)}, {len(x[0])})")
print(f"y shape: ({len(y)}, {len(y[0])})")
print(f"batch starts, measured in token positions: {starts}")
print()

print("first row of x:")
print(x[0])
print("first row of y:")
print(y[0])
print()

print("decoded first x row:")
print(repr(decode_text(x[0], id_to_token)))
print("decoded first y row:")
print(repr(decode_text(y[0], id_to_token)))
print()

round_trip_text = decode_text(token_ids, id_to_token)
shift_is_correct = all(
    x_row[1:] == y_row[:-1]
    for x_row, y_row in zip(x, y)
)

print("oracle checks:")
print_check("training creates merge rules", len(merge_rules) > 0)
print_check("encoding turns text into fewer tokens than characters", len(token_ids) < len(text))
print_check("decode(encode(text)) == text", round_trip_text == text)
print_check("x/y batches have the requested shape", len(x) == BATCH_SIZE and len(x[0]) == BLOCK_SIZE and len(y[0]) == BLOCK_SIZE)
print_check("y is x shifted one BPE token forward", shift_is_correct)

assert len(merge_rules) > 0
assert len(token_ids) < len(text)
assert round_trip_text == text
assert len(x) == BATCH_SIZE
assert len(x[0]) == BLOCK_SIZE
assert len(y[0]) == BLOCK_SIZE
assert shift_is_correct
