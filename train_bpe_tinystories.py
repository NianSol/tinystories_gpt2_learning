from collections import Counter
from pathlib import Path
import re


TRAIN_PATH = Path("data/tinystories/TinyStories-train.txt")
TRAIN_FRACTION = 0.01
TARGET_VOCAB_SIZE = 256
MIN_PAIR_COUNT = 2
CHECK_CHARS = 100_000


def unique_in_order(items):
    seen = set()
    ordered = []
    for item in items:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def read_fraction(path, fraction):
    file_size = path.stat().st_size
    byte_count = int(file_size * fraction)

    with path.open("rb") as file:
        raw = file.read(byte_count)

    text = raw.decode("utf-8", errors="ignore")
    return text, file_size, byte_count


def split_chunks(text):
    return re.findall(r"\S+|\s+", text)


def count_pairs(sequence_counts):
    pair_counts = Counter()

    for sequence, sequence_count in sequence_counts.items():
        for left, right in zip(sequence, sequence[1:]):
            pair_counts[(left, right)] += sequence_count

    return pair_counts


def merge_sequence(sequence, pair_to_merge):
    merged = []
    i = 0

    while i < len(sequence):
        current_pair = (
            sequence[i],
            sequence[i + 1] if i + 1 < len(sequence) else None,
        )

        if current_pair == pair_to_merge:
            merged.append(sequence[i] + sequence[i + 1])
            i += 2
        else:
            merged.append(sequence[i])
            i += 1

    return tuple(merged)


def merge_all_sequences(sequence_counts, pair_to_merge):
    merged_counts = Counter()

    for sequence, sequence_count in sequence_counts.items():
        merged_sequence = merge_sequence(sequence, pair_to_merge)
        merged_counts[merged_sequence] += sequence_count

    return merged_counts


def train_bpe(text, target_vocab_size, min_pair_count, verbose=True):
    initial_tokens = list(text)
    vocab = set(initial_tokens)
    chunks = split_chunks(text)
    sequence_counts = Counter(tuple(chunk) for chunk in chunks)
    merge_rules = []

    if verbose:
        print(f"total chunks: {len(chunks):,}")
        print(f"unique chunks: {len(sequence_counts):,}")
        print(f"initial vocab size: {len(vocab)}")
        print()

    while len(vocab) < target_vocab_size:
        pair_counts = count_pairs(sequence_counts)
        if not pair_counts:
            break

        best_pair, best_count = pair_counts.most_common(1)[0]
        if best_count < min_pair_count:
            break

        new_token = best_pair[0] + best_pair[1]
        sequence_counts = merge_all_sequences(sequence_counts, best_pair)
        vocab.add(new_token)
        merge_rules.append(
            {
                "pair": best_pair,
                "new_token": new_token,
                "count": best_count,
            }
        )

        if verbose:
            print(
                f"merge {len(merge_rules):03d} | "
                f"count {best_count:>8,} | "
                f"{best_pair!r} -> {new_token!r} | "
                f"vocab {len(vocab)}"
            )

    vocab_tokens = unique_in_order(initial_tokens)
    vocab_tokens.extend(rule["new_token"] for rule in merge_rules)
    token_to_id = {token: token_id for token_id, token in enumerate(vocab_tokens)}
    id_to_token = {token_id: token for token, token_id in token_to_id.items()}

    return merge_rules, vocab_tokens, token_to_id, id_to_token


def encode(text, merge_rules, token_to_id):
    encoded_tokens = []
    chunk_cache = {}

    for chunk in split_chunks(text):
        if chunk not in chunk_cache:
            tokens = tuple(chunk)
            for merge_rule in merge_rules:
                tokens = merge_sequence(tokens, merge_rule["pair"])
            chunk_cache[chunk] = tokens

        tokens = chunk_cache[chunk]
        encoded_tokens.extend(tokens)

    token_ids = [token_to_id[token] for token in encoded_tokens]
    return encoded_tokens, token_ids


def decode(token_ids, id_to_token):
    tokens = [id_to_token[token_id] for token_id in token_ids]
    text = "".join(tokens)
    return tokens, text


def print_check(name, passed):
    status = "PASS" if passed else "FAIL"
    print(f"{status}: {name}")


def main():
    text, file_size, byte_count = read_fraction(TRAIN_PATH, TRAIN_FRACTION)

    print(f"training file: {TRAIN_PATH}")
    print(f"file size bytes: {file_size:,}")
    print(f"training fraction: {TRAIN_FRACTION:.2%}")
    print(f"bytes read: {byte_count:,}")
    print(f"decoded characters: {len(text):,}")
    print(f"target vocab size: {TARGET_VOCAB_SIZE}")
    print(f"minimum pair count: {MIN_PAIR_COUNT}")
    print()

    merge_rules, vocab_tokens, token_to_id, id_to_token = train_bpe(
        text,
        TARGET_VOCAB_SIZE,
        MIN_PAIR_COUNT,
    )

    check_text = text[:CHECK_CHARS]
    encoded_tokens, encoded_ids = encode(check_text, merge_rules, token_to_id)
    decoded_tokens, decoded_text = decode(encoded_ids, id_to_token)

    print()
    print("training summary:")
    print(f"merge rules learned: {len(merge_rules)}")
    print(f"final vocab size: {len(vocab_tokens)}")
    print(f"first 25 vocab tokens: {vocab_tokens[:25]!r}")
    print(f"last 25 vocab tokens: {vocab_tokens[-25:]!r}")
    print()

    print("encoding check:")
    print(f"original characters checked: {len(check_text):,}")
    print(f"encoded token count: {len(encoded_ids):,}")
    print(f"compression ratio tokens/chars: {len(encoded_ids) / len(check_text):.4f}")
    print("first 80 encoded tokens:")
    print(encoded_tokens[:80])
    print("first 80 token ids:")
    print(encoded_ids[:80])
    print()

    print("oracle checks:")
    training_created_merges = len(merge_rules) > 0
    encoding_compressed_text = len(encoded_ids) < len(check_text)
    round_trip_matches = decoded_text == check_text

    print_check("training creates merge rules", training_created_merges)
    print_check("encoding turns text into fewer tokens than characters", encoding_compressed_text)
    print_check("decode(encode(text)) == text", round_trip_matches)

    assert training_created_merges
    assert encoding_compressed_text
    assert round_trip_matches


if __name__ == "__main__":
    main()
