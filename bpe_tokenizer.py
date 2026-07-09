from collections import Counter


corpus = "low lower lowest low lower"
target_vocab_size = 256
min_pair_count = 2


def unique_in_order(items):
    seen = set()
    ordered = []
    for item in items:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def count_pairs(tokens):
    pairs = Counter()
    for left, right in zip(tokens, tokens[1:]):
        pairs[(left, right)] += 1
    return pairs


def merge_pair(tokens, pair_to_merge):
    merged_tokens = []
    i = 0

    while i < len(tokens):
        current_pair = tokens[i], tokens[i + 1] if i + 1 < len(tokens) else None

        if current_pair == pair_to_merge:
            merged_tokens.append(tokens[i] + tokens[i + 1])
            i += 2
        else:
            merged_tokens.append(tokens[i])
            i += 1

    return merged_tokens


def learn_one_merge(tokens, min_count):
    pair_counts = count_pairs(tokens)
    if not pair_counts:
        return tokens, None

    best_pair, best_count = pair_counts.most_common(1)[0]
    if best_count < min_count:
        return tokens, None

    new_token = best_pair[0] + best_pair[1]
    tokens = merge_pair(tokens, best_pair)
    merge_rule = {
        "pair": best_pair,
        "new_token": new_token,
        "count": best_count,
    }
    return tokens, merge_rule


def build_vocab(initial_tokens, merge_rules):
    vocab_tokens = unique_in_order(initial_tokens)
    for merge_rule in merge_rules:
        vocab_tokens.append(merge_rule["new_token"])

    token_to_id = {token: token_id for token_id, token in enumerate(vocab_tokens)}
    id_to_token = {token_id: token for token, token_id in token_to_id.items()}
    return vocab_tokens, token_to_id, id_to_token


def encode(text, merge_rules, token_to_id):
    tokens = list(text)

    for merge_rule in merge_rules:
        tokens = merge_pair(tokens, merge_rule["pair"])

    token_ids = [token_to_id[token] for token in tokens]
    return tokens, token_ids


def decode(token_ids, id_to_token):
    tokens = [id_to_token[token_id] for token_id in token_ids]
    text = "".join(tokens)
    return tokens, text


def print_check(name, passed):
    status = "PASS" if passed else "FAIL"
    print(f"{status}: {name}")


tokens = list(corpus)
initial_tokens = tokens.copy()
vocab = set(tokens)
merge_rules = []

print(f"corpus: {corpus!r}")
print(f"target vocab size: {target_vocab_size}")
print(f"minimum pair count: {min_pair_count}")
print()
print("initial character tokens:")
print(tokens)
print(f"initial vocab size: {len(vocab)}")
print(f"initial vocab: {sorted(vocab)!r}")

while len(vocab) < target_vocab_size and len(tokens) > 1:
    # 1. Count adjacent pairs.
    # 2. Pick the most common pair.
    # 3. Merge that pair everywhere it appears.
    # 4. Save the merge rule so the tokenizer can reuse it later.
    tokens, merge_rule = learn_one_merge(tokens, min_pair_count)
    if merge_rule is None:
        break

    best_pair = merge_rule["pair"]
    new_token = merge_rule["new_token"]
    best_count = merge_rule["count"]
    vocab.add(new_token)
    merge_rules.append(merge_rule)

    print()
    print(f"merge {len(merge_rules)}")
    print(f"most common pair: {best_pair!r}, count: {best_count}")
    print(f"new token: {new_token!r}")
    print("tokens after merge:")
    print(tokens)
    print(f"vocab size: {len(vocab)}")

print()
print("final tokens:")
print(tokens)
print()
print("learned merges:")
for merge_rule in merge_rules:
    print(
        f"{merge_rule['pair']!r} -> {merge_rule['new_token']!r} "
        f"(count {merge_rule['count']})"
    )

vocab_tokens, token_to_id, id_to_token = build_vocab(initial_tokens, merge_rules)

print()
print("vocab tokens:")
for token in vocab_tokens:
    print(f"{token!r} -> {token_to_id[token]}")

new_text = "lower"
encoded_tokens, encoded_ids = encode(new_text, merge_rules, token_to_id)

print()
print(f"encode new text: {new_text!r}")
print("tokens after replaying learned merge rules:")
print(encoded_tokens)
print("token ids:")
print(encoded_ids)

decoded_tokens, decoded_text = decode(encoded_ids, id_to_token)

print()
print(f"decode token ids: {encoded_ids}")
print("decoded tokens:")
print(decoded_tokens)
print("decoded text:")
print(decoded_text)

example_ids = [4, 5]
example_tokens, example_text = decode(example_ids, id_to_token)

print()
print(f"decode token ids: {example_ids}")
print("decoded tokens:")
print(example_tokens)
print("decoded text:")
print(example_text)

print()
print("oracle checks:")

training_created_merges = len(merge_rules) > 0
encoding_compressed_text = len(encoded_ids) < len(new_text)
round_trip_matches = decoded_text == new_text

print_check("training creates merge rules", training_created_merges)
print_check("encoding turns text into fewer tokens than characters", encoding_compressed_text)
print_check("decode(encode(text)) == text", round_trip_matches)

assert training_created_merges
assert encoding_compressed_text
assert round_trip_matches
