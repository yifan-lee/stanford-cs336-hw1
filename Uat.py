from collections import defaultdict
import regex as re

def load_txt_as_str(input_path: str) -> str:
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    return text

def split_string(string: str, special_tokens: list[str]) -> list[str]:
    pattern = "|".join(re.escape(tok) for tok in special_tokens)
    return re.split(pattern,string)

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
def get_tok_counts(string_list: list[str]) -> dict[str, int]:
    counts = defaultdict(int)
    for s in string_list:
        tokens = re.finditer(PAT, s)
        for m in tokens:
            tok = m.group(0)
            counts[tok] += 1
    return counts

def get_byte_counts(counts: dict[str, int])-> dict[str, int]:
    element_counts = defaultdict(int)
    for token, count in counts.items():
        elements = tuple(token.encode("utf-8"))
        element_counts[elements] += count
    return element_counts

def get_pair_counts(element_counts: dict[str, int]) -> dict[tuple[str,str], int]:
    pair_counts = defaultdict(int)
    for elements, count in element_counts.items():
        for i in range(len(elements)-1):
            pair_counts[(elements[i],elements[i+1])] += count
    return pair_counts


def update_element_counts(byte_level_counts: dict[str, int], pair: tuple[str, str], new_index: int) -> dict[str, int]:
    new_byte_level_counts = {}
    for elements, counts in byte_level_counts.items():
        new_element = []
        elements_len = len(elements)
        index = 0
        while index <= elements_len-1:
            if (index < elements_len-1) and (elements[index] == pair[0]) and (elements[index+1] == pair[1]):
                new_element.append(new_index)
                index += 2
            else:
                new_element.append(elements[index])
                index += 1
        new_byte_level_counts[tuple(new_element)] = counts
    return new_byte_level_counts  

def initiate_vocab(special_tokens: list[str]) ->  dict[int, bytes]:
    vocab = {i:bytes([i]) for i in range(256)}
    for i, tok in enumerate(special_tokens, start=256):
        vocab[i] = tok.encode("utf-8")
    return vocab   


def pre_tokenize(string: str,vocab_size: int,special_tokens: list[str]) -> tuple[dict[int, bytes],list[tuple[bytes, bytes]]]:
    merges = []
    string_list = split_string(string, special_tokens)
    word_level_counts = get_tok_counts(string_list)
    byte_level_counts = get_byte_counts(word_level_counts)
    vocab = initiate_vocab(special_tokens)
    vocab_len = len(vocab)

    while vocab_len<vocab_size:
        pair_counts = get_pair_counts(byte_level_counts)
        if len(pair_counts) == 0:
            break
        pair = max(pair_counts, key=lambda k: (pair_counts[k], k))
        index1, index2 = pair
        new_token = vocab[int(index1)]+vocab[int(index2)]
        new_index = vocab_len
        byte_level_counts = update_element_counts(byte_level_counts, pair,new_index)
        merges.append((vocab[int(index1)], vocab[int(index2)]))
        vocab[new_index] = new_token
        vocab_len+=1
    return vocab, merges

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    string = load_txt_as_str(input_path)
    vocab, merges = pre_tokenize(string, vocab_size,special_tokens)
    return vocab, merges