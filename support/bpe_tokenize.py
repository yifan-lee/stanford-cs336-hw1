import pandas as pd
from collections import defaultdict
import regex as re
from multiprocessing import Pool
from support.find_chunk_boundaries import find_chunk_boundaries
from memory_profiler import profile
import time, tracemalloc
from dataclasses import dataclass


Pair = tuple[int,int]
Encoded_Token = tuple[int, ...]
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""



@dataclass(frozen=True)
class TokenMergePlan():
    old_token: Encoded_Token
    new_token: Encoded_Token
    count: int
    pair_positions: list[int]

class PairFreqsDelta():
    inc: defaultdict[Pair, int]
    inc: defaultdict[Pair, int]
    def __init__(self):
        self.inc = defaultdict(int)
        self.dec = defaultdict(int)

class PairInhereitDelta():
    add: defaultdict[Pair,set[Encoded_Token]]
    remove: defaultdict[Pair,set[Encoded_Token]]
    def __init__(self):
        self.add = defaultdict(set[Encoded_Token])
        self.remove = defaultdict(set[Encoded_Token])


def pre_tokenize(string: str,special_tokens: list[str]) -> dict[str, int]:
    string_list = split_by_special_tokens(string, special_tokens)
    token_freqs = count_tokens(string_list)
    return token_freqs

def tokenize_chunk(args):
    chunk_text, special_tokens = args
    return pre_tokenize(chunk_text, special_tokens)

def read_text_chunks(input_path: str, special_tokens: list[str], num_processes: int = 16) -> list[str]:
    with open(input_path, "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        chunks = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            chunks.append((chunk, special_tokens))
    return chunks


def combine_counts(results: list[dict[str,int]]) -> dict[str,int]:
    token_freqs = defaultdict(int)
    for partial_count in results:
        for word, count in partial_count.items():
            token_freqs[word] += count
    return token_freqs

def read_text_file(input_path: str) -> str:
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    return text

def split_by_special_tokens(string: str, special_tokens: list[str]) -> list[str]:
    pattern = "|".join(re.escape(tok) for tok in special_tokens)
    return re.split(pattern,string)

def count_tokens(string_list: list[str]) -> dict[str, int]:
    counts = defaultdict(int)
    for s in string_list:
        tokens = re.finditer(PAT, s)
        for m in tokens:
            tok = m.group(0)
            counts[tok] += 1
    return counts

def encode_and_count_tokens(counts: dict[str, int])-> dict[Encoded_Token, int]:
    encoded_token_freqs = defaultdict(int)
    for token, count in counts.items():
        elements = tuple(token.encode("utf-8"))
        encoded_token_freqs[elements] += count
    return encoded_token_freqs

def build_initial_vocab(special_tokens: list[str]) ->  dict[int, bytes]:
    vocab = {i:bytes([i]) for i in range(256)}
    for i, tok in enumerate(special_tokens, start=256):
        vocab[i] = tok.encode("utf-8")
    return vocab

def get_byte_pairs(encoded_token_freqs: dict[Encoded_Token, int]) -> dict[Pair, int]:
    pair_freqs = defaultdict(int)
    for tok, count in encoded_token_freqs.items():
        for i in range(len(tok)-1):
            pair_freqs[(tok[i],tok[i+1])] += count
    return pair_freqs

def get_byte_pairs_inhereit(encoded_token_freqs: dict[Encoded_Token, int]) -> dict[Pair, set[Encoded_Token]]:
    pair_inhereit = defaultdict(set)
    for tok, count in encoded_token_freqs.items():
        for i in range(len(tok)-1):
            pair_inhereit[(tok[i],tok[i+1])].add(tok)
    return pair_inhereit

def select_merge_pair(pair_freqs: dict[Pair, int], vocab:dict[int, bytes]) -> Pair:
    max_count = max(pair_freqs.values())
    candidate_pairs = [key for key, value in pair_freqs.items() if value == max_count]
    def sort_pair(pair):
        index1, index2 = pair
        return(vocab[index1], vocab[index2])
    pair = max(candidate_pairs, key = sort_pair)
    return pair

def update_encoded_token(encoded_token: Encoded_Token, pair: Pair, new_index: int) -> Encoded_Token:
    result = []
    i = 0
    while i < len(encoded_token):
        if i < len(encoded_token) - 1 and (encoded_token[i], encoded_token[i + 1]) == pair:
            result.append(new_index)
            i += 2
        else:
            result.append(encoded_token[i])
            i += 1
    return tuple(result)

def find_subtuple_index(sequence: tuple, subseq: tuple) -> list[int]:
    position = []
    subseq_len = len(subseq)
    for i in range(len(sequence)-subseq_len+1):
        if sequence[i:i+subseq_len] == subseq:
            position.append(i)
    return position

def remove_or_decrement_pair(pair_freqs: dict[Pair, int], pair: Pair, count: int) -> dict[Pair, int]:
    updated_pair_freqs = pair_freqs.copy()
    if updated_pair_freqs[pair] == count:
        del updated_pair_freqs[pair]
    else:
        updated_pair_freqs[pair] -= count
    return updated_pair_freqs

def build_merge_plan(tok_need_update: set[Encoded_Token], encoded_token_freqs: dict[Encoded_Token, int], pair: Pair, new_index: int) -> list[TokenMergePlan]:
    plan = []
    for encoded_token in tok_need_update:
        new_encoded_token = update_encoded_token(encoded_token, pair, new_index)
        count = encoded_token_freqs[encoded_token]
        pair_positions = find_subtuple_index(encoded_token,pair)
        plan.append(TokenMergePlan(encoded_token,new_encoded_token,count,pair_positions))
    return plan

def update_encoded_token_freqs(plan: list[TokenMergePlan], encoded_token_freqs: dict[Encoded_Token, int]) ->  dict[Encoded_Token, int]:
    new_encoded_token_freqs = encoded_token_freqs.copy()
    for item in plan:
        del new_encoded_token_freqs[item.old_token]
        new_encoded_token_freqs[item.new_token] = item.count
    return new_encoded_token_freqs

def compute_freqs_deltas(plan: list[TokenMergePlan], new_index:int) -> PairFreqsDelta:
    pair_freqs_d = PairFreqsDelta()
    for item in plan:
        old_token = item.old_token
        count = item.count
        for pos in item.pair_positions:
            if pos > 0:
                pre_token = old_token[pos-1]
                old_pair = (pre_token,old_token[pos])
                new_pair = (pre_token, new_index)
                pair_freqs_d.dec[old_pair] += count
                pair_freqs_d.inc[new_pair] += count
            if pos < len(old_token)-2:
                pos_token = old_token[pos+2]
                old_pair = (old_token[pos+1],pos_token)
                new_pair = (new_index, pos_token)
                pair_freqs_d.dec[old_pair] += count
                pair_freqs_d.inc[new_pair] += count
    return pair_freqs_d

def compute_inhereit_deltas(plan: list[TokenMergePlan]) -> PairInhereitDelta:
    pair_inhereit_d = PairInhereitDelta()
    for item in plan:
        if len(item.old_token) > 1:
            for old_pair in zip(item.old_token,item.old_token[1:]):
                pair_inhereit_d.remove[old_pair].add(item.old_token)
        if len(item.new_token) > 1:
            for new_pair in zip(item.new_token,item.new_token[1:]):
                pair_inhereit_d.add[new_pair].add(item.new_token)
    return pair_inhereit_d 

def exclude_pair_from_dict(d: dict, pair: Pair):
    new_d = d.copy()
    del new_d[pair]
    return new_d

def update_pair_freqs(pair_freqs: dict[Pair, int], pair_freqs_d: PairFreqsDelta):
    new_pair_freqs = pair_freqs.copy()
    for key, value in pair_freqs_d.dec.items():
        new_pair_freqs = remove_or_decrement_pair(new_pair_freqs, key, value)
    for key, value in pair_freqs_d.inc.items():
        new_pair_freqs[key]+=value
    return new_pair_freqs

def update_pair_inhereit(pair_inhereit: dict[Pair, set[Encoded_Token]], pair_inhereit_d: PairInhereitDelta):
    new_pair_inhereit = pair_inhereit.copy()
    for key, value in pair_inhereit_d.remove.items():
        new_pair_inhereit[key] -= value
    for key, value in pair_inhereit_d.add.items():
        new_pair_inhereit[key] = new_pair_inhereit[key] | value
    return new_pair_inhereit

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    
    tracemalloc.start()
    total_start_time = time.perf_counter()
    steps = []
    num_processes = 16
    
    @timed("📖 加载文本并切 chunk", steps)
    def step1():
        return read_text_chunks(input_path, special_tokens, num_processes)
    
    @timed("🧮 多进程分词统计", steps)
    def step2(chunks):
        with Pool(num_processes) as pool:
            return pool.map(tokenize_chunk, chunks)
    
    @timed("📦 合并词频", steps)
    def step3(results):
        token_freqs = combine_counts(results)
        return token_freqs
    
    @timed("🔤 构建 byte 级统计", steps)
    def step4(token_freqs):
        return encode_and_count_tokens(token_freqs)
    
    @timed("🧠 BPE 合并主循环", steps)
    def step5(encoded_token_freqs):
        vocab = build_initial_vocab(special_tokens)
        vocab_len = len(vocab)
        merges = []
        
        pair_freqs = get_byte_pairs(encoded_token_freqs)
        pair_inhereit = get_byte_pairs_inhereit(encoded_token_freqs)
        
        total_get_pair_time = 0.0
        total_get_plan_time = 0.0
        total_update_encoded_token_freqs_time = 0.0
        total_get_pair_freqs_delta_time = 0.0
        total_update_pair_freqs_time = 0.0
        total_update_pair_inhereit_time = 0.0
        total_get_pair_inhereit_delta_time = 0.0
        while vocab_len < vocab_size:
            t0 = time.perf_counter()
            if not pair_freqs:
                break
            pair = select_merge_pair(pair_freqs, vocab)
            index1, index2 = pair
            new_token = vocab[int(index1)]+vocab[int(index2)]
            new_index = vocab_len
            vocab[new_index] = new_token
            merges.append((vocab[int(index1)], vocab[int(index2)]))
            tok_need_update = pair_inhereit[pair]
            t1 = time.perf_counter()
            plan = build_merge_plan(tok_need_update, encoded_token_freqs, pair, new_index)
            t2 = time.perf_counter()
            encoded_token_freqs = update_encoded_token_freqs(plan, encoded_token_freqs)
            t3 = time.perf_counter()
            pair_freqs_d = compute_freqs_deltas(plan, new_index)
            t4 = time.perf_counter()
            pair_freqs = exclude_pair_from_dict(pair_freqs, pair)
            pair_freqs = update_pair_freqs(pair_freqs, pair_freqs_d)
            t5 = time.perf_counter()
            pair_inhereit_d = compute_inhereit_deltas(plan)
            t6 = time.perf_counter()
            pair_inhereit = exclude_pair_from_dict(pair_inhereit, pair)
            pair_inhereit = update_pair_inhereit(pair_inhereit, pair_inhereit_d)
            vocab_len+=1
            t7 = time.perf_counter()
            total_get_pair_time += t1 - t0
            total_get_plan_time += t2 - t1
            total_update_encoded_token_freqs_time += t3 - t2
            total_get_pair_freqs_delta_time += t4 - t3
            total_update_pair_freqs_time += t5 - t4
            total_update_pair_inhereit_time += t6 - t5
            total_get_pair_inhereit_delta_time += t7 - t6
        print(f"\n⏱️ 总计 count_byte_pairs 时间: {total_get_pair_time:.2f}s")
        print(f"⏱️ 总计 get_plan 时间: {total_get_plan_time:.2f}s")
        print(f"⏱️ 总计 update_encoded_token_freqs 时间: {total_update_encoded_token_freqs_time:.2f}s")
        print(f"⏱️ 总计 get_pair_freqs_delta 时间: {total_get_pair_freqs_delta_time:.2f}s")
        print(f"⏱️ 总计 update_pair_freqs 时间: {total_update_pair_freqs_time:.2f}s")
        print(f"⏱️ 总计 update_pair_inhereit 时间: {total_update_pair_inhereit_time:.2f}s")
        print(f"⏱️ 总计 et_pair_inhereit_delta 时间: {total_get_pair_inhereit_delta_time:.2f}s")
        return vocab, merges
    
    chunks = step1()
    results = step2(chunks)
    token_freqs = step3(results)
    encoded_token_freqs = step4(token_freqs)
    vocab, merges = step5(encoded_token_freqs)
    
    total_end_time = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    

    print("\n✅ 各阶段耗时与内存增量统计:")
    for name, t, mem in steps:
        print(f"{name:<20} ⏱️ {t:.2f}s | 📦 {mem:.2f}MB")
        
    print(f"\n✅ 总运行时间: {total_end_time - total_start_time:.2f}s")
    print(f"📦 当前内存使用: {current / 1024 / 1024 / 1024:.2f} GB")
    print(f"📦 峰值内存使用: {peak / 1024 / 1024 / 1024:.2f} GB")


    return vocab, merges

def timed(name, snapshot_list):
    def wrapper(func):
        def inner(*args, **kwargs):
            print(f"\n--- 开始 {name} ---")
            t0 = time.perf_counter()
            snapshot_before = tracemalloc.take_snapshot()

            result = func(*args, **kwargs)

            t1 = time.perf_counter()
            snapshot_after = tracemalloc.take_snapshot()
            stats = snapshot_after.compare_to(snapshot_before, 'lineno')
            total_mem = sum(stat.size_diff for stat in stats) / 1024 / 1024 / 1024

            print(f"{name} ⏱️耗时: {t1 - t0:.2f}s | 📦内存增长: {total_mem:.2f}GB")
            snapshot_list.append((name, t1 - t0, total_mem))
            return result
        return inner
    return wrapper