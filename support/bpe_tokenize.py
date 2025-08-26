from collections import defaultdict
import regex as re
from multiprocessing import Pool
from support.find_chunk_boundaries import find_chunk_boundaries
# from memory_profiler import profile
import time, tracemalloc


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

def get_pair_counts(element_counts: dict[str, int]) -> dict[tuple[int,int], int]:
    pair_counts = defaultdict(int)
    for elements, count in element_counts.items():
        for i in range(len(elements)-1):
            pair_counts[(elements[i],elements[i+1])] += count
    return pair_counts


def update_element_counts(byte_level_counts: dict[str, int], pair: tuple[int, int], new_index: int) -> dict[str, int]:
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

def find_max_pair(pair_counts: dict[tuple[int,int], int], vocab:dict[int, bytes]) -> tuple[int, int]:
    max_count = max(pair_counts.values())
    candidate_pairs = [key for key, value in pair_counts.items() if value == max_count]
    def sort_pair(pair):
        index1, index2 = pair
        return(vocab[index1], vocab[index2])
    pair = max(candidate_pairs, key = sort_pair)
    return pair


def pre_tokenize(string: str,special_tokens: list[str]) -> dict[str, int]:
    string_list = split_string(string, special_tokens)
    word_level_counts = get_tok_counts(string_list)
    return word_level_counts

def process_chunk(args):
    chunk_text, special_tokens = args
    return pre_tokenize(chunk_text, special_tokens)

def load_txt_in_chunks(input_path: str, special_tokens: list[str], num_processes: int = 16) -> list[str]:
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
    word_level_counts = defaultdict(int)
    for partial_count in results:
        for word, count in partial_count.items():
            word_level_counts[word] += count
    return word_level_counts

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    
    tracemalloc.start()
    total_start_time = time.perf_counter()
    steps = []
    num_processes = 16
    
    @timed("📖 加载文本并切 chunk", steps)
    def step1():
        return load_txt_in_chunks(input_path, special_tokens, num_processes)
    
    @timed("🧮 多进程分词统计", steps)
    def step2(chunks):
        with Pool(num_processes) as pool:
            return pool.map(process_chunk, chunks)
    
    @timed("📦 合并词频", steps)
    def step3(results):
        word_level_counts = combine_counts(results)
        return word_level_counts
    
    @timed("🔤 构建 byte 级统计", steps)
    def step4(word_counts):
        return get_byte_counts(word_counts)
    
    @timed("🧠 BPE 合并主循环", steps)
    def step5(byte_counts):
        vocab = initiate_vocab(special_tokens)
        vocab_len = len(vocab)
        merges = []
        while vocab_len < vocab_size:
            pair_counts = get_pair_counts(byte_counts)
            if not pair_counts:
                break
            pair = find_max_pair(pair_counts, vocab)
            index1, index2 = pair
            new_token = vocab[index1] + vocab[index2]
            new_index = vocab_len
            byte_counts = update_element_counts(byte_counts, pair, new_index)
            merges.append((vocab[index1], vocab[index2]))
            vocab[new_index] = new_token
            vocab_len += 1
        return vocab, merges
    
    chunks = step1()
    results = step2(chunks)
    word_level_counts = step3(results)
    byte_level_counts = step4(word_level_counts)
    vocab, merges = step5(byte_level_counts)
    
    total_end_time = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    

    print("\n✅ 各阶段耗时与内存增量统计:")
    for name, t, mem in steps:
        print(f"{name:<20} ⏱️ {t:.2f}s | 📦 {mem:.2f}MB")
        
    print(f"\n✅ 总运行时间: {total_end_time - total_start_time:.2f}s")
    print(f"📦 当前内存使用: {current / 1024 / 1024:.2f} MB")
    print(f"📦 峰值内存使用: {peak / 1024 / 1024:.2f} MB")


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
            total_mem = sum(stat.size_diff for stat in stats) / 1024 / 1024

            print(f"{name} ⏱️耗时: {t1 - t0:.2f}s | 📦内存增长: {total_mem:.2f}MB")
            snapshot_list.append((name, t1 - t0, total_mem))
            return result
        return inner
    return wrapper