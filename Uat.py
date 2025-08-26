import time, tracemalloc

from support.bpe_tokenize import train_bpe





if __name__ == "__main__":

    special_tokens = ['<|endoftext|>']
    input_path = r'./data/TinyStoriesV2-GPT4-train.txt'
    vocab_size = 1000

    vocab, merges = train_bpe(input_path, vocab_size, special_tokens)
    