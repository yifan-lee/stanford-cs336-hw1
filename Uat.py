import time, tracemalloc
import re
import pickle

from support.bpe_tokenize import train_bpe



def save_parameter(parameter, path):
    with open(path, "wb") as f:
        pickle.dump(parameter, f)

def load_parameter(path):
    with open(path, "rb") as f:
        return pickle.load(f)



if __name__ == "__main__":

    special_tokens = ['<|endoftext|>']
    input_path = r'./data/owt_train.txt'
    vocab_size = 32000

    vocab, merges = train_bpe(input_path, vocab_size, special_tokens)
    
    match = re.search(r"./data/(.*?)\.txt", input_path)
    if match:
        data_name = match.group(1)
    else:
        data_name = ''
    vocab_path = f'./data/{data_name}_{vocab_size}_vocab.pkl'
    merges_path = f'./data/{data_name}_{vocab_size}_merges.pkl'
    
    save_parameter(vocab, vocab_path)
    save_parameter(merges, merges_path)
    