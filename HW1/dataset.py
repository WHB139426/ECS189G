import torch
import urllib.request
import os

def load_tiny_shakespeare():
    """Download Tiny Shakespeare dataset"""
    data_path = 'input.txt'
    if not os.path.exists(data_path):
        print("Downloading Tiny Shakespeare dataset...")
        url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        urllib.request.urlretrieve(url, data_path)

    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Tokenization
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])

    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data)) # 90% train, 10% validation
    train_data = data[:n]
    val_data = data[n:]

    return train_data, val_data, vocab_size, encode, decode

def get_batch(data, context_length, batch_size, device):
    """generate a batch of data"""
    ix = torch.randint(len(data) - context_length, (batch_size,))
    x = torch.stack([data[i:i+context_length] for i in ix])
    y = torch.stack([data[i+1:i+context_length+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y