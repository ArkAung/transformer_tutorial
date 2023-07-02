import urllib.request
import torch

DEFAULT_SOURCE = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DEFAULT_FILENAME = "input.txt"

stoi, itos = {}, {}

encode = lambda s: [
    stoi[c] for c in s
]  # encoder: take a string, output a list of integers
decode = lambda l: "".join(
    [itos[i] for i in l]
)  # decoder: take a list of integers, output a string


def download_file(url, filename):
    try:
        urllib.request.urlretrieve(url, filename)
        print("Download successful!")
    except Exception as e:
        print(f"Download failed: {e}")


def get_training_corpus(source=DEFAULT_SOURCE, output_filename=DEFAULT_FILENAME):
    global stoi, itos
    download_file(source, output_filename)
    with open(output_filename, "r", encoding="utf-8") as f:
        text = f.read()
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    return text, vocab_size


def train_val_split(corpus, split_ratio=0.8):
    # Train and test splits
    data = torch.tensor(encode(corpus), dtype=torch.long)
    n = int(split_ratio * len(data))
    train_data = data[:n]
    val_data = data[n:]
    return train_data, val_data


def get_batch(split, device, train_data, val_data, block_size, batch_size):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(
    model, eval_iters, device, train_data, val_data, block_size, batch_size
):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(
                split, device, train_data, val_data, block_size, batch_size
            )
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
