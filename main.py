import torch

from utils import estimate_loss, get_batch, get_training_corpus, decode, train_val_split
from language_model import SimpleLanguageModel

if __name__ == "__main__":
    # hyperparameters
    batch_size = 16  # how many independent sequences will we process in parallel?
    block_size = 32  # what is the maximum context length for predictions?
    max_iters = 200
    eval_interval = 100
    learning_rate = 1e-3
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    eval_iters = 200
    n_embd = 64
    n_head = 4
    n_layer = 4
    dropout = 0.0
    # ------------

    torch.manual_seed(1337)

    text, vocab_size = get_training_corpus()
    train_data, val_data = train_val_split(text)
    # here are all the unique characters that occur in this text

    # create a mapping from characters to integers

    model = SimpleLanguageModel(vocab_size, n_embd, block_size, n_head, n_layer, dropout, device)
    m = model.to(device)
    # print the number of parameters in the model
    print(sum(p.numel() for p in m.parameters()) / 1e6, 'M parameters')

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for iter in range(max_iters):

        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(model, eval_iters, device, train_data, val_data, block_size, batch_size)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # sample a batch of data
        xb, yb = get_batch('train', device, train_data, val_data, block_size, batch_size)

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # generate from the model
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(m.generate(context, max_new_tokens=100, block_size=block_size)[0].tolist()))
