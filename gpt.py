import torch 
import torch.nn as nn
from torch.nn import functional as F

batch_size = 4
block_size = 8

learning_rate = 1e-3
n_embd = 32

torch.manual_seed(681)

with open("input.txt",'r',encoding='utf-8') as f:
    text = f.read()

char_set = sorted(list(set(text)))
vocab_size = len(char_set)
stoi = {s:i for i,s in enumerate(char_set)}
itos = {i:s for s,i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text),dtype=torch.long)
n = int(0.9 * len(data))
train_set = data[:n]
test_set = data[n:]

def get_batch(split):
  data = train_set if split == 'train' else test_set
  ix = torch.randint(len(data)-block_size,(batch_size,))
  x = torch.stack([data[i:i+block_size] for i in ix])
  y = torch.stack([data[i+1:i+block_size+1] for i in ix])
  return x,y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'test']:
        losses = torch.zeros(100)
        for iter in range(100):
            X, y = get_batch(split)
            logits, loss = model(X,y)
            losses[iter] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    def __init__(self,head_size):
        super().__init__()
        self.key = nn.Linear(n_embd,head_size,bias=False)
        self.query = nn.Linear(n_embd,head_size,bias=False)
        self.value = nn.Linear(n_embd,head_size,bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size,block_size)))

    def forward(self,x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2,-1) * C ** -0.5 # (B,T,C) @ (B,C,T) --> (B,T,T) *** where T is 'vocab_size'
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        v = self.value(x)
        out = wei@v
        return out
    
class BiGramLanguageModel(nn.Module):
    def __init__(self,vocab_size):
        super().__init__()
        self.token_embd_table = nn.Embedding(vocab_size,n_embd)
        self.pos_embd = nn.Embedding(block_size,n_embd)
        self.sa_head = Head(n_embd)
        self.lm_head = nn.Linear(n_embd,vocab_size)

    def forward(self,idx,targets=None):
        B, T = idx.shape
        tok_embd = self.token_embd_table(idx)
        pos_embd = pos_embd = self.pos_embd(torch.arange(T, device=idx.device))
        x = tok_embd + pos_embd
        x = self.sa_head(x)
        logits = self.lm_head(x)

        if targets == None:
            loss = None
        else: 
            B,T,C = logits.shape
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits,targets)
        return logits, loss
    
    def generate(self,idx,max_new_token):
        for _ in range(max_new_token):
            idx_cont = idx[:,-block_size:]
            logits, loss = self(idx_cont)
            logits = logits[:,-1,:]
            probs = F.softmax(logits,dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

model = BiGramLanguageModel(vocab_size)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(10000):
    if iter % 1000 == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['test']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long)
print(decode(model.generate(context, max_new_token=500)[0].tolist()))
