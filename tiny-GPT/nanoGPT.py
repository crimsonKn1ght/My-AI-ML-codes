import tiktoken as ttn
import torch

txt_dataset = "C:/Users/gouro/Desktop/text_dataset.txt"

with open(txt_dataset, 'r', encoding = 'utf-8') as f:
    text = f.read()
    
print("The number of dataset in characters: ", len(text))

chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)

enc = ttn.get_encoding('gpt2')
print(enc.n_vocab)

data = torch.tensor(enc.encode(text), dtype = torch.long)
print(data.shape, data.dtype)
# print(data[:1000])

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

block_size = 8
x = train_data[:block_size]
y = train_data[1:block_size+1]

for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    print(f"When input is {context} target is {target}")
    
torch.manual_seed(1337)
batch_size = 4

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size, ))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

xb, yb = get_batch('train')
print('Inputs:')
print(xb.shape)
print(xb)
print('Targets:')
print(yb.shape)
print(yb)

print('-----')

for b in range(batch_size):
    for t in range(block_size):
        context = xb[b, :t+1]
        target = yb[b, t]
        print(f"When input is {context.tolist()} the target: {target}")