import os
import ast
from input_to_encode import Input_to_enode

if os.path.exists(r'data/encoded_input.txt'):
    print('File Exists')
    with open(r'data/encoded_input.txt', 'r', encoding='utf-8') as f:
        data = ast.literal_eval(f.read())
    with open('files/stoi.txt', 'r', encoding='utf-8') as f:
        stoi = ast.literal_eval(f.read())
    with open('files/itos.txt', 'r', encoding='utf-8') as f:
        itos = ast.literal_eval(f.read())
else:
    Input_to_enode()
    print('encoded_input.txt created!')

n = int(0.8*len(data))
train_data = data[:n]
val_data = data[n:]

block_size = 8

x = train_data[:block_size]
y = train_data[1:block_size+1]

for idx in range(block_size):
    context = x[:idx+1]
    target = y[idx]
    print(f'When context is {context} the target is {target}')