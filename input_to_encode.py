import torch
import os

class Input_to_enode():
    def __init__(self):
        try:
            with open(r'data/input.txt', 'r', encoding='utf-8') as f:
                text = f.read()
        except:
            print('Input file not found')

        chars = sorted(list(set(text)))

        stoi = {ch:i for i,ch in enumerate(chars)}
        itos = {i:ch for i,ch in enumerate(chars)}

        encode = lambda s : [stoi[c] for c in s]
        decode = lambda l : "".join(itos[idx] for idx in l)

        data = torch.tensor(encode(text), dtype=torch.long)
        print('Input Data shape: ',data.shape)
        print('Input Data type: ', data.type,'\n')

        directory = 'files'
        
        if not os.path.exists(directory):
            os.makedirs(directory)

        try:
            file = open('data/encoded_input.txt', 'w')
            file.write(str(data.tolist()))
            file.close()

            file = open(os.path.join(directory,'stoi.txt'), 'w')
            file.write(str(stoi))
            file.close()

            file = open(os.path.join(directory,'itos.txt'), 'w')
            file.write(str(itos))
            file.close()
        except Exception as e:
            print('Error: ', e)

if __name__ == '__main__':
    Input_to_enode()
