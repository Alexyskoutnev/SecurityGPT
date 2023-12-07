import os
import argparse

from securityGPT.GPT.model_gpt import GPT
from securityGPT.utils.gpt_utils import *
from securityGPT.utils.utils import Cfgloader

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='Path to model weights', dest='model', default='2023-10-29_17-58-46_GPT.pth')
    args = vars(parser.parse_args())
    args['model'] = os.path.join(MODEL_SAVE_DIR, args['model'])
    return args

if __name__ == "__main__":
    DATASET_PATH = "../data"
    dataset_path = os.path.join(DATASET_PATH, 'Shakespeare.txt')
    with open(dataset_path, 'r', encoding='utf-8') as f:
        text = f.read()
    # here are all the unique characters that occur in this text
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    # create a mapping from characters to integers
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    decode = lambda l: ''.join([itos[i] for i in l])
    encode = lambda s: [stoi[c] for c in s]
    #=========CONFIG===========
    CONFIG_NAME_GPT = "test_gpt.yml"
    CONFIG_NAME_DATASET = "dataset.yml"
    DATASET_PATH = "../../data"
    MODEL_SAVE_DIR = "../models/gpt"
    cfg_loader = Cfgloader("../securityGPT/config")
    GPT_config = cfg_loader.load_config("test_gpt.yml")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    GPT_config['device'] = device
    #=========Model===========
    model = GPT(config=GPT_config).to(device)
    args = parser()
    load_model(model, args['model'])
    
    #=========Test============
    model.eval()
    context = torch.ones((1, 1), dtype=torch.long, device=device) * 1
    print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))