import datetime
import torch

def get_batch(split, train_data, val_data, block_size, batch_size=64, device='cpu'):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model, eval_iters, train_data, test_data, block_size=256, batch_size=64, device='cpu'):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters).to(device)
        for k in range(eval_iters):
            X, Y = get_batch(split, train_data, test_data, block_size, batch_size, device)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
    
def save_model(model : torch.nn.Module, save_directory : str):
    try:
        current_datetime = datetime.datetime.now()
        current_date_str = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
        save_file = f"{current_date_str}_{model.__class__.__name__}.pth"
        torch.save(model.state_dict(), save_directory + '/' + save_file)
        print(f"Model saved to {save_directory}/{save_file}")
    except Exception as e:
        print(f"Model failed to save. Error: {str(e)}")

def load_model(model, model_path : str) -> torch.nn.Module:
    return model.load_state_dict(torch.load(model_path))