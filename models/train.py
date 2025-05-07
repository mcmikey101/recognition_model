import torch
import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
from tqdm import tqdm
from torch.utils.data import DataLoader

def train_epoch(train_data, model, opt, lossfn):
    model.train()
    epoch_loss = 0
    for x, y in tqdm(train_data):
        pred = model(x)
        loss = lossfn(pred, y)

        opt.zero_grad()
        loss.backward()
        opt.step()

        epoch_loss += loss.item()

    return epoch_loss / len(train_data)

def val_epoch(val_data, model, lossfn):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for x, y in tqdm(val_data):
            pred = model(x)
            loss = lossfn(pred, y)

            epoch_loss += loss

    return epoch_loss / len(val_data)

@hydra.main(config_path="../configs/train", config_name="train")
def main(cfg: DictConfig):
    train_dataset = instantiate(cfg.train_dataset)
    val_dataset = instantiate(cfg.val_dataset)

    train_dataloader = DataLoader(train_dataset, batch_size=cfg.training.batch)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.training.batch)
    model = instantiate(cfg.model) 
    opt = instantiate(cfg.opt)
    scheduler = instantiate(cfg.scheduler)
    loss = instantiate(cfg.loss)

    for epoch in range(cfg.training.epochs):
        train_loss = train_epoch(train_dataloader, model, opt, scheduler, loss)
        val_loss = val_epoch(val_dataloader, model, loss)
        print(f'Train loss: {train_loss}')
        print(f"Validation loss: {val_loss}")
        
        scheduler.step()

if __name__ == "__main__":
    main()