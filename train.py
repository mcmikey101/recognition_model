import torch
import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
from tqdm import tqdm
from torch.utils.data import DataLoader

def train_epoch(train_data, model, opt, lossfn):
    model.train()
    epoch_loss = 0
    correct_sum = 0
    for x, y in tqdm(train_data):
        pred_classes, pred_embedding = model(x)
        loss = lossfn(pred_embedding, y)

        opt.zero_grad()
        loss.backward()
        opt.step()

        pred_labels = pred_classes.argmax(dim=1)
        correct = (pred_labels == y).sum().item()
        correct_sum += correct
        epoch_loss += loss.item()

    return epoch_loss / len(train_data), correct_sum / len(train_data)

def val_epoch(val_data, model, lossfn):
    model.eval()
    epoch_loss = 0
    correct_sum = 0
    with torch.no_grad():
        for x, y in tqdm(val_data):
            pred_classes, pred_embedding = model(x)
            loss = lossfn(pred_embedding, y)

            epoch_loss += loss
            pred_labels = pred_classes.argmax(dim=1)
            correct = (pred_labels == y).sum().item()
            correct_sum += correct

    return epoch_loss / len(val_data), correct_sum / len(val_data)

@hydra.main(config_path="./configs", config_name="train")
def main(cfg: DictConfig):
    train_dataset = instantiate(cfg.train_dataset)
    val_dataset = instantiate(cfg.val_dataset)

    train_dataloader = DataLoader(train_dataset, batch_size=cfg.training.batch)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.training.batch)

    for x, y in train_dataloader:
        print(x, y)

    model = instantiate(cfg.model)
    opt = instantiate(cfg.opt, params=model.parameters())
    scheduler = instantiate(cfg.scheduler, optimizer=opt, total_steps=cfg.training.epochs * len(train_dataset))
    loss = instantiate(cfg.loss)
    
    for epoch in range(cfg.training.epochs):
        train_loss, train_accuracy = train_epoch(train_dataloader, model, opt, scheduler, loss)
        val_loss, val_accuracy = val_epoch(val_dataloader, model, loss)
        print(f'Epoch {epoch} train loss: {train_loss}, train acc: {train_accuracy}')
        print(f'validaton loss: {val_loss}, validation acc: {val_accuracy}')
        
        scheduler.step()

if __name__ == "__main__":
    main()