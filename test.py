import os
os.environ["HYDRA_FULL_ERROR"] = "1"
import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate

@hydra.main(config_path='./configs', config_name='train')
def main(cfg: DictConfig):
    model = instantiate(cfg.model)
    opt = instantiate(cfg.opt, params=model.parameters())
    scheduler = instantiate(cfg.scheduler, optimizer=opt, max_lr=0, total_steps=cfg.training.epochs)
    loss = instantiate(cfg.loss)
    train_dataset = instantiate(cfg.train_dataset)

    print(train_dataset)

if __name__ == '__main__':
    main()