import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate

@hydra.main(config_path='./configs', config_name='train')
def main(cfg: DictConfig):
    train_dataset = instantiate(cfg.dataset)

    model = instantiate(cfg.model)
    opt = instantiate(cfg.opt, params=model.parameters())
    scheduler = instantiate(cfg.scheduler, optimizer=opt, total_steps=cfg.training.epochs * len(train_dataset))
    loss = instantiate(cfg.loss)

    print(type(model))
    print(type(opt))
    print(type(scheduler))
    print(type(loss))
    print(type(train_dataset))

if __name__ == '__main__':
    main()