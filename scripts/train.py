import torch
import torch.nn as nn
from torch.nn import parameter
import torchvision
import wandb
import hydramodule
import model
import data
import config


PROJECT_NAME = config.PROJECT_NAME
ENTITY_NAME = config.ENTITY_NAME
RUN_NAME = config.RUN_NAME
FINAL_SAVE_PATH = config.FINAL_SAVE_PATH
DATA_SET_PATH = config.DATA_SET_PATH
parameters = config.parameters

if __name__ =='__main__':
    
    dataset = data.AnimeDataset(DATA_SET_PATH)
    
    wandb.init(entity = ENTITY_NAME, project = PROJECT_NAME, name = RUN_NAME, config = parameters)
    wandb_config = wandb.config
    model = model.GAN(wandb_config)
    
    model.fit(dataset,
              validation_dataset = None,
              epochs = wandb_config.epochs,
              batch_size = wandb_config.batch_size,
              num_workers = 2,
              checkpoint_metric = None,
              wandb_p = None,
              model_checkpoint_path= None,
              mixed_precision= False)
    
    
    model.to(device = torch.device('cpu'))
    model.model_save(FINAL_SAVE_PATH)