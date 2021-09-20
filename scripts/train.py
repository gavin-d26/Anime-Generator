import torch
import torch.nn as nn
from torch.nn import parameter
import torchvision
import hydramodule
import model
import data
import config



FINAL_SAVE_PATH = config.FINAL_SAVE_PATH
DATA_SET_PATH = config.DATA_SET_PATH
parameters = config.parameters

if __name__ =='__main__':
    
    dataset = data.AnimeDataset(DATA_SET_PATH)
    
    model = model.GAN(parameters)
    
    model.fit(dataset,
              validation_dataset = None,
              epochs = parameters['epochs'],
              batch_size = parameters['batch_size'],
              num_workers = 2,
              checkpoint_metric = None,
              wandb_p = None,
              model_checkpoint_path= None,
              mixed_precision= False)
    
    
    model.to(device = torch.device('cpu'))
    torch.save(model, FINAL_SAVE_PATH)