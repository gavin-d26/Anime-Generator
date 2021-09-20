import torch
import os
import torchvision
import torchvision.io as io



class AnimeDataset(torch.utils.data.Dataset):
    def __init__(self, dir_path):
        super(AnimeDataset, self).__init__()
        self.length = len(os.path.listdir(dir_path))
        self.dir_path = dir_path
        
        
    def __getitem__(self, index):
        img_path = os.path.join(self.dir_path, str(index)+'.png')
        image = io.read_image(img_path, mode = io.image.ImageReadMode.RGB)
        return image
    
    def __len__(self):
        return self.length