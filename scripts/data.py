import torch
import os
import torchvision
import torchvision.io as io
import torchvision.transforms as T
import PIL



class AnimeDataset(torch.utils.data.Dataset):
    def __init__(self, dir_path, stats = ((0.5,0.5,0.5),(0.5,0.5,0.5))):
        super(AnimeDataset, self).__init__()
        self.imglist = os.listdir(dir_path)
        self.length = len(self.imglist)
        self.dir_path = dir_path
        self.stats = stats
        self.transforms = T.Compose([T.ToTensor(),
                                    T.Normalize(*stats)])
        
        
    def __getitem__(self, index):
        img_path = os.path.join(self.dir_path, self.imglist[index])
        image = io.read_image(img_path, mode = io.image.ImageReadMode.RGB)
        image = PIL.Image.open(img_path)
        image = self.transforms(image)
        return image
    
    def __len__(self):
        return self.length