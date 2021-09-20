import torch
import torch.nn as nn
from torch.nn.modules.loss import BCEWithLogitsLoss
import torchvision
import hydramodule



class GenBlock(nn.Module):
    def __init__(self, inc, outc, ksize, stride, padding):
        super(GenBlock, self).__init__()
        self.upsample = nn.ConvTranspose2d(inc, outc, ksize, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(outc)
        self.activation = nn.ReLU(True)
        
    def forward(self, input):
        x = self.upsample(input)
        x= self.bn(x)
        x= self.activation(x)
        return x    
        




class Generator(nn.Module):
    def __init__(self,nz, ngf=64, nc = 3):
        super(Generator, self).__init__()
        self.encoder = nn.Sequential(
            GenBlock(nz, ngf * 8, 4, 1, 0, bias=False),
            GenBlock(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            GenBlock(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            GenBlock(ngf * 2, ngf*2, 4, 2, 1, bias=False),
            #final block
            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.Conv2d(ngf, ngf, bias= False),
            nn.BatchNorm2d(ngf),
            nn.Conv2d(ngf, nc, bias = True),
            nn.Tanh()
        )

    def forward(self, input):
        output = self.encoder(input)
        return output
    
    
class DisBlock(nn.Module):
    def __init__(self, inc, outc, ksize, stride, padding):
        super(GenBlock, self).__init__()
        self.conv = nn.Conv2d(inc, outc, ksize, stride, padding, bias=False),
        self.bn = nn.BatchNorm2d(outc),
        self.activation = nn.LeakyReLU(0.2, inplace=True),
        
    def forward(self, input):
        x = self.conv(input)
        x= self.bn(x)
        x= self.activation(x)
        return x        
    
    
    
class Discriminator(nn.Module):
    def __init__(self, nc = 3, ndf = 64):
        super(Discriminator, self).__init__()
        
        self.encoder = nn.Sequential(
            # input is (nc) x 64 x 64
            DisBlock(nc, ndf, 4, 2, 1, bias=False),
            DisBlock(ndf, ndf * 2, 4, 2, 1, bias=False),
            DisBlock(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            DisBlock(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            #nn.Sigmoid()
        )

    def forward(self, input):
        output = self.encoder(input)
        return output
    
    
class GAN(hydramodule.HydraModule):
    def __init__(self, config):
        super(GAN, self).__init__()
        self.config = config
        self.nz = config['nz']
        self.training_steps_count = 0
        self.dis_freq = config['dis_freq']
        G = Generator(self.nz) 
        D = Discriminator()
        G.apply(self.weights_init)
        D.apply(self.weights_init)
        self.G = G
        self.D = D
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.example_noise = self.get_noise(5)
    
    def configure_loss_fn(self, *args, **kwargs):
        return [BCEWithLogitsLoss]
    
    def configure_optimizers_and_schedulers(self):
        dis_opt = torch.optim.Adam(self.D.parameters(), lr = self.config['lr'], betas(self.config['beta1'], self.config['beta2']))
        gen_opt = torch.optim.Adam(self.G.parameters(), lr = self.config['lr'], betas(self.config['beta1'], self.config['beta2']))
        return [dis_opt, gen_opt]
    
    
    
    @staticmethod    
    def weights_init(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
        if isinstance(m, nn.BatchNorm2d):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
            torch.nn.init.constant_(m.bias, 0)    
    
    
    def forward(self, input):
        x = self.G(input)
        return x 
    
    
    def get_noise(self, n_samples):
        return torch.randn(n_samples, self.nz, device = self.device)


    def train_step(self, batch):
        dis_opt, gen_opt = self.get_optimizers()
        loss_fn = self.get_loss_fn(self)
        batch_len = len(batch)
        real_images = batch
        ## discriminator training
        if self.training_steps_count % self.dis_freq == 0:
            dis_opt.zero_grad()
            fake_noise = self.get_noise(batch_len)
            with torch.no_grad():
                fake_images = self.G(fake_noise)
            
            dis_fake_pred = self.D(fake_images)
            dis_fake_loss = loss_fn(dis_fake_pred, torch.zeros_like(dis_fake_pred))
            
            dis_real_pred = self.D(real_images)
            dis_real_loss = loss_fn(dis_real_pred, torch.ones_like(dis_real_pred))
            
            dis_net_loss = (dis_real_loss + dis_fake_loss)/2
            
            dis_net_loss.backward()
            dis_opt.step()
        
        # generator training
        
        gen_opt.zero_grad()
        fake_noise2 = self.get_noice(batch_len)
        
        dis_fake_pred2 = self.D(self.G(fake_noise2))
        
        gen_loss = loss_fn(dis_fake_pred2, torch.ones_like(dis_fake_pred2)) 
        gen_loss.backward()
        gen_opt.step()
        
        
        # print loss values and display example images
        if self.training_steps_count % (self.dis_freq * 10) == 0:
            print(f'disc_loss: {dis_net_loss}    gen_loss: {gen_loss}' )
            torchvision.utils.make_grid(self.G(self.example_noise), nrow = 1)
            
        self.training_steps_count+=1
        
        return None, None, None
        
        
        
        
            
        
        
        
        
                  
           