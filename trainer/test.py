#!/usr/bin/python3

import argparse
import itertools
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os
from torchvision.transforms import RandomAffine
from torchvision.transforms import RandomAffine,ToPILImage
from skimage import measure
import numpy as np
import cv2

class Nice_Trainer():
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = torch.device("cpu")

        ## def networks
        
        if config['regist']:
            self.R_A = Reg(config['size'], config['size'],config['input_nc'],config['input_nc'])
            self.spatial_transform = Transformer_2D()
            self.optimizer_R_A = torch.optim.Adam(self.R_A.parameters(), lr=config['lr'], betas=(0.5, 0.999))
        if config['bidirect']:
            self.netG_A2B = ResnetGenerator(config['input_nc']*128, config['output_nc'])
            self.netG_B2A = ResnetGenerator(config['input_nc']*128, config['output_nc'])
            
            self.netD_A = Discriminator(config['input_nc'])
            self.netD_B = Discriminator(config['input_nc'])
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A2B.parameters(), self.netG_B2A.parameters()),lr=config['lr'], betas=(0.5, 0.999))
            self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=config['lr'], betas=(0.5, 0.999))
            self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=config['lr'], betas=(0.5, 0.999))
        else:
            self.netD_B = Discriminator2(config['input_nc'])
            self.netG_A2B = ResnetGenerator2(config['input_nc'], config['output_nc'])
            self.optimizer_G = torch.optim.Adam(self.netG_A2B.parameters(), lr=config['lr'], betas=(0.5, 0.999))
            self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=config['lr'], betas=(0.5, 0.999))
                
        # Losses
        self.MSE_loss = torch.nn.MSELoss()
        self.L1_loss = torch.nn.L1Loss()
        # Inputs & targets memory allocation
        Tensor = torch.Tensor
        self.input_A = Tensor(config['batchSize'], config['input_nc'], config['size'], config['size'])
        self.input_B = Tensor(config['batchSize'], config['output_nc'], config['size'], config['size'])

        # Dataset loader
        level = config['noise_level']
        transforms_1 = [ToPILImage(),
                   RandomAffine(degrees=level,translate=[0.02*level, 0.02*level],scale=[1-0.02*level, 1+0.02*level],fill=-1),
                   ToTensor(),
                   Resize(size_tuple = (config['size'], config['size']))]
    
        transforms_2 = [ToPILImage(),
                   RandomAffine(degrees=level,translate=[0.02*level, 0.02*level],scale=[1-0.02*level, 1+0.02*level],fill=-1),
                   ToTensor(),
                   Resize(size_tuple = (config['size'], config['size']))]

        self.dataloader = DataLoader(ImageDataset(config['dataroot'],0, transforms_1=transforms_1, transforms_2=transforms_2, unaligned=False),
                                batch_size=config['batchSize'], shuffle=True)
        
        
        val_transforms = [ToTensor(),
                    Resize(size_tuple = (config['size'], config['size']))]
        
        self.val_data = DataLoader(ValDataset(config['val_dataroot'], transforms_ =val_transforms, unaligned=False),
                                batch_size=config['batchSize'], shuffle=False)

 
       # Loss plot
        self.logger = Logger(config['name'],config['port'],config['n_epochs'], len(self.dataloader))       
        
    def train(self):
        ###### Training ######
        for epoch in range(self.config['epoch'], self.config['n_epochs']):
            for i, batch in enumerate(self.dataloader):
                # Set model input
                real_A = Variable(self.input_A.copy_(batch['A'])).cpu()
                real_B = Variable(self.input_B.copy_(batch['B'])).cpu()
                if self.config['bidirect']:   # b dir
                    if self.config['regist']:    # + reg 
                        self.optimizer_D_A.zero_grad()
                        self.optimizer_D_B.zero_grad()

                        real_LA_logit,real_GA_logit, real_A_cam_logit, _, real_A_z = self.netD_A(real_A)
                        real_LB_logit,real_GB_logit, real_B_cam_logit, _, real_B_z = self.netD_B(real_B)

                        fake_A2B = self.netG_A2B(real_A_z)
                        fake_B2A = self.netG_B2A(real_B_z)

                        fake_B2A = fake_B2A.detach()
                        fake_A2B = fake_A2B.detach()

                        fake_LA_logit, fake_GA_logit, fake_A_cam_logit, _, _ = self.netD_A(fake_B2A)
                        fake_LB_logit, fake_GB_logit, fake_B_cam_logit, _, _ = self.netD_B(fake_A2B)

                        D_ad_loss_GA = self.MSE_loss(real_GA_logit, torch.ones_like(real_GA_logit)) + self.MSE_loss(fake_GA_logit, torch.zeros_like(fake_GA_logit))
                        D_ad_loss_LA = self.MSE_loss(real_LA_logit, torch.ones_like(real_LA_logit)) + self.MSE_loss(fake_LA_logit, torch.zeros_like(fake_LA_logit))
                        D_ad_loss_GB = self.MSE_loss(real_GB_logit, torch.ones_like(real_GB_logit)) + self.MSE_loss(fake_GB_logit, torch.zeros_like(fake_GB_logit))
                        D_ad_loss_LB = self.MSE_loss(real_LB_logit, torch.ones_like(real_LB_logit)) + self.MSE_loss(fake_LB_logit, torch.zeros_like(fake_LB_logit))            
                        D_ad_cam_loss_A = self.MSE_loss(real_A_cam_logit, torch.ones_like(real_A_cam_logit)) + self.MSE_loss(fake_A_cam_logit, torch.zeros_like(fake_A_cam_logit))
                        D_ad_cam_loss_B = self.MSE_loss(real_B_cam_logit, torch.ones_like(real_B_cam_logit)) + self.MSE_loss(fake_B_cam_logit, torch.zeros_like(fake_B_cam_logit))

                        loss_D_A = (D_ad_loss_GA + D_ad_cam_loss_A + D_ad_loss_LA)
                        loss_D_B = (D_ad_loss_GB + D_ad_cam_loss_B + D_ad_loss_LB)

                        Discriminator_loss = self.config['Adv_lamda'] * (loss_D_A + loss_D_B)
                        Discriminator_loss.backward()
                        self.optimizer_D_A.step()
                        self.optimizer_D_B.step()

                    # train G_A2B and G_B2A
                    self.optimizer_G.zero_grad()

                    fake_A2B = self.netG_A2B(real_A_z)
                    fake_B2A = self.netG_B2A(real_B_z)

                    fake_LA_logit, fake_GA_logit, fake_A_cam_logit, _, _ = self.netD_A(fake_B2A)
                    fake_LB_logit, fake_GB_logit, fake_B_cam_logit, _, _ = self.netD_B(fake_A2B)

                    cycle_A2A = self.netG_B2A(fake_A2B)
                    cycle_B2B = self.netG_A2B(fake_B2A)
                    # Identity loss
                    loss_id_A = self.L1_loss(real_A, cycle_A2A)
                    loss_id_B = self.L1_loss(real_B, cycle_B2B)
                    # GAN loss
                    loss_G_A = self.MSE_loss(fake_GA_logit, torch.ones_like(fake_GA_logit))
                    loss_G_B = self.MSE_loss(fake_GB_logit, torch.ones_like(fake_GB_logit))
                    # cycle loss
                    loss_cycle_A = self.L1_loss(cycle_A2A, real_A)
                    loss_cycle_B = self.L1_loss(cycle_B2B, real_B)
                    # cam loss
                    loss_cam_A = self.MSE_loss(fake_A_cam_logit, torch.ones_like(fake_A_cam_logit))
                    loss_cam_B = self.MSE_loss(fake_B_cam_logit, torch.ones_like(fake_B_cam_logit))
                    # Total loss
                    loss_G = self.config['gan_lamda'] * (loss_G_A + loss_G_B) + self.config['cycle_lamda'] * (loss_cycle_A + loss_cycle_B) + self.config['cam_lamda'] * (loss_cam_A + loss_cam_B) + self.config['id_lamda'] * (loss_id_A + loss_id_B)

                    loss_G.backward()
                    self.optimizer_G.step()
                    self.logger.log({'loss_G': loss_G.item(), 'loss_G_A': loss_G_A.item(), 'loss_G_B': loss_G_B.item(), 'loss_cycle_A': loss_cycle_A.item(), 'loss_cycle_B': loss_cycle_B.item(), 'loss_idt_A': loss_id_A.item(), 'loss_idt_B': loss_id_B.item(), 'loss_cam_A': loss_cam_A.item(), 'loss_cam_B': loss_cam_B.item(), 'loss_D_A': loss_D_A.item(), 'loss_D_B': loss_D_B.item()})

                else:  # Unidirectional
                    if self.config['regist']:
                        self.optimizer_D_B.zero_grad()
                        real_LB_logit, real_GB_logit, real_B_cam_logit, _, real_B_z
                        real_LB_logit, real_GB_logit, real_B_cam_logit, _, real_B_z = self.netD_B(real_B)

                        fake_B2A = self.netG_A2B(real_B_z)

                        fake_B2A = fake_B2A.detach()

                        fake_LB_logit, fake_GB_logit, fake_B_cam_logit, _, _ = self.netD_B(fake_B2A)

                        D_ad_loss_GB = self.MSE_loss(real_GB_logit, torch.ones_like(real_GB_logit)) + self.MSE_loss(fake_GB_logit, torch.zeros_like(fake_GB_logit))
                        D_ad_loss_LB = self.MSE_loss(real_LB_logit, torch.ones_like(real_LB_logit)) + self.MSE_loss(fake_LB_logit, torch.zeros_like(fake_LB_logit))            
                        D_ad_cam_loss_B = self.MSE_loss(real_B_cam_logit, torch.ones_like(real_B_cam_logit)) + self.MSE_loss(fake_B_cam_logit, torch.zeros_like(fake_B_cam_logit))

                        loss_D_B = (D_ad_loss_GB + D_ad_cam_loss_B + D_ad_loss_LB)

                        Discriminator_loss = self.config['Adv_lamda'] * loss_D_B
                        Discriminator_loss.backward()
                        self.optimizer_D_B.step()

                    self.optimizer_G.zero_grad()
                    fake_B2A = self.netG_A2B(real_B_z)
                    fake_LB_logit, fake_GB_logit, fake_B_cam_logit, _, _ = self.netD_B(fake_B2A)
                    cycle_B2B = self.netG_A2B(fake_B2A)
                    loss_cycle_B = self.L1_loss(cycle_B2B, real_B)
                    loss_G_B = self.MSE_loss(fake_GB_logit, torch.ones_like(fake_GB_logit))
                    loss_cam_B = self.MSE_loss(fake_B_cam_logit, torch.ones_like(fake_B_cam_logit))

                    loss_G = self.config['gan_lamda'] * loss_G_B + self.config['cycle_lamda'] * loss_cycle_B + self.config['cam_lamda'] * loss_cam_B
                    loss_G.backward()
                    self.optimizer_G.step()
                    self.logger.log({'loss_G': loss_G.item(), 'loss_G_B': loss_G_B.item(), 'loss_cycle_B': loss_cycle_B.item(), 'loss_cam_B': loss_cam_B.item(), 'loss_D_B': loss_D_B.item()})
                
                # Display the log and images
                if i % self.config['sample_interval'] == 0:
                    self.logger.display_status(epoch, self.config['n_epochs'], i, len(self.dataloader), {'loss_G': loss_G.item(), 'loss_G_A': loss_G_A.item(), 'loss_G_B': loss_G_B.item(), 'loss_cycle_A': loss_cycle_A.item(), 'loss_cycle_B': loss_cycle_B.item(), 'loss_idt_A': loss_id_A.item(), 'loss_idt_B': loss_id_B.item(), 'loss_cam_A': loss_cam_A.item(), 'loss_cam_B': loss_cam_B.item(), 'loss_D_A': loss_D_A.item(), 'loss_D_B': loss_D_B.item()})
                    self.logger.log_images(fake_A2B, fake_B2A, real_A, real_B, epoch, i, len(self.dataloader))

            # Update learning rates
            self.lr_scheduler_G.step()
            self.lr_scheduler_D_A.step()
            self.lr_scheduler_D_B.step()
            if epoch % self.config['checkpoint_interval'] == 0:
                # Save models checkpoints
                torch.save(self.netG_A2B.state_dict(), self.config['save_root'] + 'netG_A2B.pth')
                torch.save(self.netG_B2A.state_dict(), self.config['save_root'] + 'netG_B2A.pth')
                torch.save(self.netD_A.state_dict(), self.config['save_root'] + 'netD_A.pth')
                torch.save(self.netD_B.state_dict(), self.config['save_root'] + 'netD_B.pth')
                if self.config['regist']:
                    torch.save(self.R_A.state_dict(), self.config['save_root'] + 'netR_A.pth')

    def test(self):
        ###### Testing ######
        mean_psnr = []
        mean_ssim = []
        for i, batch in enumerate(self.val_data):
            real_A = Variable(self.input_A.copy_(batch['A'])).cpu()
            real_B = Variable(self.input_B.copy_(batch['B'])).cpu()
            with torch.no_grad():
                if self.config['regist']:
                    fake_A2B = self.R_A(real_A)
                else:
                    fake_A2B = self.netG_A2B(real_A)
                if self.config['bidirect']:
                    if self.config['regist']:
                        fake_B2A = self.R_A(real_B)
                    else:
                        fake_B2A = self.netG_B2A(real_B)

                    if self.config['eval']:
                        fake_B2A = self.spatial_transform(fake_B2A, real_A)

                else:
                    fake_B2A = self.netG_B2A(real_B)

                if self.config['regist']:
                    real_A = self.R_A(real_A)

                self.save_img(real_A, self.config['save_val_root'], '_real_A_', i)
                self.save_img(real_B, self.config['save_val_root'], '_real_B_', i)
                self.save_img(fake_A2B, self.config['save_val_root'], '_fake_A2B_', i)
                self.save_img(fake_B2A, self.config['save_val_root'], '_fake_B2A_', i)

                if self.config['eval']:
                    fake_A2B = self.spatial_transform(fake_A2B, real_B)
                    self.save_img(fake_A2B, self.config['save_val_root'], '_fake_A2B_eval_', i)
                    
                # Calculate PSNR and SSIM
                psnr = self.calculate_psnr(fake_A2B, real_B)
                ssim = self.calculate_ssim(fake_A2B, real_B)
                mean_psnr.append(psnr)
                mean_ssim.append(ssim)

        # Calculate mean PSNR and SSIM
        mean_psnr = np.mean(mean_psnr)
        mean_ssim = np.mean(mean_ssim)
        print(f"Mean PSNR: {mean_psnr:.4f}, Mean SSIM: {mean_ssim:.4f}")

    def save_img(self, img, save_dir, prefix, idx):
        img = img.squeeze(0).permute(1, 2, 0).numpy()
        img = (img * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(save_dir, f"{prefix}{idx}.png"), img)

    def calculate_psnr(self, img1, img2):
        mse = torch.mean((img1 - img2) ** 2)
        psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
        return psnr.item()

    def calculate_ssim(self, img1, img2):
        img1 = img1.squeeze(0).permute(1, 2, 0).numpy()
        img2 = img2.squeeze(0).permute(1, 2, 0).numpy()
        ssim = measure.compare_ssim(img1, img2, multichannel=True)
        return ssim
