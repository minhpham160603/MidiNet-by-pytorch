import numpy as np 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import ipdb
import torchvision.utils as vutils
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model2 import *
from ops import *
import os

class get_dataloader(object):
    def __init__(self, data, prev_data, y):
        self.size = data.shape[0]
        self.data = torch.from_numpy(data).float()
        self.prev_data = torch.from_numpy(prev_data).float()  # (m, 1, 16, 128)
        self.y   = torch.from_numpy(y).float()  # (m, 13)

         # self.label = np.array(label)
    def __getitem__(self, index):
        return self.data[index],self.prev_data[index], self.y[index]

    def __len__(self):
        return self.size
    
def creat_directory(directory):
    # Create the directory if it doesn't exist
    print(directory)
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def clean_folder(folder_path):
    # Iterate over all files in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # Check if the path is a file (not a directory)
        if os.path.isfile(file_path):
            # Delete the file
            os.remove(file_path)
            print(f"Deleted file: {file_path}")

def load_data(augpath):
    #######load the data########
    check_range_st = 0
    check_range_ed = 129
    pitch_range = check_range_ed - check_range_st-1
    # print('pitch range: {}'.format(pitch_range))
    path = f"./dataset/{augpath}/"
    X_tr = np.load(os.path.join(path, "X_tr.npy"))
    prev_X_tr = np.load(os.path.join(path, "prev_X_tr.npy"))
    y_tr    = np.load(os.path.join(path, "Y_tr.npy"))
    X_tr = X_tr[:,:,:,check_range_st:check_range_ed]
    prev_X_tr = prev_X_tr[:,:,:,check_range_st:check_range_ed]
    #test data shape(5048, 1, 16, 128)
    #train data shape(45448, 1, 16, 128)

    train_iter = get_dataloader(X_tr,prev_X_tr,y_tr)
    kwargs = {'num_workers': 4, 'pin_memory': True}# if args.cuda else {}
    train_loader = DataLoader(
                   train_iter, batch_size=72, shuffle=True, **kwargs)

    print('data preparation is completed')
    #######################################
    return train_loader

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_preset(model, attempt):
    if model == 3:
        if attempt == 1:
            return 0.0002, 0.0002, 1.0, 0.01, 0.1 
        if attempt == 2:
            return 0.0001, 0.0002, 0.9, 0.01, 0.1
    if model == 1:
        if attempt == 1:
            return 0.0002, 0.0002, 1.0, 0.1, 1.0
        if attempt == 2:
            return 0.0001, 0.0002, 0.9, 0.1, 1.0
    if model == 2:
        if attempt == 1:
            return 0.0002, 0.0002, 1.0, 0.01, 0.1
        if attempt == 2:
            return 0.0001, 0.0002, 0.9, 0.01, 0.1

def main():
    modes = input("train | draw |sample: ")
    modes = modes.split(" ")
    is_train = int(modes[0])
    is_draw = int(modes[1])
    is_sample = int(modes[2])

    model_version = int(input("model version: "))
    attempt = int(input("attempt: "))
    is_augmented = int(input("augmented: ")) 
    aug_path = "augmented"
    if not is_augmented:
        aug_path = "not_augmented"

    init_path = f"./{model_version}/{aug_path}/{model_version}.{attempt}"
    creat_directory(init_path)

    epochs = 100
    lr_d, lr_g, softer, fm_1, fm_2 = get_preset(model_version, attempt)
    check_range_st = 0
    check_range_ed = 129
    pitch_range = check_range_ed - check_range_st-1
    
    device = torch.device('cuda')
    train_loader = load_data(aug_path)
    loss_path = init_path + "/loss"
    img_path = init_path + "/img_file"
    model_path = init_path + "/models"
    creat_directory(loss_path)
    creat_directory(img_path)
    creat_directory(model_path)

    if is_train == 1 :
        clean_folder(loss_path)
        clean_folder(img_path)
        clean_folder(model_path)
        netG = generator(pitch_range).to(device)
        netD = discriminator(pitch_range).to(device)  

        netD.train()
        netG.train()
        optimizerD = optim.Adam(netD.parameters(), lr=lr_g, betas=(0.5, 0.999))
        optimizerG = optim.Adam(netG.parameters(), lr=lr_g, betas=(0.5, 0.999)) 
        print("Number of params net D: ", count_params(netD))
        print("Number of params net G: ", count_params(netG))
        batch_size = 72
        nz = 100
        fixed_noise = torch.randn(batch_size, nz, device=device)
        real_label = 1
        fake_label = 0
        average_lossD = 0
        average_lossG = 0
        average_D_x   = 0
        average_D_G_z = 0

        lossD_list =  []
        lossD_list_all = []
        lossG_list =  []
        lossG_list_all = []
        D_x_list = []
        D_G_z_list = []
        for epoch in range(epochs):
            sum_lossD = 0
            sum_lossG = 0
            sum_D_x   = 0  # X: real data, D(X): 0 - 1;
            sum_D_G_z = 0  # G(z): output of generator - fake data D(G(z)): [0, 1]
            for i, (data,prev_data,chord) in enumerate(train_loader, 0):
                
                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                # We need D(x) closed to 1 and D(G(z)) closed to 0 so that log close to 0
                ###########################
                # train with real
                netD.zero_grad()
                real_cpu = data.to(device)
                prev_data_cpu = prev_data.to(device)
                chord_cpu = chord.to(device)

                batch_size = real_cpu.size(0)
                label = torch.full((batch_size,), real_label, device=device)
                D, D_logits, fm = netD(real_cpu,chord_cpu,batch_size,pitch_range)

                #####loss
                d_loss_real = reduce_mean(sigmoid_cross_entropy_with_logits(D_logits, softer*torch.ones_like(D)))
                d_loss_real.backward(retain_graph=True)
                D_x = D.mean().item()
                sum_D_x += D_x 

                # train with fake
                noise = torch.randn(batch_size, nz, device=device)
                fake = netG(noise,prev_data_cpu,chord_cpu,batch_size,pitch_range)
                label.fill_(fake_label)
                D_, D_logits_, fm_ = netD(fake.detach(),chord_cpu,batch_size,pitch_range)
                d_loss_fake = reduce_mean(sigmoid_cross_entropy_with_logits(D_logits_, (1.0 - softer)*torch.ones_like(D_)))
      
                d_loss_fake.backward(retain_graph=True)
                D_G_z1 = D_.mean().item()
                errD = d_loss_real + d_loss_fake
                errD = errD.item()
                lossD_list_all.append(errD)
                sum_lossD += errD
                optimizerD.step()

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                netG.zero_grad()
                label.fill_(real_label)  # fake labels are real for generator cost
                D_, D_logits_, fm_= netD(fake,chord_cpu,batch_size,pitch_range)

                ###loss
                g_loss0 = reduce_mean(sigmoid_cross_entropy_with_logits(D_logits_, torch.ones_like(D_)))
                #Feature Matching
                features_from_g = reduce_mean_0(fm_)
                features_from_i = reduce_mean_0(fm)
                fm_g_loss1 =torch.mul(l2_loss(features_from_g, features_from_i), 0.1)

                mean_image_from_g = reduce_mean_0(fake)
                smean_image_from_i = reduce_mean_0(real_cpu)
                fm_g_loss2 = torch.mul(l2_loss(mean_image_from_g, smean_image_from_i), 0.01)

                errG = g_loss0 + fm_g_loss1 + fm_g_loss2
                errG.backward(retain_graph=True)
                D_G_z2 = D_.mean().item()
                optimizerG.step()
              
                ############################
                # (3) Update G network again: maximize log(D(G(z)))
                ###########################
                netG.zero_grad()
                label.fill_(real_label)  # fake labels are real for generator cost
                D_, D_logits_, fm_ = netD(fake,chord_cpu,batch_size,pitch_range)

                ###loss
                g_loss0 = reduce_mean(sigmoid_cross_entropy_with_logits(D_logits_, torch.ones_like(D_)))
                #Feature Matching
                features_from_g = reduce_mean_0(fm_)
                features_from_i = reduce_mean_0(fm)
                loss_ = nn.MSELoss(reduction='sum')
                feature_l2_loss = loss_(features_from_g, features_from_i)/2
                fm_g_loss1 =torch.mul(feature_l2_loss, fm_2)

                mean_image_from_g = reduce_mean_0(fake)
                smean_image_from_i = reduce_mean_0(real_cpu)
                mean_l2_loss = loss_(mean_image_from_g, smean_image_from_i)/2
                fm_g_loss2 = torch.mul(mean_l2_loss, fm_1)
                errG = g_loss0 + fm_g_loss1 + fm_g_loss2
                sum_lossG +=errG
                errG.backward()
                lossG_list_all.append(errG.item())

                D_G_z2 = D_.mean().item()
                sum_D_G_z += D_G_z2
                optimizerG.step()
            
                # if epoch % 5 == 0:
                #     print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                #           % (epoch, epochs, i, len(train_loader),
                #              errD, errG, D_x, D_G_z1, D_G_z2))

                if i % 100 == 0 and (epoch % 100 == 0 or epoch == epochs - 1):
                    # vutils.save_image(real_cpu,
                    #         '%s/real_samples.png' % f'file/{model_version}.{attempt}',
                    #         normalize=True)

                    vutils.save_image(real_cpu,
                            os.path.join(img_path, "real_sample.png"),
                            normalize=True)
                    fake = netG(fixed_noise,prev_data_cpu,chord_cpu,batch_size,pitch_range)
                    # vutils.save_image(fake.detach(),
                    #         '%s/fake_samples_epoch_%03d.png' % ('file', epoch),
                    #         normalize=True)
                    vutils.save_image(fake.detach(),
                            os.path.join(img_path, f"fake_sample_epochs_{epoch}.png"),
                            normalize=True)

            if epoch % 10 == 0:
                torch.save(netG.state_dict(), os.path.join(model_path, f'netG_epoch_{epoch}.pth'))
                torch.save(netD.state_dict(), os.path.join(model_path, f'netD_epoch_{epoch}.pth'))    

            average_lossD = (sum_lossD / len(train_loader))
            average_lossG = (sum_lossG / len(train_loader))
            average_D_x = (sum_D_x / len(train_loader))
            average_D_G_z = (sum_D_G_z / len(train_loader))
            lossD_list.append(average_lossD)
            lossG_list.append(average_lossG)            
            D_x_list.append(average_D_x)
            D_G_z_list.append(average_D_G_z)

            print('==> Epoch: {} Average lossD: {:.10f} average_lossG: {:.10f},average D(x): {:.10f},average D(G(z)): {:.10f} '.format(
              epoch, average_lossD, average_lossG, average_D_x, average_D_G_z)) 


        lossG_list_tensor = torch.tensor(lossG_list)
        np.save(os.path.join(loss_path, 'lossD_list.npy'), lossD_list)
        np.save(os.path.join(loss_path, 'lossG_list.npy'), lossG_list_tensor.cpu().numpy())
        np.save(os.path.join(loss_path, 'lossD_list_all.npy'),lossD_list_all)
        np.save(os.path.join(loss_path, 'lossG_list_all.npy'),lossG_list_all)
        np.save(os.path.join(loss_path, 'D_x_list.npy'),D_x_list)
        np.save(os.path.join(loss_path, 'D_G_z_list.npy'),D_G_z_list)
        
        # do checkpointing


        

    if is_draw == 1:
        
        lossD_print = np.load(os.path.join(loss_path, 'lossD_list.npy'))
        lossG_print = np.load(os.path.join(loss_path, 'lossG_list.npy'))
        length = lossG_print.shape[0]

        x = np.linspace(0, length-1, length)
        x = np.asarray(x)
        plt.figure()
        plt.plot(x, lossD_print,label=' lossD',linewidth=1.5)
        plt.plot(x, lossG_print,label=' lossG',linewidth=1.5)

        plt.legend(loc='upper right')
        plt.xlabel('data')
        plt.ylabel('loss')

        log_path = init_path + "/log"
        creat_directory(log_path)
        plt.savefig(os.path.join(log_path, f'lr={lr_g}_epoch={epochs}.png'))

    if is_sample == 1:
        batch_size = 8
        nz = 100
        n_bars = 7
        X_te = np.load('./dataset/augmented/X_te.npy')
        prev_X_te = np.load('./dataset/augmented/prev_X_te.npy')
        prev_X_te = prev_X_te[:,:,check_range_st:check_range_ed,:]
        y_te    = np.load('./dataset/augmented/Y_te.npy')
       
        test_iter = get_dataloader(X_te,prev_X_te,y_te)
        kwargs = {'num_workers': 4, 'pin_memory': True}# if args.cuda else {}
        test_loader = DataLoader(test_iter, batch_size=batch_size, shuffle=False, **kwargs)

        netG = sample_generator()
        netG.load_state_dict(torch.load(os.path.join(model_path,'netG_epoch_70.pth'))) ###

        output_songs = []
        output_chords = []
        for i, (data,prev_data,chord) in enumerate(test_loader, 0):
            list_song = []
            first_bar = data[0].view(1,1,16,128)
            list_song.append(first_bar)

            list_chord = []
            first_chord = chord[0].view(1,13).numpy()
            list_chord.append(first_chord)
            noise = torch.randn(batch_size, nz)

            for bar in range(n_bars):
                z = noise[bar].view(1,nz)
                y = chord[bar].view(1,13)
                if bar == 0:
                    prev = data[0].view(1,1,16,128)
                else:
                    prev = list_song[bar-1].view(1,1,16,128)
                sample = netG(z, prev, y, 1,pitch_range)
                # print(sample.shape)
                list_song.append(sample.detach().cpu())
                list_chord.append(y.numpy())
            # if len(list_song) != 8:
            #     print(len(list_song), [x.shape for x in list_song])
            print('num of output_songs: {}'.format(len(output_songs)))
            output_songs.append([bar.numpy() for bar in list_song])
            output_chords.append(list_chord)
        # output_chords_tensor = torch.tensor(output_chords)

        output_path = init_path + "/output"
        creat_directory(output_path)
        np.save(os.path.join(output_path, 'output_songs.npy'),output_songs)
        np.save(os.path.join(output_path, 'output_chords.npy'),output_chords)

        print('creation completed, check out what I make!')


if __name__ == "__main__" :

    main()


"""
To-do:
- Train model 3: 
    - with augmented and not augmented 
        - train stable or not, how is convergence, equilibrium, loss
        - compare sample, listen,...
    - save more models at each 5 or 10 epochs, compare model at each epochs. (2 model same epochs, 1 models different epochs)
"""