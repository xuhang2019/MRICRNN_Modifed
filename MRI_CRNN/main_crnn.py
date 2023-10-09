#!/usr/bin/env python
from __future__ import print_function, division

import os
from pathlib import Path
import time
import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt

from os.path import join
from scipy.io import loadmat
import numpy as np
import random

from utils import compressed_sensing as cs
from utils.metric import complex_psnr

from cascadenet_pytorch.model_pytorch import *
from cascadenet_pytorch.dnn_io import to_tensor_format
from cascadenet_pytorch.dnn_io import from_tensor_format

from aiden_utils import prep_data, load_ckpt_str
from mri_loader import get_splited_loader_and_mask
# add tensorboard ... 
from torch.utils.tensorboard import SummaryWriter




def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def prep_input(im, acc=4.0):
    """Undersample the batch, then reformat them into what the network accepts.

    Parameters
    ----------
    gauss_ivar: float - controls the undersampling rate.
                        higher the value, more undersampling
    """
    mask = cs.cartesian_mask(im.shape, acc, sample_n=8)
    im_und, k_und = cs.undersample(im, mask, centred=False, norm='ortho')
    im_gnd_l = torch.from_numpy(to_tensor_format(im))
    im_und_l = torch.from_numpy(to_tensor_format(im_und))
    k_und_l = torch.from_numpy(to_tensor_format(k_und))
    mask_l = torch.from_numpy(to_tensor_format(mask, mask=True))

    return im_und_l, k_und_l, mask_l, im_gnd_l

def prep_tensor(im, mask=False, tensor_type=None):
    t = torch.from_numpy(to_tensor_format(im, mask=mask))
    if tensor_type is not None:
        t = t.type(tensor_type)
    return t


def iterate_minibatch(data, batch_size, shuffle=True):
    n = len(data)

    if shuffle:
        data = np.random.permutation(data)

    for i in range(0, n, batch_size):
        yield data[i:i+batch_size]
        
def iterate_idx(n, shuffle=True):
    idx = np.arange(n)
    if shuffle:
        idx = np.random.permutation(idx)
    for i in idx:
        yield i


def create_dummy_data():
    """Create small cardiac data based on patches for demo.

    Note that in practice, at test time the method will need to be applied to
    the whole volume. In addition, one would need more data to prevent
    overfitting.

    """
    
    data = loadmat(join(project_root, './data/cardiac.mat'))['seq'] # 256,256,30
    nx, ny, nt = data.shape # 256, 256, 3
    ny_red = 8
    sl = ny//ny_red # 256/8 = 32
    data_t = np.transpose(data, (2, 0, 1)) # 30,256,256 
    
    
    
    # Synthesize data by extracting patches
    train = np.array([data_t[..., i:i+sl] for i in np.random.randint(0, sl*3, 20)])
    validate = np.array([data_t[..., i:i+sl] for i in (sl*4, sl*5)])
    test = np.array([data_t[..., i:i+sl] for i in (sl*6, sl*7)])

    return train, validate, test

@torch.no_grad()
def evaluate(model = None, checkpoints= None, patient=None, is_single_time=True):
    print(patient)
    # is_single_time = True
    t_start = time.time()
    
    if checkpoints is not None:
        model.load_state_dict(torch.load(checkpoints))
    model.eval()
    
    im_und, im_grd, k_und, k_grd, mask = prep_data(patient=patient, norm=True)
    im_u = torch.from_numpy(to_tensor_format(im_und))
    im_g = torch.from_numpy(to_tensor_format(im_grd))
    k_u = torch.from_numpy(to_tensor_format(k_und))
    mask_t = torch.from_numpy(to_tensor_format(mask, mask=True))
    
    n_s = im_u.shape[0] # slice
    im_u = im_u.to(device, dtype=torch.float32)
    im_g = im_g.to(device, dtype=torch.float32)
    k_u = k_u.to(device, dtype=torch.float32)
    mask_t = mask_t.to(device, dtype=torch.float32)
    
    if not is_single_time: # full situation 
        # original for loop, FIXME: change to batch instead of one patient
        pred = model(im_u, k_u, mask_t, test=True)
    else:
        n_t = 9 # T1 mapping: 9, T2 mapping: 4
        im_u = im_u.permute(0,4,1,2,3).reshape(-1,2,171,72,1)
        k_u = k_u.permute(0,4,1,2,3).reshape(-1,2,171,72,1)
        mask_t = mask_t.permute(0,4,1,2,3).reshape(-1,2,171,72,1)
        pred = model(im_u, k_u, mask_t, test=True)
        
        # transform to original shape
        im_u = im_u.reshape(n_s, n_t, 2, 171, 72).permute(0, 2, 3, 4, 1)
        pred = pred.reshape(n_s, n_t, 2, 171, 72).permute(0, 2, 3, 4, 1)
        # breakpoint()
    
    base_psnr = 0.0
    test_psnr = 0.0
    
    cnt_b = 0
    for im_i, und_i, pred_i in zip(from_tensor_format(im_g.detach().cpu().numpy()),
                                    from_tensor_format(im_u.detach().cpu().numpy()),from_tensor_format(pred.detach().cpu().numpy())):
        
        base_psnr += complex_psnr(im_i, und_i, peak='max')
        test_psnr += complex_psnr(im_i, pred_i, peak='max')
        cnt_b += 1

    base_psnr /= cnt_b
    test_psnr /= cnt_b
    t_eval = time.time() - t_start
    print(base_psnr, test_psnr, 'using_time: ', t_eval)



@torch.no_grad()
def evaluate_with_loader(model = None,checkpoint=None, loader=None, dataset_index=2):  
    # mask_t and epoch is introduced from the training process
    
    t_start = time.time()
    
    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint))
        
    model.eval()
    
    for idx, (im_u, im_g, k_u, k_y) in enumerate(loader):    
        # set the variable to device
        im_u = im_u.to(device, dtype=torch.float32)
        im_g = im_g.to(device, dtype=torch.float32)
        k_u = k_u.to(device, dtype=torch.float32)
        k_y = k_y.to(device, dtype=torch.float32)
        
        # FIXME: mask_t
        pred = rec_net(im_u, k_u, mask_t, test=False)
        
    base_psnr = 0.0
    test_psnr = 0.0
    
    cnt_b = 0
    #TODO: use torch.Tensor to calculate the psnr, otherwise it is too slow
    for im_i, und_i, pred_i in zip(from_tensor_format(im_g.detach().cpu().numpy()),
                                    from_tensor_format(im_u.detach().cpu().numpy()),from_tensor_format(pred.detach().cpu().numpy())):
        base_psnr += complex_psnr(im_i, und_i, peak='max')
        test_psnr += complex_psnr(im_i, pred_i, peak='max')
        cnt_b += 1

    base_psnr /= cnt_b
    test_psnr /= cnt_b
    
    t_eval = time.time() - t_start
    print('base: ', base_psnr, "test: ", test_psnr, 'using_time: ', t_eval)
    writer.add_scalars('val_psnr', {'base': base_psnr, 'test': test_psnr}, epoch)

    
    

if __name__ == '__main__':
    set_seed(42)
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--num_epoch', metavar='int', nargs=1, default=['100'],
                        help='number of epochs')
    parser.add_argument('--batch_size', metavar='int', nargs=1, default=['1'],
                        help='batch size')
    parser.add_argument('--lr', metavar='float', nargs=1,
                        default=['0.001'], help='initial learning rate') # decay
    parser.add_argument('--acceleration_factor', metavar='float', nargs=1,
                        default=['4.0'],
                        help='Acceleration factor for k-space sampling')
    parser.add_argument('--savefig', action='store_true',
                        help='Save output images and masks')
    parser.add_argument('--dataset_index', type=int, default=2)
    parser.add_argument('--model_name', type=str, default='crnn')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--ckpt_str', type=str, default='')
    parser.add_argument('--debug', action='store_true', help='debug mode')

    args = parser.parse_args()
    
    enable_train = True
    enable_test = False
    
    cuda = True if torch.cuda.is_available() else False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

    # Project config
    model_name = args.model_name
    acc = float(args.acceleration_factor[0])  # undersampling rate
    num_epoch = int(args.num_epoch[0])
    batch_size = int(args.batch_size[0])
    
    Nx, Ny, Nt = 256, 256, 30
    Ny_red = 8
    save_fig = args.savefig
    save_every = 5

    # Configure directory info
    project_root = '.'
    save_dir = join(project_root, 'models/%s' % model_name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    writer = SummaryWriter(join(save_dir,'runs'))
         
    train_loader, val_loader, test_loader, mask_t = get_splited_loader_and_mask(dataset=args.dataset_index, args = args)
    mask_t = mask_t.to(device, dtype=torch.float32)
    

    rec_net = CRNN_MRI(use_dcl=False)
    
    # TODO: std meanm min - max
    # TODO: mask
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(rec_net.parameters(), lr=float(args.lr[0]), betas=(0.5, 0.999))
    scheduler = StepLR(optimizer, step_size=50, gamma=0.1)

    if cuda:
        rec_net = rec_net.cuda()
        criterion.cuda()

    
    if args.mode == 'train':
        i = 0
        for epoch in range(num_epoch):
            t_start = time.time()
            # Training
            train_err = 0
            train_batches = 0
            print("Epoch {}/{}".format(epoch+1, num_epoch))
            
            for idx, (im_u, im_g, k_u, k_y) in enumerate(tqdm(train_loader)):
                
                # set the variable to device
                im_u = im_u.to(device, dtype=torch.float32)
                im_g = im_g.to(device, dtype=torch.float32)
                k_u = k_u.to(device, dtype=torch.float32)
                k_y = k_y.to(device, dtype=torch.float32)
                
                optimizer.zero_grad()
                rec = rec_net(im_u, k_u, mask_t, test=False)
                
                loss = criterion(rec, im_g)
                loss.backward()
                optimizer.step()
                # add scheduler ...

                train_err += loss.item()
                train_batches += 1
            
            scheduler.step()
            
            # define validation

            train_err /= train_batches
            t_end = time.time()
            
            # Then we print the results for this epoch:
            print("Epoch {}/{}".format(epoch+1, num_epoch))
            print(" time: {}s".format(t_end - t_start))

            print(" training loss:\t\t{:.6f}".format(train_err))

            writer.add_scalar('Train/Training loss', train_err, epoch)

            #FIXME: validation rough version
            evaluate_with_loader(rec_net, loader=val_loader)
            
            curr_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('Train/Learning rate', curr_lr, epoch)
            recon_images = from_tensor_format(rec.detach().cpu().numpy())
            recon_images = np.transpose(recon_images, (1, 0, 2, 3))
            writer.add_image('Recon Images', recon_images[0,0], epoch, dataformats='HW')
                        
            print("epoch:",epoch, "Finished!")
            # save the model
            
            
            if (not args.debug) and (epoch % 5 == 0 or epoch in [1, num_epoch-1]):
                name = '%s_epoch_%d.npz' % (model_name, epoch)
                torch.save(rec_net.state_dict(), join(save_dir, name))
                print('model parameters saved at %s' % join(os.getcwd(), name))
                print('')
            
            if args.debug and epoch % 100 == 0:
                name = '%s_epoch_%d.npz' % (model_name, epoch)
                torch.save(rec_net.state_dict(), join(save_dir, name))
                print('model parameters saved at %s' % join(os.getcwd(), name))
                print('')
            
            if epoch in [1, 2, num_epoch-1]:
                if save_fig:
                    if enable_test:
                        for im_i, pred_i, und_i, mask_i in vis:
                            im = abs(np.concatenate([und_i[0], pred_i[0], im_i[0], im_i[0] - pred_i[0]], 1))
                            plt.imsave(join(save_dir, 'im{0}_x.png'.format(i)), im, cmap='gray')

                            im = abs(np.concatenate([und_i[..., 0], pred_i[..., 0],
                                                    im_i[..., 0], im_i[..., 0] - pred_i[..., 0]], 0))
                            plt.imsave(join(save_dir, 'im{0}_t.png'.format(i)), im, cmap='gray')
                            plt.imsave(join(save_dir, 'mask{0}.png'.format(i)),
                            np.fft.fftshift(mask_i[..., 0]), cmap='gray')
                            i += 1

    elif args.mode == 'test':
        evaluate(model=rec_net, checkpoints=load_ckpt_str(args.ckpt_str), patient='P003', is_single_time=False)
        
    writer.close()