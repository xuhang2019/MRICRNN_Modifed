from pathlib import Path

import pytorch_lightning as pl
import torch.optim as optim
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from cascadenet_pytorch.model_pytorch import CRNN_MRI
from cascadenet_pytorch.dnn_io import from_tensor_format
from mri_loader import get_splited_loader_and_mask
from options import arg_parser, load_ckpt_str
from utils.aiden_utils import set_seed
from utils.metric import complex_psnr


class CRNN_PL(pl.LightningModule):
    def __init__(self, mask_t, args):
        """
        :param n_ch: number of channels
        :param nf: number of filters
        :param ks: kernel size
        :param nc: number of iterations
        :param nd: number of CRNN/BCRNN/CNN layers in each iteration
        """
        super(CRNN_PL, self).__init__()
        self.save_hyperparameters(args)
        self.args = args
        
        # use config to determine the model
        self.n_ch=2
        self.nc=5
        self.nd=5
        self.nf=64
        self.ks=3
        self.use_dcl = args.use_dcl # default False
        self.mask_t = mask_t # here to.device() makes no use. 
        
        self.CRNN = CRNN_MRI(self.n_ch, self.nf, self.ks, self.nc, self.nd, self.use_dcl)
        
        self.train_epoch_loss = []
        self.val_epoch_loss = {
            'base_psnr': [],
            'test_psnr': [],
        }
        
    def forward(self,  x, k, m):
        m = m.to(self.device).float() 
        return self.CRNN(x, k, m)
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3, betas=(0.5, 0.999))
        # scheduler = StepLR(optimizer, step_size=50, gamma=0.1)
        return optimizer
    
    def training_step(self, train_batch, batch_idx):
        im_u, im_g, k_u, k_y = train_batch
        if self.args.dataset_index == 1:
            im_u = im_u.permute(0,4,1,2,3).reshape(-1,2,171,72,1)
            k_u = k_u.permute(0,4,1,2,3).reshape(-1,2,171,72,1)
        # check k_y and k_u relationship
        rec = self.forward(im_u, k_u,self.mask_t)
        loss = F.mse_loss(rec, im_g)
        # self.log('train_loss', loss.item())
        self.train_epoch_loss.append(loss.item())
        
        # `return loss` is a must for pl to do backpropagation
        return loss
    
    def on_train_epoch_end(self):
        total_loss = sum(self.train_epoch_loss) / len(self.train_epoch_loss)
        print('total_loss: {:.6f}'.format(total_loss))
        self.log('train/total_loss', total_loss)
        self.train_epoch_loss.clear()
    
    def validation_step(self, val_batch, batch_idx):
        im_u, im_g, k_u, k_y = val_batch
        if self.args.dataset_index == 1:
            im_u = im_u.permute(0,4,1,2,3).reshape(-1,2,171,72,1)
            k_u = k_u.permute(0,4,1,2,3).reshape(-1,2,171,72,1)
        pred = self.forward(im_u,k_u,self.mask_t)
        
        base_psnr, test_psnr = 0.0, 0.0
        for idx, (im_i, und_i, pred_i) in enumerate(zip(from_tensor_format(im_g.detach().cpu().numpy()),from_tensor_format(im_u.detach().cpu().numpy()),from_tensor_format(pred.detach().cpu().numpy()))):
            base_psnr += complex_psnr(im_i, und_i, peak='max')
            test_psnr += complex_psnr(im_i, pred_i, peak='max')

        base_psnr /= (idx+1)
        test_psnr /= (idx+1)
        
        self.val_epoch_loss['base_psnr'].append(base_psnr)
        self.val_epoch_loss['test_psnr'].append(test_psnr)  
        
    def on_validation_epoch_end(self):
        # calculate the average loss in val_epoch_loss
        avg_base_psnr = sum(self.val_epoch_loss['base_psnr']) / len(self.val_epoch_loss['base_psnr'])
        avg_test_psnr = sum(self.val_epoch_loss['test_psnr']) / len(self.val_epoch_loss['test_psnr'])
        print('avg_base_psnr: {:.4f}, avg_test_psnr: {:.4f}'.format(avg_base_psnr, avg_test_psnr))
        # log val_loss to let the monitor know
        self.log('val/base_psnr', avg_base_psnr)
        self.log('val/val_psnr',avg_test_psnr)
        
        self.log('val_poss',avg_test_psnr)
        
        # clear the val_epoch_loss
        self.val_epoch_loss['base_psnr'].clear()
        self.val_epoch_loss['test_psnr'].clear()
        return {'val_loss':avg_test_psnr}        
        
if __name__ == '__main__':
    set_seed(42)
    args = arg_parser()
    
    PROJECT_DIR = Path('./')
    SAVE_DIR = PROJECT_DIR / f'models/{args.model_name}' 
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    
    checkpoint_callback = ModelCheckpoint(filename='{epoch}-{val_loss:.2f}', save_top_k=1, verbose=True, monitor='val_loss', mode='min', every_n_epochs=100)
    
    if not args.debug:
        wandb_logger= WandbLogger(name=args.model_name, project='MRI_CRNN')
    else:
        wandb_logger = None
    
    # model.load_from_checkpoints()
    train_loader, val_loader, test_loader, mask_t = get_splited_loader_and_mask(dataset=args.dataset_index, args = args)
    model = CRNN_PL(mask_t=mask_t, args=args)
    trainer = pl.Trainer(max_epochs=1000, default_root_dir=SAVE_DIR, callbacks=[checkpoint_callback], logger=wandb_logger)
    trainer.fit(model, train_loader, val_loader)
    