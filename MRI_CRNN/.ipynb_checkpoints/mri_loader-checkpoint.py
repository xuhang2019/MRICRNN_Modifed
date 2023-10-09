from pathlib import Path
from glob import glob
import random
import numpy as np
import yaml
import torch

from torch.utils.data import Dataset, DataLoader, random_split
from cascadenet_pytorch.dnn_io import to_tensor_format

from aiden_utils import load_mat, KCrop


# FIXME: lazy load and caching
class MRIMappingDataset(Dataset):
    def __init__(self, args, transform = None):
        super(MRIMappingDataset, self).__init__()
        self.args = args
        
        self.debug = args.debug
        
        self.data_dir = Path('/rds/general/user/xc2322/home/git_projects/data/SingleCoil')
        self.acc_f_dir = self.data_dir / 'AccFactor04'
        self.mask_dir = self.acc_f_dir /'P001'/ 'T1map_mask.mat'
        self.full_dir = self.data_dir / 'FullSample'
        
        self.normalize = KCrop()
        self.mask = None
        
        self.transform = transform
        
        self.x_paths = []
        self.y_paths = []
        
        self.setup_paths()
        self.setup_mask()
             
    def setup_paths(self):
        # acc_f_list = list(sorted(self.acc_f_dir.glob('P*')))
        # full_list = list(sorted(self.full_dir.glob('P*')))
        
        # exclude P002
        acc_f_list = [dir for dir in sorted(self.acc_f_dir.glob('P*')) if dir.name != 'P002']
        full_list = [dir for dir in sorted(self.full_dir.glob('P*')) if dir.name != 'P002']
        
        assert len(acc_f_list) == len(full_list), "self.acc_f_dir and self.full_dir should have the same length"
        
        for dir in acc_f_list:
            dir_name = dir.name
            x = dir / 'T1map.mat'
            y = self.full_dir / dir_name / 'T1map.mat'
            
            self.x_paths.append(x)
            self.y_paths.append(y)
            
            # in debug mode, only load one sample
            if self.debug:
                break
            
        # can be 5, 6, 7 slices
        self.cumulative_slices = [0]
        for x_path in self.x_paths:
            x = load_mat(x_path, complex=True)
            self.cumulative_slices.append(self.cumulative_slices[-1] + x.shape[1])

    def setup_mask(self):
        self.mask = self.normalize(load_mat(self.mask_dir, complex=False), is_mask=True)
        self.mask_tile = np.tile(self.mask, (9,1,1))
        self.mask_tensor_format = to_tensor_format(self.mask_tile, mask=True)
        
    def __len__(self):
        return self.cumulative_slices[-1]
    
    def __getitem__(self, idx):
        # FIXME: single P002 error upload
        sample_idx = next(i for i, count in enumerate(self.cumulative_slices) if count > idx) - 1
        slice_idx = idx - self.cumulative_slices[sample_idx]
        
        # 5, 9, 171, 72 (single mat)
        # -> should convert to 9, 171, 72
        x = load_mat(self.x_paths[sample_idx], complex=True)
        y = load_mat(self.y_paths[sample_idx], complex=True)
        
        x = x[:,slice_idx,...]
        y = y[:,slice_idx,...]
        
        if self.transform:
            x = self.transform(x)
            y = self.transform(y)
        
        im_x, k_x = self.normalize(x)
        im_y, k_y = self.normalize(y)

        im_x = to_tensor_format(im_x)
        im_y = to_tensor_format(im_y)
        k_x = to_tensor_format(k_x)
        k_y = to_tensor_format(k_y)
        
        return (im_x, im_y, k_x, k_y)
    
    
class MRIMappingSingleTimeDataset(MRIMappingDataset):
    def __init__(self, args):
        self.x_slices = []
        self.y_slices = []
        super(MRIMappingSingleTimeDataset, self).__init__(args=args)
                
    def setup_mask(self):
        self.mask = self.normalize(load_mat(self.mask_dir, complex=False), is_mask=True)
        self.mask_tile = np.tile(self.mask, (1,1,1))
        self.mask_tensor_format = to_tensor_format(self.mask_tile, mask=True)  
        
    def setup_paths(self):
        #TODO: seperate P002 patient.
        acc_f_list = [dir for dir in sorted(self.acc_f_dir.glob('P*')) if dir.name != 'P002']
        full_list = [dir for dir in sorted(self.full_dir.glob('P*')) if dir.name != 'P002']
        assert len(acc_f_list) == len(full_list), "self.acc_f_dir and self.full_dir should have the same length"
        
        for dir in acc_f_list:
            dir_name = dir.name
            x_path = dir / 'T1map.mat'
            y_path = self.full_dir / dir_name / 'T1map.mat'
            
            x_data = load_mat(x_path, complex=True)
            num_blocks = x_data.shape[1]  # This can be 5, 6, 7, etc.
            
            for block in range(num_blocks):
                for slice in range(9):  # Assuming 9 slices per block
                    self.x_paths.append(x_path)
                    self.y_paths.append(y_path)
                    self.x_slices.append((block, slice))
                    self.y_slices.append((block, slice))
        
    def __len__(self):
        return len(self.x_paths)
    
    def __getitem__(self, idx):
        # Assuming x is loaded as (5, 9, 171, 72) or (6, 9, 171, 72) or similar
        x = load_mat(self.x_paths[idx], complex=True)
        y = load_mat(self.y_paths[idx], complex=True)
        
        block_idx, slice_idx = self.x_slices[idx]
        
        x = x[slice_idx,block_idx,...]
        y = y[slice_idx,block_idx,...]
        
        x = x[np.newaxis,...]
        y = y[np.newaxis,...]
        
        if self.transform:
            x = self.transform(x)
            y = self.transform(y)
        
        im_x, k_x = self.normalize(x)
        im_y, k_y = self.normalize(y)
        
        im_x = to_tensor_format(im_x)
        im_y = to_tensor_format(im_y)
        k_x = to_tensor_format(k_x)
        k_y = to_tensor_format(k_y)
        
        return (im_x, im_y, k_x, k_y)
    

def get_splited_loader_and_mask(dataset=None, ratio=0.8, args=None):        
    if dataset == 1:
        dataset = MRIMappingSingleTimeDataset(args = args)
    elif dataset == 2 or dataset is None:
        dataset = MRIMappingDataset(args = args)
    
    batch_size = 1

    train_len = int(ratio * len(dataset))  # 80% of the data
    test_len = len(dataset) - train_len
    train_dataset, test_dataset = random_split(dataset, [train_len, test_len])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # process the mask to corresponding size of the input data
    mask_mat = dataset.mask_tensor_format
    mask_tensor = torch.from_numpy(mask_mat)
    mask_tensor = mask_tensor.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)
    
    return train_loader, test_loader, mask_tensor

if __name__ == "__main__":
    # D2 = MRIMappingSingleTimeDataset()
    # tr, te, m = get_splited_loader_and_mask(dataset=D2)
    # print(m.shape)
    class Args:
        def __init__(self, debug):
            self.debug = debug
            
    args = Args(debug=True)
    
    D = MRIMappingDataset(args)
    # load a sample from the dataset
    sample = D[0]

    print('Sample loaded')
    print('end')
    