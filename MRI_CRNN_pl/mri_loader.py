from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

from cascadenet_pytorch.dnn_io import to_tensor_format
from utils.aiden_utils import load_mat, KCrop, only_gt_x


# force to unify the data type
def to_float32(*arrays):
    return tuple(arr.astype(np.float32) for arr in arrays)

# FIXME: lazy load and caching
class MRIMappingDataset(Dataset):
    def __init__(self, args, transform = None, mode='train'):
        super(MRIMappingDataset, self).__init__()
        self.args = args
        self.mode = mode
        
        self.debug = args.debug
        
        self.data_dir = Path('/rds/general/user/xc2322/home/git_projects/data/SingleCoil')
        self.acc_f_dir = self.data_dir / 'AccFactor04'
        self.mask_dir = self.acc_f_dir /'P001'/ 'T1map_mask.mat'
        self.full_dir = self.data_dir / 'FullSample'
        
        self.normalizor = KCrop(self.args.only_gt)
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
        
        # FIXME: debug mode
        if self.debug and self.mode=='train':
            acc_f_list = acc_f_list[:1]
            full_list = full_list[:1]
        if self.debug and self.mode == 'val':
            acc_f_list = acc_f_list[1:2]
            full_list = full_list[1:2]
        
        assert len(acc_f_list) == len(full_list), "self.acc_f_dir and self.full_dir should have the same length"
        
        for dir in acc_f_list:
            dir_name = dir.name
            x = dir / 'T1map.mat'
            
            # locate y by dir_name, safer than sorting matching
            y = self.full_dir / dir_name / 'T1map.mat' 
            
            self.x_paths.append(x)
            self.y_paths.append(y)
            
            
        # can be 5, 6, 7 slices
        self.cumulative_slices = [0]
        for x_path in self.x_paths:
            x = load_mat(x_path, complex=True)
            self.cumulative_slices.append(self.cumulative_slices[-1] + x.shape[1])

    def setup_mask(self):
        self.mask = self.normalizor(load_mat(self.mask_dir, complex=False), is_mask=True) # numpy
        self.mask_tile = np.tile(self.mask, (9,1,1))
        self.mask_tensor_format = to_tensor_format(self.mask_tile, mask=True) # tensor
        
    def __len__(self):
        return self.cumulative_slices[-1]
    
    def __getitem__(self, idx):
        # FIXME: single P002 error upload
        sample_idx = next(i for i, count in enumerate(self.cumulative_slices) if count > idx) - 1
        slice_idx = idx - self.cumulative_slices[sample_idx]
        
        # 5, 9, 171, 72 (single mat)
        # -> should convert to 9, 171, 72
        if not self.args.only_gt:
            x = load_mat(self.x_paths[sample_idx], complex=True)
            x = x[:,slice_idx,...]
        
        y = load_mat(self.y_paths[sample_idx], complex=True)
        y = y[:,slice_idx,...] 
        
        if self.transform:
            x = self.transform(x)
            y = self.transform(y)
        
        im_y, k_y = self.normalizor(y) # y is k-space data 
        
        if not self.args.only_gt:
            im_x, k_x = self.normalizor(x)  
        else: # only gt mode:
            im_x, k_x = only_gt_x(im_y, self.mask)
            
        im_x = to_tensor_format(im_x)
        im_y = to_tensor_format(im_y)
        k_x = to_tensor_format(k_x)
        k_y = to_tensor_format(k_y)
        
        # default type is np.float64 （model input: float32)
        data_item = to_float32(im_x, im_y, k_x, k_y)
        
        return data_item
    
    
class MRIMappingSingleTimeDataset(MRIMappingDataset):
    def __init__(self, args, mode='train'):
        self.x_slices = []
        self.y_slices = []
        super(MRIMappingSingleTimeDataset, self).__init__(args=args, mode=mode)
        self.only_gt = args.only_gt
                
    def setup_mask(self):
        self.mask = self.normalizor(load_mat(self.mask_dir, complex=False), is_mask=True)
        self.mask_tile = np.tile(self.mask, (1,1,1))
        self.mask_tensor_format = to_tensor_format(self.mask_tile, mask=True)  
        
    def setup_paths(self):
        acc_f_list = [dir for dir in sorted(self.acc_f_dir.glob('P*'))] # dir.name == 'P002'
        full_list = [dir for dir in sorted(self.full_dir.glob('P*'))]
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
        
        im_y, k_y = self.normalizor(y)
        if self.only_gt:
            im_x, k_x = only_gt_x(im_y, self.mask)
        else:
            im_x, k_x = self.normalizor(x)
        
        im_x = to_tensor_format(im_x)
        im_y = to_tensor_format(im_y)
        k_x = to_tensor_format(k_x)
        k_y = to_tensor_format(k_y)
        
        data_item = to_float32(im_x, im_y, k_x, k_y)
        
        return data_item
    

def get_splited_loader_and_mask(dataset=None, ratio=0.8, args=None):  
    """
        Args:
            dataset: 1 for single time dataset, 2 for full dataset
            ratio: train/test ratio
            args: args from main.py
    """      
    batch_size = 1
    
    # FIXME: debug mode, optimize: dataset_index add to args.
    if args.debug:
        train_dataset=MRIMappingDataset(args = args, mode='train') if dataset == 2 else MRIMappingSingleTimeDataset(args = args, mode='train')
        val_dataset = MRIMappingDataset(args = args, mode='val') if dataset == 2 else MRIMappingSingleTimeDataset(args = args, mode='val')
        test_dataset = None
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=32)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=32)
        
        # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        mask_mat = train_dataset.mask_tensor_format
        mask_tensor = torch.from_numpy(mask_mat)
        mask_tensor = mask_tensor.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)
        
        return train_loader, val_loader, None, mask_tensor
        
    
    if dataset == 1:
        dataset = MRIMappingSingleTimeDataset(args = args)
    elif dataset == 2 or dataset is None:
        dataset = MRIMappingDataset(args = args)
        

    train_len = int(ratio * len(dataset))  # 80% of the data
    val_len = int(0.05*len(dataset)) # 5% of the data
    test_len = len(dataset) - train_len - val_len
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_len, val_len, test_len])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader =  DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # process the mask to corresponding size of the input data
    mask_mat = dataset.mask_tensor_format
    mask_tensor = torch.from_numpy(mask_mat)
    mask_tensor = mask_tensor.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)
    
    return train_loader, val_loader, test_loader, mask_tensor

if __name__ == "__main__":
    # D2 = MRIMappingSingleTimeDataset()
    # tr, te, m = get_splited_loader_and_mask(dataset=D2)
    # print(m.shape)
    class Args:
        def __init__(self, debug):
            self.debug = debug
            self.only_gt = True
            
    args = Args(debug=True)
    
    D = MRIMappingDataset(args)
    # load a sample from the dataset
    sample = D[0]

    print('Sample loaded')
    print('end')
    