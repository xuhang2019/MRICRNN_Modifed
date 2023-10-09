import numpy as np
import h5py
import os
from utils.mymath import ifft2c, fft2c # fft2c and ifft2c

from utils import compressed_sensing as cs
from utils.metric import complex_psnr
from cascadenet_pytorch.dnn_io import to_tensor_format
from cascadenet_pytorch.dnn_io import from_tensor_format


# load mat file from MRI dataset, return a numpy ndarray (complex)
def load_mat(file_path:str, complex=False):
    with h5py.File(file_path, 'r') as f:
        key0 = list(f.keys())[0]
        assert len(list(f.keys())) == 1, "There is more than 1 key in the mat file."
        # print(f.keys())
        # print(f['kspace_single_sub04']) # shape (9, 5, 10, 144, 512)
        try:
            dataset = f[key0][:]
        except KeyError:
            print(f'Key Error, options:{f.keys()}')
    
    if complex:
        dataset = to_complex(dataset)
    return dataset


def to_complex(data: np.ndarray) -> np.ndarray:
    return data['real'] + 1j * data['imag']

def prep_data(patient = 'P001', norm=False):
    acc_factor_04_path = '/rds/general/user/xc2322/home/git_projects/data/SingleCoil/AccFactor04'
    ground_truth_path = '/rds/general/user/xc2322/home/git_projects/data/SingleCoil/FullSample'
    
    k_und = to_complex(load_mat(os.path.join(acc_factor_04_path, patient, 'T1map.mat')))
    # change axis order to 0,1,2,3 -> 0,1,3,2
    k_und = np.transpose(k_und, (1,0,3,2)) 
    im_und = ifft2c(k_und)
    
    # mask
    mask = load_mat(os.path.join(acc_factor_04_path, patient, 'T1map_mask.mat')) # 144 * 512
    mask = mask.transpose(1,0) # 512 * 144
    
    # repeat and expand mask to match the shape of k_und
    b, t, _, _ = k_und.shape
    mask = np.tile(mask[np.newaxis, np.newaxis], (b, t, 1, 1))

    k_grd =  to_complex(load_mat(os.path.join(ground_truth_path, patient, 'T1map.mat')))
    k_grd = np.transpose(k_grd, (1,0,3,2)) # 5 (slice/batch)*9(kt)*512(kx)*144(ky)
    im_grd = ifft2c(k_grd)
    
    # whether use normalization 
    if norm:
        image_undersample, kspace_undersample = normalized(im_und)
        image_ground_truth, kspace_ground_truth = normalized(im_grd)
        mask_crop = crop_cmrx(mask)
        
        image_undersample = to_tensor_format(image_undersample)
        kspace_undersample = to_tensor_format(kspace_undersample)
        image_ground_truth = to_tensor_format(image_ground_truth)
        kspace_ground_truth = to_tensor_format(kspace_ground_truth)
        mask_crop = to_tensor_format(mask_crop, mask=True)
        
        return image_undersample, image_ground_truth, kspace_undersample, kspace_ground_truth, mask_crop # 5,2,171,72,9
    
    else:
        return im_und, im_grd, k_und, k_grd, mask # 5, 9, 171, 72


def normalized(complex_data_im, m_min=0, m_max=0.004):
    # im_x, im_y -> normalized im_x, im_y
    complex_data_im = crop_cmrx(complex_data_im)
    
    phase = np.angle(complex_data_im)
    magnitude = np.abs(complex_data_im)
    normalized_magnitude = (magnitude-m_min) / (m_max - m_min)
    im_crop = normalized_magnitude * np.exp(1j * phase)
    
    # Note: after cropping, you should fft2c img_crop rather than directly cropping k-space
    k_crop = fft2c(im_crop)
    return im_crop, k_crop

def denormalized(complex_data_im, t_mean=1.8283e-04, t_std=1.8676e-04):
    # receive a b, 2, kt, kx, ky (fastMRI pesudo tensor type) -> denormalized b, kt, kx, ky
    ...


# crop function
def crop_cmrx(im):
    # input: kt, kx, ky
    if len(im.shape) >= 3:
        kx, ky = im.shape[-2:]
        im_crop = im[...,kx//3:2*kx//3, ky//4:3*ky//4]
    elif len(im.shape) == 2:
        kx, ky = im.shape
        im_crop = im[kx//3:2*kx//3, ky//4:3*ky//4]
    return im_crop
    
    
class KCrop:
    # only valid for (x,) 171, 72 
    def __init__(self):
        # implement for the dataset. NO use for single data!
        # normalize function for complex data and mask
        self.t_mean = 1.8283e-04
        self.t_std=1.8676e-04
    def __call__(self, k_space, is_mask=False):
        if not is_mask:
            assert isinstance(k_space, np.ndarray), "k_space should be a numpy ndarray"
            
            # transpose to (, kt, kx, ky) for crop
            if len(k_space.shape) == 3:
                k_complex = np.transpose(k_space, (0,2,1)) 
            elif len(k_space.shape) == 2:
                k_complex = np.transpose(k_space, (1,0))

            im_complex = ifft2c(k_complex)
            
            return normalized(im_complex, self.t_mean, self.t_std)
        
        else:
            mask = np.transpose(k_space, (1,0))
            return crop_cmrx(mask)
        
    
# def prep_input(im, acc=4.0):
#     """Undersample the batch, then reformat them into what the network accepts.

#     Parameters
#     ----------
#     gauss_ivar: float - controls the undersampling rate.
#                         higher the value, more undersampling
#     """
#     mask = cs.cartesian_mask(im.shape, acc, sample_n=8)
#     im_und, k_und = cs.undersample(im, mask, centred=False, norm='ortho')
#     im_gnd_l = torch.from_numpy(to_tensor_format(im))
#     im_und_l = torch.from_numpy(to_tensor_format(im_und))
#     k_und_l = torch.from_numpy(to_tensor_format(k_und))
#     mask_l = torch.from_numpy(to_tensor_format(mask, mask=True))

#     return im_und_l, k_und_l, mask_l, im_gnd_l



# using absolute path, if you want to use relative path, please change it
CKPT_DICT = {
    '0808_dcl': '/rds/general/user/xc2322/home/git_projects/MRI_CRNN/models/overfit_2/overfit_2_epoch_900.npz',
    '0808_nodcl': '/rds/general/user/xc2322/home/git_projects/MRI_CRNN/models/overfit_nodcl/overfit_nodcl_epoch_900.npz',
}

def load_ckpt_str(ckpt_str):
    assert ckpt_str in CKPT_DICT.keys(), "--ckpt_str error, allowed ckpt_str: {}".format(CKPT_DICT.keys())
    
    return CKPT_DICT[ckpt_str]
    

    
if __name__ == '__main__':
    im_und, im_grd, k_und, k_grd, mask = prep_data(patient='P003', norm=True)
    # breakpoint()
    print(im_und.shape)
    

    

