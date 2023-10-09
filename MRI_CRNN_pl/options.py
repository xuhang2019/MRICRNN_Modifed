import argparse

def arg_parser():
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
    parser.add_argument('--use_dcl', action='store_true', help='debug mode')
    parser.add_argument('--only_gt', action='store_false', help='debug mode')
    args = parser.parse_args()
    return args

CKPT_DICT = {
    '0808_dcl': '/rds/general/user/xc2322/home/git_projects/MRI_CRNN/models/overfit_2/overfit_2_epoch_900.npz',
    '0808_nodcl': '/rds/general/user/xc2322/home/git_projects/MRI_CRNN/models/overfit_nodcl/overfit_nodcl_epoch_900.npz',
}

def load_ckpt_str(ckpt_str):
    assert ckpt_str in CKPT_DICT.keys(), "--ckpt_str error, allowed ckpt_str: {}".format(CKPT_DICT.keys())
    
    return CKPT_DICT[ckpt_str]

