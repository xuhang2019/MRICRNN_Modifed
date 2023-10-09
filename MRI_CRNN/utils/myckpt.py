
CKPT_DICT = {
    
}

def load_ckpt_dict(ckpt_str):
    assert ckpt_str in CKPT_DICT.keys(), "ckpt_str not in ckpt_dict.keys(), allowed ckpt_str: {}".format(CKPT_DICT.keys())
    
    return CKPT_DICT[ckpt_str]
    