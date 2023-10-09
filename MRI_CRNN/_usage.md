# Readme

- save path
    - ./models/self.algo

- run scripts
    ```bash
        # train
        python /rds/general/user/xc2322/home/git_projects/MRI_CRNN/main_crnn.py --model_name='0729_crnn_single' --dataset_index=1 

        # test
            # checkpoints save: aiden_utils.py, add to this ckpt_dict
            # default test patient: P001
        
        python /rds/general/user/xc2322/home/git_projects/MRI_CRNN/main_crnn.py --mode='test' --ckpt_str='0808_nodcl'


        # debug mode:
            # debug will result in:
            #  train_loader: P001, val_loader: P003 (pass `args.debug` to loader module)
        python main_crnn.py --mode='train' --debug --num_epoch=1000 --model_name='overfit_nodcl_2'

        
    ```
    - dataset_index=1 : single mode, 2: time series mode

- test pre-processing method
    ```bash
    full: 5 * (9*171*72) -> can directly input
    single: 5 * 9* (1 * 171 * 72) -> view and then aggerate

    [5, 2, 171, 72, 9]

    python main_crnn.py --mode='test'
    ```