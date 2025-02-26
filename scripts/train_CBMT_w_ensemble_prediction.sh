data_dir='/path/to/your/dataset/'
resume_ckpt_path='/path/to/your/checkpoint.pth.tar'
fail_image_list='/path/to/your/fail_image_list.csv'
npz_path='/path/to/your/npz_file.npz'
file_name='Training_cbmt_w_ensemble_prediction'

python train_target_ts_pseudo.py \
        --data-dir ${data_dir} \
        --epoch 3 \
        --model-ema-rate 0.995 \
        --batch-size 2 \
        --mask_optic_disc True \
        --run_all_success True \
        --gpu '0' \
        --file_name ${file_name} \
        --lr 1e-4 \
        # --save_root ${save_root} \
        # --annealing-factor 'cos' \
        # --resume_ckpt_path ${resume_ckpt_path} \