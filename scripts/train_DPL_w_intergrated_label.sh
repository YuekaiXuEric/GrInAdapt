
data_dir='/path/to/your/dataset/'
resume_ckpt_path='/path/to/your/checkpoint.pth.tar'
fail_image_list='/path/to/your/fail_image_list.csv'
file_name='Training_image_level_model_cos_annealing_av_faz_lr8e-5_noise_0.05_0.1'

python train_target_single_model.py \
        --data-dir ${data_dir} \
        --epoch 3 \
        --batch-size 3 \
        --mask_optic_disc True \
        --run_all_success True \
        --gpu '3' \
        --file_name ${file_name} \
        --lr 8e-5 \
        # --resume-ckpt-path ${resume_ckpt_path} \