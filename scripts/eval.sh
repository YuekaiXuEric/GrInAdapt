image_save_metric=assd
data_dir='/path/to/your/dataset/'
resume_ckpt_path='/path/to/your/checkpoint.pth.tar'
file_name=Evaluation${image_save_metric}

python eval.py \
        --data-dir ${data_dir} \
        --batch-size 1 \
        --mask_optic_disc True \
        --run_all_success True \
        --gpu '0' \
        --file_name ${file_name} \
        --image_save_metric ${image_save_metric} \
        # --resume_ckpt_path ${resume_ckpt_path} \
