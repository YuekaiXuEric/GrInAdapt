image_save_metric=assd
data_dir='/projects/chimera/zucksliu/AI-READI-2.0/dataset/'
resume_ckpt_path='/m-ent1/ent1/zucksliu/SFDA-CBMT_results/20250222_214131Training_image_level_ema0.995_cos_annealing_av_faz_lr8e-5_best_e0step300/checkpoint_epoch0_step300.pth.tar'
file_name=Evaluation_ours_new_3_${image_save_metric}

python eval.py \
        --data-dir ${data_dir} \
        --batch-size 1 \
        --mask_optic_disc True \
        --run_all_success True \
        --gpu '1' \
        --file_name ${file_name} \
        --image_save_metric ${image_save_metric} \
        --resume_ckpt_path ${resume_ckpt_path} \
