
data_dir='/projects/chimera/zucksliu/AI-READI-2.0/dataset/'
# resume_ckpt_path='/m-ent1/ent1/zucksliu/SFDA-CBMT_results/20250220_172011_image_level_merge_label_no_disc/after_adaptation.pth.tar'
file_name='Training_image_level_model_pseudo'
# save_root='/data/zucksliu/SFDA-CBMT_results/'

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