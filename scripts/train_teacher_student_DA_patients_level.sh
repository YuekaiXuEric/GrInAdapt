
data_dir='/projects/chimera/zucksliu/AI-READI-2.0/dataset/'
resume_ckpt_path='/m-ent1/ent1/zucksliu/SFDA-CBMT_results/20250220_164228_patient_level_merge_label/after_adaptation.pth.tar'

python train_target_ts_patient.py \
        --data-dir ${data_dir} \
        --epoch 3 \
        --model-ema-rate 0.995 \
        --batch-size 1 \
        --mask_optic_disc True \
        --run_all_success True \
        --gpu '0' \
        --resume_ckpt_path ${resume_ckpt_path} \