## TODO: 20250206

1. [**YX ZL finished**] How to let the dataloader's suffle same every time
   1. fit random seed
   2. [YX Done][TD]set randomsamplerwithresume

2. [**YX finished**] Save the models ckpt, optimizer, loss, and run evaluation every 1000 steps. (For batch_size=2, totally 4700 steps, do not run the #4000 step saving and evaluation)
   1. [YX Done][TD] split the save_ckpt without log_metric

3. [**YX finished**] When training process terminate by some errors, run the step 2 and exit.
   1. [YX Done][TD] Change with 2 accordingly

4. [**YX finished**] When step 3 happened and exist, rerun the process by skipping previous step to last saved before terminate.
   1. Need to save the step when terminate
   2. Think about how to skip the previous step by not calling getitem in dataset
   [TD] See 1.

4.1 Check optim, and maybe save pred_bank at /data/zucksliu/somewhere/

5. [Final goal for Classifier]: using 2D model train and predict on classifier and evaluate it more frequently.


About threshold analysis:
1. Add one more figure that shows vessel probability (artery + vein)
2. vessel_threshold=0.k (k=1,3,5,7), we get all pixels that with vessel prob > k, say it's v_k, and then plot v_a / v_k and v_v / v_k (here we know v_a + v_v = v_k)


## TODO 20250207

1. CPR repo implement
2. threshold analysis: vessel_threshold=0.k (k=1,3,5,7), we get all pixels that with vessel prob > k, say it's v_k, and then plot v_a / v_k and v_v / v_k (here we know v_a + v_v = v_k)
3. patient level training debug on OOM
4. Laterality
5. inverse homography