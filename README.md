## Installation
* Install python 3.10.5, pytorch 1.12.0, CUDA 11.6 and other essential packages (Note that using other versions of packages may affect performance.)
* Clone this repo
```
conda env create -f requirement.yaml
conda activate grinadapt
```

## Training GrInAdapt

Choose your dataset path and save path.
```
sh GrInAdapt_Adaption/scripts/train_teacher_student_DA_image_level.sh
```

## Training DPL w/ integrated label

Choose your dataset path and save path.
```
sh GrInAdapt_Adaption/scripts/train_target_model_DA_image_level.sh
```

## Training CMBT w/ ensemble prediction

Specify the model path and data path in `./generate_pseudo.py` and then run python `generate_pseudo.py`.

Specify the source model path, data path, and the pseudo label path and run
```
sh GrInAdapt_Adaption/scripts/train_teacher_student_DA_image_level_pseudo.sh
```

