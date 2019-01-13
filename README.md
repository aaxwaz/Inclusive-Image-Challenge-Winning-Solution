3rd place winning solution for [Inclusive Images Challenge (NIPS Competition track 2018)](https://www.kaggle.com/c/inclusive-images-challenge)
================================
This repo documents the code to reproduce team `WorldWideIncludive` winning model. The model scored `0.37102` (3rd place) on final stage 2 Leader Board. 

For inference, please refer to `Stage 2 Inference Pipeline` 

For training, please refer to `Training pipeline part 1` and `Training pipeline part 2`

### dependencies: 
- python 3.6
- tf: 1.8.0
- keras: 2.2.2
- cv2: 3.4.3
- sklearn: 0.19.1
- spicy: 1.1.0
- pyvips: 2.1.4


### Stage 2 Inference Pipeline - for generating final submission: 

- Change constants in `a00_common_functions.py`:

  - `TEST_IMAGES_PATH` - path to stage 2 test images

  - `IS_TEST` = 1 - for inference

- Run script: `python3 r40_final_inference_submit.py`



### Training pipeline part 1 (zfturbo's part):

- Change constants in `a00_common_functions.py`:

  - `DATASET_PATH` - path to OID images

  - `IS_TRAIN` - 1 - for training

- Run each script one by one:

  - `python3 a02_common_training_structures.py`

  - `python3 inception_resnet_v2/r40_inception_resnet_v2_training_model.py`

  - `python3 inception_resnet_v2/r50_inception_resnet_v2_validation.py`

  - `python3 inception_resnet_v2/r52_validation_with_thr.py`

  - `python3 resnet50_336/r40_resnet50_sh_336_training_model.py`

  - `python3 resnet50_336/r50_resnet50_sh_336_validation.py`

  - `python3 resnet50_336/r52_resnet50_validation_with_thr.py`


### Training pipeline part 2 (weimin's part):

Please note that all scripts related to weimin's model training are under directory `../weimin_model_training/*`: 

1) Change constants in `a00_common_functions.py`: 

  - `DATASET_PATH`: path to OID images - must be a directory that contains one and only one subdir (any name), that contains all training images
  - `ROOT_PATH`: root directory that has the input folder where all competition data csv files stay
  - `TUNING_IMAGE_PATH`: absolute path pattern that finds all 1000 tuning label images 

2) Run each script one by one:

- `python weimin_main_train_inception_resnet_0.1.py --train_new_model=True`

- `python weimin_main_train_inception_resnet_0.08.py --train_new_model=True` 

- `python weimin_main_train_inception_resnet_0.05.py --train_new_model=True` 

- `python weimin_main_train_inception_resnet_0.15.py --train_new_model=True` 

- `python weimin_main_train_xception.py --train_new_model=True` 


- `python weimin_validation_data_gen_ext_0.py`

- `python weimin_validation_data_gen_ext_1.py`

- `python weimin_validation_data_gen_ext_2.py`

- `python weimin_validation_data_gen_ext_3.py`

- `python weimin_validation_data_gen_ext_4.py`

- `python weimin_validation_data_gen_ext_10.py`

3) Collapse - At the end of training you will have models and relevant threshold arrays.

- weimin's models and threshold paths are stored below: 

  - `ROOT_PATH + 'model_0/inception_resnet_v2_latest.h5'`

  - `ROOT_PATH + 'model_1/inception_resnet_v2_latest.h5'`

  - `ROOT_PATH + 'model_2/inception_resnet_v2_latest.h5'`

  - `ROOT_PATH + 'model_3/inception_resnet_v2_latest.h5'`

  - `ROOT_PATH + 'model_4/new_xception_latest.h5'`

  - `ROOT_PATH + 'modified_data/thr_arr_inception_resnet_version_1_sp_0.1_ep_0.9_min_1_def_0.9.pklz'`

  - `ROOT_PATH + 'modified_data/thr_arr_inception_resnet_version_2_sp_0.1_ep_0.9_min_3_def_0.9.pklz'`

  - `ROOT_PATH + 'modified_data/thr_arr_inception_resnet_weimin_version_3_sp_0.01_ep_0.99_min_1_def_0.99.pklz'`

  - `ROOT_PATH + 'modified_data/thr_arr_inception_resnet_weimin_version_4_sp_0.01_ep_0.99_min_1_def_0.99.pklz'`

  - `ROOT_PATH + 'modified_data/thr_arr_xception_sp_0.01_ep_0.99_min_1_def_0.99.pklz'`

  - `ROOT_PATH + 'modified_data/thr_arr_xception_sp_0.01_ep_0.99_min_1_def_0.9999.pklz' # used for indexing only`