# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo, Weimin: https://kaggle.com/weimin'

if __name__ == '__main__':
    import os

    gpu_use = (0,1)
    print('GPU use: {}'.format(gpu_use))
    os.environ["KERAS_BACKEND"] = "tensorflow"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

from a01_neural_nets import *
from a02_common_training_structures import *
from keras.applications.xception import preprocess_input as xception_preprocess_input
from keras.applications.inception_resnet_v2 import preprocess_input as inception_resnet_preprocess_input
import sys 

PRECISION = 6
EPS = 0.00001

def readLines(f):
    with open(f, 'r') as f:
        return f.read().split('\n')

def process_tst_images(model_path, box_size, cache_path_test, preproc, is_weights=False, model_name=None):
    from keras.models import load_model
    from keras.optimizers import Adam
    from keras.utils import multi_gpu_model

    if not os.path.isdir(cache_path_test):
        os.mkdir(cache_path_test)

    restore_from_cache = True
    if is_weights:
        if model_name == 'zfturbo_inception_resnet':
            model = get_model_inception_resnet_v2()
            model.load_weights(model_path)

        elif model_name == 'zfturbo_resnet':
            model = get_model_resnet50_336()
            model.load_weights(model_path)

        elif model_name == 'weimin_inception_resnet':
            model = get_model_inception_resnet_v2()
            optim = Adam(lr=0.001)
            model = multi_gpu_model(model, gpus=2)
            model.compile(optimizer=optim, loss='binary_crossentropy', metrics=[f2beta_loss, fbeta])
            model.load_weights(model_path)

        elif model_name == 'weimin_xception':
            model=get_model_xception() 
            optim = Adam(lr=0.001)
            model = multi_gpu_model(model, gpus=2)
            model.compile(optimizer=optim, loss='binary_crossentropy', metrics=[f2beta_loss, fbeta])
            model.load_weights(model_path)

        else:
            print("Unknown model_name: %s"%model_name)
            return None
        print("\nUsing weights to load model: \n%s loaded."%model_name)
    else:
        model = load_model(model_path, custom_objects={'f2beta_loss': f2beta_loss, 'fbeta': fbeta})
        print("\nUsing model architecture .h5 to load model: \n%s loaded."%model_name)

    test_files = glob.glob(TEST_IMAGES_PATH + '*.jpg')
    print('Number of images to predict:', len(test_files))

    for i, f in enumerate(sorted(test_files)):
        id = os.path.basename(f)[:-4]
        path = f

        cache_path = cache_path_test + id + '.pkl'
        if not os.path.isfile(cache_path) or restore_from_cache is False:
            im_full_big = read_single_image(path)
            im_full_big = cv2.resize(im_full_big, (box_size, box_size), cv2.INTER_LANCZOS4)

            batch_images = []
            batch_images.append(im_full_big.copy())
            batch_images.append(im_full_big[:, ::-1, :].copy())
            batch_images = np.array(batch_images, dtype=np.float32)
            batch_images = preproc(batch_images)

            preds = model.predict(batch_images)
            preds[preds < EPS] = 0
            preds = np.round(preds, PRECISION)
            save_in_file(preds, cache_path)

def create_submission(folder_path, thr_arr, out_file):
    index_arr_forward, index_arr_backward = get_classes_to_index_dicts()
    out = open(out_file, 'w')
    out.write('image_id,labels\n')
    test_files = glob.glob(TEST_IMAGES_PATH + '*.jpg')
    #print('Test valid:', len(test_files))
    for i, f in enumerate(test_files):
        id = os.path.basename(f)[:-4]
        cache_path = folder_path + id + '.pkl'
        preds = load_from_file(cache_path)
        preds = preds.mean(axis=0)
        total = 0
        out.write(id + ',')
        for j in range(len(preds)):
            if preds[j] >= thr_arr[j]:
                out.write(index_arr_backward[j] + ' ')
                total += 1
        out.write('\n')
        # print('Go for {} {}. Stored: {}'.format(i, id, total))
    out.close()

def voting_ensemble(subm_arr, out_path):
    s = pd.read_csv(subm_arr[0])
    s['labels_0'] = s['labels']
    #print(len(s))
    for i in range(1, len(subm_arr)):
        s1 = pd.read_csv(subm_arr[i])
        s1['labels_{}'.format(i)] = s1['labels']
        s = s.merge(s1[['image_id', 'labels_{}'.format(i)]], on='image_id', how='left')
        #print(s1.shape)
    #print(len(s))

    ## majority vote
    limit = len(subm_arr) // 2

    print('\nmajority vote num threshold: {}\n'.format(limit))
    merged_labels = []
    for index, row in s.iterrows():
        res = dict()
        for i in range(len(subm_arr)):
            line = row['labels_{}'.format(i)]
            if str(line) == 'nan':
                continue
            arr = line.strip().split(' ')
            for a in arr:
                if a in res:
                    res[a] += 1
                else:
                    res[a] = 1

        out_str = ''
        for el in res:
            if res[el] > limit:
                out_str += el + ' '

        # union for empty rows
        if out_str == '':
            temp_res = []
            for i in range(len(subm_arr)):
                if str(row['labels_{}'.format(i)]) != 'nan':
                    for i in row['labels_{}'.format(i)].strip().split():
                        temp_res.append(i)
            temp_res = list(set(temp_res))
            out_str = ' '.join(temp_res)

        merged_labels.append(out_str)
    s['labels'] = merged_labels

    s[['image_id', 'labels']].to_csv(out_path, index=False)

    return s[['image_id', 'labels']]

def voting_ensemble_mark_union(subm_arr, out_path):
    s = pd.read_csv(subm_arr[0])
    s['labels_0'] = s['labels']
    #print(len(s))
    for i in range(1, len(subm_arr)):
        s1 = pd.read_csv(subm_arr[i])
        s1['labels_{}'.format(i)] = s1['labels']
        s = s.merge(s1[['image_id', 'labels_{}'.format(i)]], on='image_id', how='left')
        #print(s1.shape)
    #print(len(s))

    ## majority vote
    limit = len(subm_arr) // 2

    print('\nmajority vote num threshold: {}\n'.format(limit))
    merged_labels = []
    use_union = []
    for index, row in s.iterrows():
        res = dict()
        for i in range(len(subm_arr)):
            line = row['labels_{}'.format(i)]
            if str(line) == 'nan':
                continue
            arr = line.strip().split(' ')
            for a in arr:
                if a in res:
                    res[a] += 1
                else:
                    res[a] = 1

        out_str = ''
        for el in res:
            if res[el] > limit:
                out_str += el + ' '

        # union for empty rows
        if out_str == '':
            use_union.append(1)
            temp_res = []
            for i in range(len(subm_arr)):
                if str(row['labels_{}'.format(i)]) != 'nan':
                    for i in row['labels_{}'.format(i)].strip().split():
                        temp_res.append(i)
            temp_res = list(set(temp_res))
            out_str = ' '.join(temp_res)
        else:
            use_union.append(0)

        merged_labels.append(out_str)
    s['labels'] = merged_labels
    s['use_union'] = use_union

    s[['image_id', 'labels', 'use_union']].to_csv(out_path, index=False)

    return s[['image_id', 'labels', 'use_union']]


def fix_empty_rows(in_subm, out_subm):
    s1 = pd.read_csv(in_subm)

    s1_ids = s1['image_id'].values
    s1_lbls = s1['labels'].values

    new_ids = []
    new_lbls = []
    total_empty = 0 
    for i in range(len(s1_ids)):
        id1 = s1_ids[i]
        lbl = s1_lbls[i]
        if str(lbl) == 'nan':
            total_empty += 1
            lbl = '/m/01g317 /m/05s2s /m/07j7r'

        new_ids.append(id1)
        new_lbls.append(lbl)

    t = pd.DataFrame({'image_id': new_ids, 'labels': new_lbls})
    t.to_csv(out_subm, index=False)
    print("Total empty rows: {}".format(total_empty))



if __name__ == '__main__':
    start_time = time.time()

    ## params to search in the for loop below 
    ## model / model weights file name,                  cache folder,                              threshold file name prefix,                   threshold points,     output file names,                img_dim, is_weights, preprocessing_fun,           model architecture string        
    params = [
        ['inception_resnet_v2_temp_320_weights.h5',       'cache_inception_resnet_v2_test/',        'thr_arr_inception_resnet_v2',                (0.01, 0.99, 1, 0.99),  'inception_resnet_v2_299.csv',    (299, True,  preprocess_input,                  'zfturbo_inception_resnet')], 
        ['resnet50_336_temp_488_weights.h5',              'cache_resnet50_test/',                   'thr_arr_resnet50_sh_336',                    (0.01, 0.99, 1, 0.99),  'resnet50_336.csv',               (336, True,  preprocess_input,                   'zfturbo_resnet')], 
        ['weights_new_xception_tlatest.h5',               'cache_xception_test/',                   'thr_arr_xception',                           (0.01, 0.99, 1, 0.99),  'xception.csv',                   (299, True,  xception_preprocess_input,          'weimin_xception')], 
        ['inception_resnet_model_oct_19_0.001_48_0.1.h5', 'cache_inception_resnet_version_1_test/', 'thr_arr_inception_resnet_version_1',         (0.01, 0.99, 1, 0.99),  'inception_resnet_version_1.csv', (299, True,  inception_resnet_preprocess_input,  'weimin_inception_resnet')], 
        ['inception_resnet_model_5.h5',                   'cache_inception_resnet_version_2_test/', 'thr_arr_inception_resnet_version_2',         (0.01, 0.99, 1, 0.99),  'inception_resnet_version_2.csv', (299, True,  inception_resnet_preprocess_input,  'weimin_inception_resnet')], 
        ['inception_resnet_model_Oct_31_v3_weights.h5',   'cache_inception_resnet_version_3_test/', 'thr_arr_inception_resnet_weimin_version_3',  (0.01, 0.99, 1, 0.99),  'inception_resnet_version_3.csv', (299, True,  inception_resnet_preprocess_input,  'weimin_inception_resnet')], 
        ['inception_resnet_weimin_version_4_weights.h5',  'cache_inception_resnet_version_4_test/', 'thr_arr_inception_resnet_weimin_version_4',  (0.01, 0.99, 1, 0.99),  'inception_resnet_version_4.csv', (299, True,  inception_resnet_preprocess_input,  'weimin_inception_resnet')],
    ]

    ## loop through each model to generate prediction on one set of threshold - this has to be run at once (just to generate those image preds cache for each model), in order to proceed for random search later on
    ## get tuning label set for evaludation 
    #test_image_classes = get_classes_for_tst_images_dict()
    #index_arr_forward, index_arr_backward = get_classes_to_index_dicts()
    
    ## get the index for special classes that don't exist in tuning label set 
    thr_0_9999 = load_from_file(OUTPUT_PATH+'thr_arr_xception_sp_0.01_ep_0.99_min_1_def_0.9999.pklz')
    col_index_9999 = np.where(thr_0_9999==0.9999)[0]
    
    subm_arr = []
    for ind, model_param in enumerate(params):
        temp_models_path, temp_output_path, thre_prefix, thre_points, subm_path, process_params = model_param

        model_path = MODELS_PATH + temp_models_path
        cache_path = OUTPUT_PATH + temp_output_path
        start_point, end_point, min_number_of_entries, default_value = thre_points
        thr_path = OUTPUT_PATH + thre_prefix + '_sp_{}_ep_{}_min_{}_def_{}.pklz'.format(start_point, end_point,
                                                                                        min_number_of_entries,
                                                                                        default_value)
        # for the three special models to be in range of 0.1-0.9
        if temp_models_path == 'resnet50_336_temp_488_weights.h5':
            thr_path = OUTPUT_PATH + thre_prefix + '_sp_{}_ep_{}_min_{}_def_{}.pklz'.format(0.1, 0.9,
                                                                                            1,
                                                                                            0.9) # 0.8
        if temp_models_path == 'inception_resnet_model_Oct_31_v3_weights.h5':
            thr_path = OUTPUT_PATH + thre_prefix + '_sp_{}_ep_{}_min_{}_def_{}.pklz'.format(0.1, 0.9,
                                                                                            1,
                                                                                            0.9) # 0.8
        if temp_models_path == 'inception_resnet_weimin_version_4_weights.h5':
            thr_path = OUTPUT_PATH + thre_prefix + '_sp_{}_ep_{}_min_{}_def_{}.pklz'.format(0.1, 0.9,
                                                                                            3,
                                                                                            0.9) # 0.9
        #print('\n\n\n', thr_path, '\n\n\n')                                                                                                                     
        
        submit_path = SUBM_PATH + subm_path[:-4] + "_thre_{}_{}_{}_{}.csv".format(start_point, end_point, min_number_of_entries, default_value)
        if temp_models_path == 'resnet50_336_temp_488_weights.h5':
            submit_path = SUBM_PATH + subm_path[:-4] + "_thre_{}_{}_{}_{}.csv".format(0.1, 0.9, 1, 0.8)
        if temp_models_path == 'inception_resnet_model_Oct_31_v3_weights.h5':
            submit_path = SUBM_PATH + subm_path[:-4] + "_thre_{}_{}_{}_{}.csv".format(0.1, 0.9, 1, 0.8)
        if temp_models_path == 'inception_resnet_weimin_version_4_weights.h5':
            submit_path = SUBM_PATH + subm_path[:-4] + "_thre_{}_{}_{}_{}.csv".format(0.1, 0.9, 3, 0.9)
                
        subm_arr.append(submit_path)

        #if os.path.exists(submit_path):
        #    print("File {} already exists. Skipping creating submission for this model. ".format(submit_path))
            ## evaluate LS for this model 
        #    s = pd.read_csv(submit_path)
        #    score, median_counts, mean_counts = evalute_local(s, 'labels', test_image_classes, index_arr_forward)
        #    print("Model {} scores {} median & mean counts of labels: {} {} for tuning label set. \n".format(temp_models_path, score, median_counts, mean_counts))
        #    continue

        process_tst_images(model_path, process_params[0], cache_path, process_params[2], is_weights=process_params[1], model_name=process_params[3])
        
        thr_arr = load_from_file(thr_path)

        ## convert special classes to certain default values  
        if temp_models_path == 'resnet50_336_temp_488_weights.h5':
            thr_arr[col_index_9999] = 0.8 
        if temp_models_path == 'inception_resnet_model_Oct_31_v3_weights.h5':
            thr_arr[col_index_9999] = 0.8
        if temp_models_path == 'inception_resnet_weimin_version_4_weights.h5':
            thr_arr[col_index_9999] = 0.9
                            
        create_submission(cache_path, thr_arr, submit_path)

        ## evaluate LS for this model 
        #s = pd.read_csv(submit_path)
        #score, median_counts, mean_counts = evalute_local(s, 'labels', test_image_classes, index_arr_forward)
        #print("Model {} scores {} median & mean counts of labels: {} {} for tuning label set. \n".format(temp_models_path, score, median_counts, mean_counts))
        #print("Done for model {}\n".format(ind+1))

    # Majority voting part
    majority_voting_submission_path = SUBM_PATH + 'majority_voting_{}_models_v2.2.csv'.format(len(subm_arr))
    s = voting_ensemble(subm_arr, majority_voting_submission_path)

    ## evaludate the result locally 
    #score, median_counts, mean_counts = evalute_local(s, 'labels', test_image_classes, index_arr_forward)
    #print("Score {} median & mean counts of labels: {} {} for tuning label set. ".format(score, median_counts, mean_counts))

    # Fix empty rows
    final_subm_path = SUBM_PATH + 'final_subm_stage_2_{}_models.csv'.format(len(subm_arr))
    fix_empty_rows(majority_voting_submission_path, final_subm_path)

    print('Time: {:.0f} sec'.format(time.time() - start_time))
                                                    
