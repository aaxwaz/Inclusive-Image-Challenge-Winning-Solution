# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'


from a00_common_functions import *
import random
from scipy.sparse import *


def get_classes_for_images_dict():
    cache_file = OUTPUT_PATH + 'overall_classes_dict.pklz'

    if not os.path.isfile(cache_file):
        index_arr_forward, index_arr_backward = get_classes_to_index_dicts()
        d1, d2 = get_classes_dict('single')
        res = dict()
        limit = 1000000000000

        s1 = pd.read_csv(INPUT_PATH + 'train_human_labels.csv')
        total = 0
        print('Go 1')
        ids = list(s1['ImageID'].values)
        lbls = s1['LabelName'].values
        confs = s1['Confidence'].values
        for i, id in enumerate(ids):
            label = lbls[i]
            conf = confs[i]
            if id not in res:
                res[id] = dict()

            if label not in d1:
                continue

            if index_arr_forward[label] not in res[id]:
                if 0:
                    print('Strange!', id, index_arr_forward[label])
                res[id][index_arr_forward[label]] = conf

            total += 1
            if total > limit:
                break

        s2 = pd.read_csv(INPUT_PATH + 'train_machine_labels.csv')
        total = 0
        print('Go 2')
        ids = list(s2['ImageID'].values)
        lbls = s2['LabelName'].values
        confs = s2['Confidence'].values
        for i, id in enumerate(ids):
            label = lbls[i]
            conf = confs[i]
            if id not in res:
                res[id] = dict()

            if label not in d1:
                continue

            if index_arr_forward[label] not in res[id]:
                # print('Strange!', id, index_arr_forward[label])
                res[id][index_arr_forward[label]] = conf

            total += 1
            if total > limit:
                break

        s3 = pd.read_csv(INPUT_PATH + 'train_bounding_boxes.csv', low_memory=False)
        total = 0
        print('Go 3')
        ids = list(s3['ImageID'].values)
        lbls = s3['LabelName'].values
        confs = s3['Confidence'].values
        for i, id in enumerate(ids):
            label = lbls[i]
            conf = confs[i]
            if id not in res:
                res[id] = dict()

            if label not in d1:
                continue

            if index_arr_forward[label] not in res[id]:
                # print('Strange!', id, index_arr_forward[label])
                res[id][index_arr_forward[label]] = conf

            total += 1
            if total > limit:
                break

        save_in_file(res, cache_file)
    else:
        res = load_from_file(cache_file)
    return res


def get_classes_for_images_dict_only_human_labels():
    cache_file = OUTPUT_PATH + 'overall_classes_dict_only_human.pklz'

    if not os.path.isfile(cache_file):
        index_arr_forward, index_arr_backward = get_classes_to_index_dicts()
        d1, d2 = get_classes_dict('single')
        res = dict()
        limit = 1000000000000

        s1 = pd.read_csv(INPUT_PATH + 'train_human_labels.csv')
        total = 0
        print('Go 1')
        ids = list(s1['ImageID'].values)
        lbls = s1['LabelName'].values
        confs = s1['Confidence'].values
        for i, id in enumerate(ids):
            label = lbls[i]
            conf = confs[i]
            if id not in res:
                res[id] = dict()

            if label not in d1:
                continue

            if index_arr_forward[label] not in res[id]:
                if 0:
                    print('Strange!', id, index_arr_forward[label])
                res[id][index_arr_forward[label]] = conf

            total += 1
            if total > limit:
                break

        save_in_file(res, cache_file)
    else:
        res = load_from_file(cache_file)
    return res


def get_images_for_classes_dict():
    cache_file = OUTPUT_PATH + 'images_for_classes_dict.pklz'
    if not os.path.isfile(cache_file):
        img_clss_dict = get_classes_for_images_dict()
        res = dict()
        for img in img_clss_dict:
            for clss in img_clss_dict[img]:
                if clss not in res:
                    res[clss] = dict()
                res[clss][img] = 1
        save_in_file(res, cache_file)
    else:
        res = load_from_file(cache_file)

    return res


def print_img_for_classes():
    index_arr_forward, index_arr_backward = get_classes_to_index_dicts()
    d1, d2 = get_classes_dict('single')
    images_for_classes = get_images_for_classes_dict()
    for el in sorted(list(images_for_classes.keys())):
        print(el, index_arr_backward[el], d1[index_arr_backward[el]], len(images_for_classes[el]))


def get_train_valid_split_of_files():
    cache_file = OUTPUT_PATH + 'train_valid_files.pklz'
    if not os.path.isfile(cache_file):
        classes_for_images = get_classes_for_images_dict()
        checker_list = list(classes_for_images.keys())
        no_class_images = set()
        for el in checker_list:
            if len(classes_for_images[el]) == 0:
                no_class_images |= {el}

        images_for_classes = get_images_for_classes_dict()

        train_files = set()
        valid_files = set()

        counts = dict()
        for el in sorted(list(images_for_classes.keys())):
            counts[el] = len(images_for_classes[el])
        counts_sorted = sort_dict_by_values(counts, reverse=False)
        print(counts_sorted)

        for (id, count) in counts_sorted:
            if count < 10:
                image_files = images_for_classes[id]
                for img in image_files:
                    if img not in train_files:
                        train_files |= {img}
                continue

            if count < 50:
                part = count // 10
            elif count < 100:
                part = count // 20
            elif count < 1000:
                part = count // 50
            elif count < 10000:
                part = count // 100
            elif count < 100000:
                part = count // 500
            else:
                part = count // 1000

            image_files = list(images_for_classes[id].keys())
            for img in image_files:
                if img in valid_files:
                    part -= 1

            random.shuffle(image_files)
            for img in image_files:
                if img in valid_files:
                    continue
                if img in train_files:
                    continue
                if part > 0:
                    valid_files |= {img}
                    part -= 1
                else:
                    train_files |= {img}

        print(len(train_files & no_class_images))
        print(len(valid_files & no_class_images))
        train_files |= no_class_images

        print(len(train_files))
        print(len(valid_files))
        print(len(train_files) + len(valid_files))
        print(len(train_files & valid_files))

        train_files = sorted(list(train_files))
        valid_files = sorted(list(valid_files))
        save_in_file((train_files, valid_files), cache_file)
    else:
        train_files, valid_files = load_from_file(cache_file)
    return train_files, valid_files


def get_classes_for_tst_images_dict():
    s1 = pd.read_csv(INPUT_PATH + 'tuning_labels.csv')
    index_arr_forward, index_arr_backward = get_classes_to_index_dicts()
    d1, d2 = get_classes_dict('single')
    res = dict()
    ids = list(s1['image_id'].values)
    lbls = s1['labels'].values

    for i, id in enumerate(ids):
        label = lbls[i]
        if id not in res:
            res[id] = dict()

        arr = label.strip().split(' ')
        for a in arr:
            if index_arr_forward[a] not in res[id]:
                # print('Strange!', id, index_arr_forward[label])
                res[id][index_arr_forward[a]] = 1.0

    # print(len(res))
    # print(res)
    return res


def get_validation_labels_sparse_matrix():
    cache_file = OUTPUT_PATH + 'validation_labels_sparse_matrix.pklz'
    if not os.path.isfile(cache_file):
        train_files, valid_files = get_train_valid_split_of_files()
        image_classes = get_classes_for_images_dict()

        matrix_real = dok_matrix((len(valid_files), 7178), dtype=np.int8)
        for i, id in enumerate(valid_files):
            for el in image_classes[id]:
                matrix_real[i, el] = 1
        save_in_file(matrix_real, cache_file)
    else:
        matrix_real = load_from_file(cache_file)

    return matrix_real


def get_tuning_labels_data_for_validation(box_size=336):
    from a01_neural_nets import preprocess_input
    cache_path = OUTPUT_PATH + 'tuning_labels_data_{}.pklz'.format(box_size)
    if not os.path.isfile(cache_path):
        s1 = pd.read_csv(INPUT_PATH + 'tuning_labels.csv')
        test_image_classes = get_classes_for_tst_images_dict()
        ids = list(s1['image_id'].values)
        images = []
        images_target = get_target_v2(ids, test_image_classes).astype(np.uint8)
        for i, id in enumerate(ids):
            path =  INPUT_PATH + 'stage_1_test_images/' + id + '.jpg'
            img = read_single_image(path)
            img = cv2.resize(img, (box_size, box_size), cv2.INTER_LANCZOS4)
            images.append(img)
        images = np.array(images, dtype=np.float32)
        images = preprocess_input(images)
        save_in_file((images, images_target), cache_path)
    else:
        images, images_target = load_from_file(cache_path)

    return images, images_target


def get_classes_distribution_in_train():
    total = 1743042
    classes = get_images_for_classes_dict()
    res = dict()
    for c in classes:
        fraction = len(classes[c]) / total
        res[c] = fraction
    return res


if __name__ == '__main__':
    get_classes_for_images_dict()
    images_for_classes = get_images_for_classes_dict()
    print_img_for_classes()
    get_train_valid_split_of_files()
    get_classes_for_tst_images_dict()
    get_train_valid_split_of_files()
    get_validation_labels_sparse_matrix()
