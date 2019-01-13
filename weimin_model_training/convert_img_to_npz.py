import cv2 
import numpy as np 
import os 
from multiprocessing.pool import ThreadPool
from time import time 
import gc

dir = '/local/home/wanweimi/.inclusive/train/train_images/'
save_dir = '/local/home/wanweimi/.inclusive/img_to_npz/'

all_img = os.listdir(dir)
all_img = sorted(all_img)

batch_size = 90000

print("Total files: ", len(all_img))

def process_each_image(f):
    temp_img = cv2.imread(dir + f)
    temp_img = cv2.cvtColor(temp_img, cv2.COLOR_BGR2RGB)
    if temp_img.shape[:2] != (299,299):
        temp_img = cv2.resize(temp_img, (299,299))
    return np.expand_dims(temp_img, 0)


for start_ind in range(0, len(all_img), batch_size):
    print("Start index for this batch: ", start_ind)

    save_file_name = save_dir + 'saved_image_npz_start_%d' % start_ind
    p = ThreadPool(8)

    now = time()
    res = p.map(process_each_image, all_img[start_ind:start_ind+batch_size])
    print("Done. Total seconds: ", time() - now)

    res = np.concatenate(res, 0)

    np.save(save_file_name, res)

    p.close()
    p.join()
    del res 
    gc.collect()

np.save('/local/home/wanweimi/.inclusive/img_to_npz/img_file_name_in_order', np.array(all_img))

print("\ndone")




