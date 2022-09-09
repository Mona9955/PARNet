import numpy as np
import os
import cv2
import math
from sklearn.utils import shuffle
import random
import config_2d as cfg
from numpy.random import randint
from PIL import Image, ImageEnhance
import threading

max_frames = cfg.MAX_FRAMES
max_points = cfg.MAX_POINTS
max_people = cfg.MAX_PEOPLE
img_size = cfg.IMG_SIZE
img_scale = cfg.IMG_SCALE
img_mean = np.array(cfg.IMG_MEAN, np.float32)
img_std = np.array(cfg.IMG_STD, np.float32)


def image_normalize(image, img_mean=img_mean, img_std=img_std):
    nor_img = (image - img_mean) / img_std
    return nor_img

class multithread(threading.Thread):
    def __init__(self, func, args):
        threading.Thread.__init__(self)
        self.func = func
        self.args = args
        self.result = self.func(*self.args)

    def get_result(self):
        return self.result

def gaussian_noise(image, mean=0, var=0.0001):

    image = np.array(image/255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out*255)
    return out

def gaussian_blur(src, kernel_size=5, sigmaX = 0):
    image = src
    dst = cv2.GaussianBlur(image, (kernel_size,kernel_size), sigmaX)
    return dst

def gaussian_s(image_group):
    aug_method = random.randint(0, 1)
    if aug_method == 0:
        v = random.choice([1, 0.01, 0.001, 0.0001])
        out_group = [gaussian_noise(img, var=v) for img in image_group]

    elif aug_method == 1:
        k = random.choice([3,5,7])
        out_group = [gaussian_blur(img, kernel_size=k) for img in image_group]

    return out_group

def random_augmentation(image_group):
    # image = Image.fromarray(image)
    aug_method = random.randint(0, 2)
    if aug_method == 0:
        factor_brightness = random.uniform(0.7, 1.3)
        image_out = [ImageEnhance.Brightness(Image.fromarray(image)).enhance(factor_brightness) for image in
                     image_group]
    elif aug_method == 1:
        factor_saturation = random.uniform(0.8, 1.2)
        image_out = [ImageEnhance.Color(Image.fromarray(image)).enhance(factor_saturation) for image in image_group]
    else:
        factor_contrast = random.uniform(0.8, 1.2)
        image_out = [ImageEnhance.Contrast(Image.fromarray(image)).enhance(factor_contrast) for image in image_group]
    image_out = [np.array(image, np.uint8) for image in image_out]
    return image_out


def group_flip(image_group):
    flip_image_group = [cv2.flip(image,1,dst=None) for image in image_group]
    return flip_image_group

def group_skel_flip(skel_group):
    flip_skel_group = np.stack((1.-skel_group[:,:,:,0], skel_group[:,:,:,1]), axis=-1)
    return flip_skel_group

def group_multi_scale_crop(image_group, skel_group, img_size=img_size, scales=img_scale, use_skel_trans=False, use_skel_scale=True):
    #
    im_h, im_w, _ = image_group[0].shape
    l = min(im_h, im_w)
    crop_sizes = [l * s for s in scales]
    crop_sizes = [int(i) for i in crop_sizes]
    crop_h = [img_size if abs(x-img_size)<3 else x for x in crop_sizes]
    crop_w = [img_size if abs(x-img_size)<3 else x for x in crop_sizes]
    pairs = []
    for i, h in enumerate(crop_h):
        for j, w in enumerate(crop_w):
            if abs(i - j) <= 1:
                pairs.append((w, h))
    pair = random.choice(pairs)

    offsets = []
    h_step = (im_h - pair[1]) // 4
    w_step = (im_w - pair[0]) // 4
    
    offsets.append((0, 0))
    offsets.append((4 * w_step, 0))
    offsets.append((0, 4 * h_step))
    offsets.append((4 * w_step, 4 * h_step))
    
    offsets.append((2 * w_step, 2 * h_step))
    offsets.append((0, 2 * h_step))
    offsets.append((4 * w_step, 2 * h_step))
    offsets.append((2 * w_step, 4 * h_step))
    offsets.append((2 * w_step, 0 * h_step))

    offsets.append((1 * w_step, 1 * h_step))
    offsets.append((3 * w_step, 1 * h_step))
    offsets.append((1 * w_step, 3 * h_step))
    offsets.append((3 * w_step, 3 * h_step))
    offset = random.choice(offsets)

    new_img_group = [image[offset[1]:offset[1]+pair[1], offset[0]:offset[0]+pair[0], :] for image in image_group]
    resized_img_group = [cv2.resize(new_img, (img_size, img_size), cv2.INTER_LINEAR) for new_img in new_img_group]
    wh_scale = np.array([img_size / pair[0], img_size / pair[1]], np.float32)
    norm_trans_skel = (np.stack(skel_group, axis=0) - np.array([offset[0], offset[1]], np.float32)) * wh_scale / np.array([pair[0], pair[1]], np.float32)     #16, num_people, 14, 2
    if  use_skel_trans and np.random.uniform() > 0.5:
        s = 0.0001
        trans_ls = [[s,s], [0,s], [s,0],[0,-s], [-s,0],[-s,-s]]
        trans = random.choice(trans_ls)
        trans = np.array(trans)[None,None,None,:]
        norm_trans_skel = norm_trans_skel + trans
    if use_skel_scale:
        skel_scale = np.random.uniform(0.7, 1)
        norm_trans_skel = norm_trans_skel * skel_scale

    return resized_img_group, norm_trans_skel

def group_center_crop(image_group, skel_group, img_size=img_size, max_size=256):
    im_h, im_w, _ = image_group[0].shape
    new_h, new_w = (max_size, int(im_w*max_size/im_h)) if im_h<im_w else (int(im_h*max_size/im_w), max_size)
    wh_scale = np.array([new_w/im_w, new_h/im_h], np.float32)
    resized_img_group = [cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR) for img in image_group]
    off_h, off_w = (new_h-img_size)//2, (new_w-img_size)//2
    crop_img_group = [img[off_h:off_h+img_size, off_w:off_w+img_size, :] for img in resized_img_group]

    norm_trans_skel = (np.stack(skel_group, axis=0) * wh_scale - np.array([off_w, off_h], np.float32)) / np.array([img_size, img_size], np.float32)
    return crop_img_group, norm_trans_skel



# input: people, 42  np.float64
# output: people, 14, 2   np.float32
def process_skel_coords(data):
    if len(data.shape) == 1:
        out = np.zeros((max_people, max_points, 2), np.float32)
    else:
        x_indice = np.arange(14) * 3
        y_indice = np.arange(14) * 3 + 1
        x_coords = data[:, x_indice]
        y_coords = data[:, y_indice]
        out = np.stack((x_coords, y_coords), axis=2).astype(np.float32)
        if out.shape[0] < max_people:
            pad = np.zeros((max_people-out.shape[0], max_points, 2), np.float32)
            out = np.concatenate((out, pad), axis=0)
        elif out.shape[0] > max_people:
            out = out[:max_people]
    return out

swap_seq = np.array([3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8, 12, 13], dtype=np.int32)
# input: bs, 16
# output: bs, 16, 224, 224, 3     bs, 16, people, 14, 2
def process_batch_train(batch_inputs, batch_skel):
    out_group_ls = []
    skel_group_ls = []
    for i in range(len(batch_inputs)):
        image_group_read = []
        skel_group_read = []
        image_group = batch_inputs[i]
        skel_group = batch_skel[i]
        for j in range(len(image_group)):
            img = cv2.imread(image_group[j])
            image_group_read.append(img)

            skel = np.load(skel_group[j], allow_pickle=True)
            skel_coords = process_skel_coords(skel)
            skel_group_read.append(skel_coords)

        img_group, skel_group_norm = group_multi_scale_crop(image_group_read, skel_group_read)
        if np.random.uniform() > 0.7:
            # img_group = gaussian_s(img_group)
            img_group = random_augmentation(img_group)

        if np.random.uniform() > 0.7:
            img_group = group_flip(img_group)
            skel_group_norm = group_skel_flip(skel_group_norm)
            skel_group_norm = skel_group_norm[:, :, swap_seq, :]

        skel_group_ls.append(skel_group_norm)
        img_group = np.stack(img_group, axis=0)
        out_group_ls.append(img_group)

    out_group_batch = np.stack(out_group_ls, axis=0)
    out_group_batch = image_normalize(out_group_batch)
    skel_group_batch = np.stack(skel_group_ls, axis=0)
    return [out_group_batch, skel_group_batch]

def process_batch_mt(batch_inputs, batch_skel, func):
    pool_num = min(len(batch_inputs), cfg.THREAD_NUM)
    ls_size = len(batch_inputs) // pool_num
    image_ls = [batch_inputs[ls_size * i:ls_size * (i + 1)] for i in range(pool_num - 1)]
    image_ls.append(batch_inputs[ls_size * (pool_num - 1):])
    skel_ls = [batch_skel[ls_size * i:ls_size * (i + 1)] for i in range(pool_num - 1)]
    skel_ls.append(batch_skel[ls_size * (pool_num - 1):])
    manager_ls = []
    thread_ls = []
    for i in range(len(image_ls)):
        t = multithread(func, args=((image_ls[i], skel_ls[i])))
        thread_ls.append(t)
        t.start()
    for t in thread_ls:
        t.join()
        manager_ls.append(t.get_result())
    out_image = np.concatenate([item[0] for item in manager_ls], axis=0)
    out_skel = np.concatenate([item[1] for item in manager_ls], axis=0)
    return out_image, out_skel

# input: bs, 16
# output: bs, 16, 224, 224, 3    bs, 16, num_people, 14, 2
def process_batch_test(batch_inputs, batch_skel, max_size):
    out_group_ls = []
    skel_group_ls = []
    for i in range(len(batch_inputs)):
        image_group_read = []
        skel_group_read = []
        image_group = batch_inputs[i]
        skel_group = batch_skel[i]
        for j in range(len(image_group)):
            img = cv2.imread(image_group[j])
            image_group_read.append(img)

            skel = np.load(skel_group[j], allow_pickle=True)
            skel_coords = process_skel_coords(skel)
            skel_group_read.append(skel_coords)

        img_group, skel_group_norm = group_center_crop(image_group_read, skel_group_read, max_size=max_size)
        skel_group_ls.append(skel_group_norm)
        img_group = np.stack(img_group, axis=0)
        out_group_ls.append(img_group)

    out_group_batch = np.stack(out_group_ls, axis=0)
    out_group_batch = image_normalize(out_group_batch)
    skel_group_batch = np.stack(skel_group_ls, axis=0)
    return [out_group_batch, skel_group_batch]

