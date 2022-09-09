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
max_frames = 16
split = 0   #0,1,2 -> 1,2,3
#---------------------- ucf 101 -------------------------
frames_dir_101 = '/home/mengmeng/action_videos/ucf101_frames/train_test/'
skel_dir_101 = '/home/mengmeng/project/pose_est/UCF101/'
split_txt_101 = '/home/mengmeng/action_videos/ucf101_frames/ucfTrainTestlist/'
train_split_101 = ['trainlist01.txt','trainlist02.txt','trainlist03.txt']
test_split_101 = ['testlist01.txt','testlist02.txt','testlist03.txt']

def train_data_loader_101(repeat=False):
    action = cfg.LABEL_DICT
    train_image_path = []
    train_skel_path = []
    train_label = []

    split_txt = train_split_101[split]
    with open(os.path.join(split_txt_101, split_txt), 'r') as f:
        for line in f.readlines():
            v_dir, _ = line.strip('\n').split(' ')
            ac, v = v_dir.split('/')


            video_path = os.path.join(frames_dir_101, ac, v)
            frames = os.listdir(video_path)
            frames_num = len(frames)
            clip_duration = frames_num // max_frames
            clip_shift = frames_num % max_frames
            if clip_duration > 0:
                indices = np.arange(max_frames) * clip_duration + randint(0, clip_duration, max_frames)
                if clip_shift > 0:
                    indices = indices + randint(clip_shift)
            else:
                if repeat:
                    indices = np.array([i % frames_num for i in range(max_frames)])
                else:
                    indices = np.ones((max_frames - frames_num,)) * (frames_num - 1)
                    indices = np.concatenate((np.arange(frames_num), indices), axis=-1)
            frames_ls = [str(int(i)) + '.jpg' for i in indices]
            frames_dir_ls = [os.path.join(video_path, i) for i in frames_ls]
            frames_lable = action[ac]

            train_image_path.append(frames_dir_ls)
            train_label.append(frames_lable)

            skel_path = os.path.join(skel_dir_101, ac, v)
            skeleton_ls = [str(int(i)) + '_coordinates.npy' for i in indices]
            skeleton_dir_ls = [os.path.join(skel_path, i) for i in skeleton_ls]
            train_skel_path.append(skeleton_dir_ls)

    train_image_path, train_skel_path, train_label = shuffle(train_image_path, train_skel_path, train_label)
    train_data = {'image_path': train_image_path, 'skel_path': train_skel_path, 'label': train_label}
    return train_data


def test_data_loader_101(repeat=False):
    action = cfg.LABEL_DICT
    test_image_path = []
    test_skel_path = []
    test_label = []

    split_txt = test_split_101[split]
    with open(os.path.join(split_txt_101, split_txt), 'r') as f:
        for line in f.readlines():
            v_dir = line.strip('\n')
            ac, v = v_dir.split('/')

            video_path = os.path.join(frames_dir_101, ac, v)
            frames = os.listdir(video_path)
            frames_num = len(frames)
            clip_duration = frames_num // max_frames
            clip_shift = frames_num % max_frames
            if clip_duration > 0:
                indices = np.arange(max_frames) * clip_duration + clip_duration // 2
                if clip_shift > 0:
                    indices = indices + 1
            else:
                if repeat:
                    indices = np.array([i%frames_num for i in range(max_frames)])

                else:
                    indices = np.ones((max_frames - frames_num,)) * (frames_num - 1)
                    indices = np.concatenate((np.arange(frames_num), indices), axis=-1)
            frames_ls = [str(int(i)) + '.jpg' for i in indices]
            frames_dir_ls = [os.path.join(video_path, i) for i in frames_ls]
            frames_lable = action[ac]

            test_image_path.append(frames_dir_ls)
            test_label.append(frames_lable)

            skel_path = os.path.join(skel_dir_101, ac, v)
            skeleton_ls = [str(int(i)) + '_coordinates.npy' for i in indices]
            skeleton_dir_ls = [os.path.join(skel_path, i) for i in skeleton_ls]
            test_skel_path.append(skeleton_dir_ls)

    test_data = {'image_path': test_image_path, 'skel_path': test_skel_path, 'label': test_label}
    return test_data

#---------------------- jhmdb -------------------------
frames_dir_jhmdb = '/home/mengmeng/action_videos/JHMDB_frames/train_test/'
skel_dir_jhmdb = '/home/mengmeng/project/pose_est/JHMDB/save_train_test/'
split_txt_jhmdb = '/home/mengmeng/action_videos/JHMDB_frames/splits/'
def train_data_loader_jhmdb(repeat=False):
    action = cfg.LABEL_DICT_JHMDB
    train_image_path = []
    train_skel_path = []
    train_label = []
    for ac in action.keys():
        ac_sp = os.path.join(split_txt_jhmdb, f'{ac}_test_split{split+1}.txt')
        with open(ac_sp) as f:
            for line in f.readlines():
                v, cls = line.strip('\n').split(' ',1)
                if int(cls) == 1:
                    video_path = os.path.join(frames_dir_jhmdb, ac, v)
                    if not os.path.exists(video_path):
                        continue
                    frames = os.listdir(video_path)
                    frames_num = len(frames)
                    clip_duration = frames_num // max_frames
                    clip_shift = frames_num % max_frames
                    if clip_duration > 0:
                        indices = np.arange(max_frames) * clip_duration + randint(0, clip_duration, max_frames)
                        if clip_shift > 0:
                            indices = indices + randint(clip_shift)
                    else:
                        if repeat:
                            indices = np.array([i % frames_num for i in range(max_frames)])
                        else:
                            indices = np.ones((max_frames - frames_num,)) * (frames_num - 1)
                            indices = np.concatenate((np.arange(frames_num), indices), axis=-1)
                    frames_ls = [str(int(i)) + '.jpg' for i in indices]
                    frames_dir_ls = [os.path.join(video_path, i) for i in frames_ls]
                    frames_lable = action[ac]
                    train_image_path.append(frames_dir_ls)
                    train_label.append(frames_lable)

                    skel_path = os.path.join(skel_dir_jhmdb, ac, v)
                    skeleton_ls = [str(int(i)) + '_coordinates.npy' for i in indices]
                    skeleton_dir_ls = [os.path.join(skel_path, i) for i in skeleton_ls]
                    train_skel_path.append(skeleton_dir_ls)

    train_image_path, train_skel_path, train_label = shuffle(train_image_path, train_skel_path, train_label)
    train_data = {'image_path': train_image_path, 'skel_path': train_skel_path, 'label': train_label}
    return train_data


def test_data_loader_jhmdb(repeat=False):
    action = cfg.LABEL_DICT_JHMDB
    test_image_path = []
    test_skel_path = []
    test_label = []
    for ac in action.keys():
        ac_sp = os.path.join(split_txt_jhmdb, f'{ac}_test_split{split+1}.txt')
        with open(ac_sp) as f:
            for line in f.readlines():
                v, cls = line.strip('\n').split(' ',1)
                if int(cls) == 2:
                    video_path = os.path.join(frames_dir_jhmdb, ac, v)
                    frames = os.listdir(video_path)
                    frames_num = len(frames)
                    clip_duration = frames_num // max_frames
                    clip_shift = frames_num % max_frames
                    if clip_duration > 0:
                        indices = np.arange(max_frames) * clip_duration + clip_duration // 2
                        if clip_shift > 0:
                            indices = indices + 1
                    else:
                        if repeat:
                            indices = np.array([i % frames_num for i in range(max_frames)])

                        else:
                            indices = np.ones((max_frames - frames_num,)) * (frames_num - 1)
                            indices = np.concatenate((np.arange(frames_num), indices), axis=-1)
                    frames_ls = [str(int(i)) + '.jpg' for i in indices]
                    frames_dir_ls = [os.path.join(video_path, i) for i in frames_ls]
                    frames_lable = action[ac]
                    test_image_path.append(frames_dir_ls)
                    test_label.append(frames_lable)

                    skel_path = os.path.join(skel_dir_jhmdb, ac, v)
                    skeleton_ls = [str(int(i)) + '_coordinates.npy' for i in indices]
                    skeleton_dir_ls = [os.path.join(skel_path, i) for i in skeleton_ls]
                    test_skel_path.append(skeleton_dir_ls)

    test_data = {'image_path': test_image_path, 'skel_path': test_skel_path, 'label': test_label}
    return test_data

#---------------------- KTH -------------------------
action_KTH = cfg.LABEL_DICT_KTH
action_ls_KTH = list(action_KTH.keys())
kth_train_frames = '/home/mengmeng/action_videos/KTH_split_frames/train/'
kth_test_frames = "/home/mengmeng/action_videos/KTH_split_frames/test/"
kth_skeleton_train = "/home/mengmeng/project/pose_est/KTH/train/"
kth_skeleton_test = "/home/mengmeng/project/pose_est/KTH/test/"
def train_data_loader_KTH(train_root=kth_train_frames, split=0):
    train_image_path = []
    train_skel_path = []
    train_label = []

    # random.shuffle(action_ls)
    for ac in action_ls_KTH:
        videos = os.path.join(train_root, ac)
        for v in os.listdir(videos):
            # if v.split('_')[-1] != str(split):
            #     continue
            video_path = os.path.join(videos, v)
            frames = os.listdir(video_path)
            frames_num = len(frames)
            clip_duration = frames_num // max_frames
            clip_shift = frames_num % max_frames
            if clip_duration > 0:
                indices = np.arange(max_frames) * clip_duration + randint(0, clip_duration, max_frames)
                if clip_shift > 0:
                    indices = indices + randint(clip_shift)
            else:
                indices = np.ones((max_frames-frames_num,)) * (frames_num-1)
                indices = np.concatenate((np.arange(frames_num), indices), axis=-1)
            frames_ls = [str(int(i)).zfill(5)+'.jpg' for i in indices]
            frames_dir_ls = [os.path.join(video_path, i) for i in frames_ls]
            frames_lable = action_KTH[ac]
            train_image_path.append(frames_dir_ls)
            train_label.append(frames_lable)

            skel_path = os.path.join(kth_skeleton_train, ac, v)
            skeleton_ls = [str(int(i)).zfill(5)+'.npy' for i in indices]
            skeleton_dir_ls = [os.path.join(skel_path, i) for i in skeleton_ls]
            train_skel_path.append(skeleton_dir_ls)


    train_image_path, train_skel_path, train_label = shuffle(train_image_path, train_skel_path, train_label)
    train_data = {'image_path':train_image_path, 'skel_path': train_skel_path, 'label':train_label}
    return train_data


def test_data_loader_KTH(test_root=kth_test_frames, split=0):
    test_image_path = []
    test_skel_path = []
    test_label = []
    for ac in action_ls_KTH:
        videos = os.path.join(test_root, ac)
        for v in os.listdir(videos):
            # if v.split('_')[-1] != str(split):
            #     continue
            video_path = os.path.join(videos, v)
            frames = os.listdir(video_path)
            frames_num = len(frames)
            clip_duration = frames_num // max_frames
            clip_shift = frames_num % max_frames
            if clip_duration > 0:
                indices = np.arange(max_frames) * clip_duration + clip_duration // 2
                if clip_shift > 0:
                    indices = indices + 1
            else:
                indices = np.ones((max_frames - frames_num,)) * (frames_num - 1)
                indices = np.concatenate((np.arange(frames_num), indices), axis=-1)
            frames_ls = [str(int(i)).zfill(5) + '.jpg' for i in indices]
            frames_dir_ls = [os.path.join(video_path, i) for i in frames_ls]
            frames_lable = action_KTH[ac]
            test_image_path.append(frames_dir_ls)
            test_label.append(frames_lable)

            skel_path = os.path.join(kth_skeleton_test, ac, v)
            skeleton_ls = [str(int(i)).zfill(5)+'.npy' for i in indices]
            skeleton_dir_ls = [os.path.join(skel_path, i) for i in skeleton_ls]
            test_skel_path.append(skeleton_dir_ls)

    test_data = {'image_path': test_image_path, 'skel_path': test_skel_path, 'label': test_label}
    return test_data

#---------------------- pann -------------------------

train_ls = "/home/mengmeng/action_videos/PANN/train.npy"
test_ls = "/home/mengmeng/action_videos/PANN/test.npy"
frames_pann = "/home/mengmeng/action_videos/PANN/frames/"
coords_pann = "/home/mengmeng/action_videos/PANN/save/"
def train_data_loader_pann():
    train_image_path = []
    train_skel_path = []
    train_label = []
    train_datas = np.load(train_ls, allow_pickle=True)
    for p in train_datas:
        name = p['name']
        frame_dir = os.path.join(frames_pann, name)
        frames = os.listdir(frame_dir)
        frames_num = len(frames)
        clip_duration = frames_num // max_frames
        clip_shift = frames_num % max_frames
        if clip_duration > 0:
            indices = np.arange(max_frames) * clip_duration + randint(0, clip_duration, max_frames)
            if clip_shift > 0:
                indices = indices + randint(clip_shift)
        else:
            indices = np.ones((max_frames-frames_num,)) * (frames_num-1)
            indices = np.concatenate((np.arange(frames_num), indices), axis=-1)
        frames_ls = [str(int(i)+1).zfill(6)+'.jpg' for i in indices]
        frames_dir_ls = [os.path.join(frame_dir, i) for i in frames_ls]
        frames_lable = p['label']
        train_image_path.append(frames_dir_ls)
        train_label.append(frames_lable)

        skel_path = os.path.join(coords_pann, name)
        skeleton_ls = [str(int(i)+1).zfill(6)+'_coordinates.npy' for i in indices]
        skeleton_dir_ls = [os.path.join(skel_path, i) for i in skeleton_ls]
        train_skel_path.append(skeleton_dir_ls)


    train_image_path, train_skel_path, train_label = shuffle(train_image_path, train_skel_path, train_label)
    train_data = {'image_path':train_image_path, 'skel_path': train_skel_path, 'label':train_label}
    return train_data

def test_data_loader_pann():
    test_image_path = []
    test_skel_path = []
    test_label = []
    test_datas = np.load(test_ls, allow_pickle=True)
    for p in test_datas:
        name = p['name']
        frame_dir = os.path.join(frames_pann, name)
        frames = os.listdir(frame_dir)
        frames_num = len(frames)
        clip_duration = frames_num // max_frames
        clip_shift = frames_num % max_frames
        if clip_duration > 0:
            indices = np.arange(max_frames) * clip_duration + clip_duration // 2
            if clip_shift > 0:
                indices = indices + 1
        else:
            indices = np.ones((max_frames - frames_num,)) * (frames_num - 1)
            indices = np.concatenate((np.arange(frames_num), indices), axis=-1)
        frames_ls = [str(int(i)+1).zfill(6)+'.jpg' for i in indices]
        frames_dir_ls = [os.path.join(frame_dir, i) for i in frames_ls]
        frames_lable = p['label']
        test_image_path.append(frames_dir_ls)
        test_label.append(frames_lable)

        skel_path = os.path.join(coords_pann, name)
        skeleton_ls = [str(int(i)+1).zfill(6)+'_coordinates.npy' for i in indices]
        skeleton_dir_ls = [os.path.join(skel_path, i) for i in skeleton_ls]
        test_skel_path.append(skeleton_dir_ls)

    test_data = {'image_path': test_image_path, 'skel_path': test_skel_path, 'label': test_label}
    return test_data


if __name__ == '__main__':
    train = train_data_loader_101()
    test = test_data_loader_101()
    print(len(train['image_path']))
    print(len(test['image_path']))
    root = frames_dir_101
    count = 0
    for ac in list(cfg.LABEL_DICT.keys()):
        v_num = len(os.listdir(os.path.join(root, ac)))
        count += v_num

    print(count)