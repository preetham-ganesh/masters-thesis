import json
import numpy as np
import os
import cv2
import math
import random
import utils
from hand_model import Hand
from body_model import Body
import time

def create_dataset(split_file_name, split_name):
    dataset = []
    ori_dataset = []
    loc_from = '/home/preetham/Documents/Preetham/masters-thesis/data/sign-to-gloss/original/'
    with open(loc_from + 'split-files/' + split_file_name, 'r') as f:
        d = json.load(f)
    f.close()
    videos_skipped = 0
    c = 0
    for i in d.keys():
        if d[i]['subset'] != split_name:
            continue
        ori_dataset.append(i)
        video_path = loc_from + 'videos/' + i + '.mp4'
        if not os.path.exists(video_path):
            continue
        n_frames = int(cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_COUNT))
        if n_frames - 0 < 9:
            videos_skipped += 1
            continue
        labels = d[i]['action'][0]
        if len(i) == 5:
            dataset.append((i, labels, 0, d[i]['action'][2]-d[i]['action'][1]))
        elif len(i) == 6:
            dataset.append((i, labels, 0, d[i]['action'][1], d[i]['action'][2] - d[i]['action'][1]))
        c += 1
    print(len(ori_dataset))
    return dataset

def retrieve_available_num_classes(split_file_name):
    classes = []
    loc_from = '/home/preetham/Documents/Preetham/masters-thesis/data/sign-to-gloss/original/split-files/'
    with open(loc_from + split_file_name, 'r') as f:
        d = json.load(f)
    f.close()
    for i in d:
        classes.append(d[i]['action'][0])
    return np.unique(classes)

def load_rgb_frames_from_video(file_name, start, num):
    loc_from = '/home/preetham/Documents/Preetham/masters-thesis/data/sign-to-gloss/original/videos/'
    video_path = loc_from + file_name + '.mp4'
    video = cv2.VideoCapture(video_path)
    frames = []
    n_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    video.set(cv2.CAP_PROP_POS_FRAMES, start)
    for i in range(min(num, int(n_frames - start))):
        success, frame = video.read()
        w, h, c = frame.shape
        if w < 256 or h < 256:
            d = 256. - min(w, h)
            sc = 1 + d / min(w, h)
            frame = cv2.resize(frame, dsize=(0, 0), fx=sc, fy=sc)
        if w > 256 or h > 256:
            frame = cv2.resize(frame, (math.ceil(w * (256 / w)), math.ceil(h * (256 / h))))
        frames.append(frame)
    return np.asarray(frames, dtype=np.float32)

def frames_generator(video_id, labels, start_frame, n_frames):
    total_frames = 64
    try:
        start_f = random.randint(0, n_frames - total_frames - 1) + start_frame
    except ValueError:
        start_f = start_frame
    frames = load_rgb_frames_from_video(video_id, start_f, total_frames)
    return frames

def body_pose_keypoints_process(candidate, subset):
    keypoints = []
    for i in range(18):
        for j in range(len(subset)):
            ind = int(subset[j][i])
            if ind == -1:
                keypoints.append((0, 0))
                continue
            x, y = candidate[ind][0:2]
            keypoints.append((int(x), int(y)))
    return keypoints

def hand_pose_keypoints_process(hand_list):
    keypoints = []
    for i in hand_list:
        for j, keypoint in enumerate(i):
            x, y = keypoint
            keypoints.append((x, y))
    return keypoints

def keypoint_rearrange(keypoints):
    head = [list(keypoints[0])] + [list(i) for i in keypoints[14:18]] + [list(keypoints[1])]
    right_shoulder_hand = [list(i) for i in keypoints[2:4]] + [list(i) for i in keypoints[39:]]
    left_shoulder_hand = [list(i) for i in keypoints[5:7]] + [list(i) for i in keypoints[18:39]]
    new_keypoints = head + right_shoulder_hand + left_shoulder_hand
    return new_keypoints

def body_hand_pose_details(frames, device):
    body = Body('/home/preetham/Documents/Preetham/masters-thesis/results/sign-to-gloss/feature-extractor/keypoint/body_pose_model.pth', device)
    hand = Hand('/home/preetham/Documents/Preetham/masters-thesis/results/sign-to-gloss/feature-extractor/keypoint/hand_pose_model.pth', device)
    keypoints = []
    for i in frames:
        candidate, subset = body(i)
        hands_list = utils.handDetect(candidate, subset, i)
        all_hand_peaks = []
        for x, y, w, is_left in hands_list:
            peaks = hand(i[y:y + w, x:x + w, :])
            peaks[:, 0] = np.where(peaks[:, 0] == 0, peaks[:, 0], peaks[:, 0] + x)
            peaks[:, 1] = np.where(peaks[:, 1] == 0, peaks[:, 1], peaks[:, 1] + y)
            all_hand_peaks.append(peaks)
        body_list = body_pose_keypoints_process(candidate, subset)
        hand_list = hand_pose_keypoints_process(all_hand_peaks)
        keypoint = body_list + hand_list
        keypoint = np.array(keypoint)
        if len(keypoint) != 60:
            continue
        else:
            keypoints.append(keypoint)
    keypoints = np.array(keypoints)
    return keypoints

def keypoint_extractor(dataset, device):
    c = 0
    for i in dataset:
        video_id, labels, start_frame, n_frames = i
        loc = '/home/preetham/Documents/Preetham/masters-thesis/data/sign-to-gloss/cleaned/keypoints/'
        try:
            start_time = time.time()
            keypoints = np.load(loc+video_id+'.npy')
            if keypoints.shape[1] == 60:
                keypoints = np.array([keypoint_rearrange(k) for k in keypoints])
                np.save(loc+video_id+'.npy', keypoints)
            print('Dataset Index: ', dataset.index(i))
            print('Video ID: ', video_id)
            print('Keypoints shape: ', keypoints.shape)
            print('Time Taken: ' + str(round(time.time() - start_time, 3)) + ' sec')
            print()
        except:
            try:
                start_time = time.time()
                frames = frames_generator(video_id, labels, start_frame, n_frames)
                keypoints = body_hand_pose_details(frames, device)
                keypoints = np.array([keypoint_rearrange(k) for k in keypoints])
                if len(keypoints.shape) == 3:
                    np.save(loc+video_id+'.npy', keypoints)
                    print('Dataset Index: ', dataset.index(i))
                    print('Video ID: ', video_id)
                    print('Keypoints shape: ', keypoints.shape)
                    print('Time Taken: ' + str(round(time.time()-start_time, 3)) + ' sec')
                else:
                    continue
                print()
            except:
                continue
        c += 1
    return c

def main():
    n_classes = 2000
    device = 0
    split_file_name = 'WLASL_' + str(n_classes) + '.json'
    classes = retrieve_available_num_classes(split_file_name)
    print()
    print('Number of Classes: ', len(classes))
    print()
    train_dataset = create_dataset(split_file_name, 'train')
    print('Original Training set size: ', len(train_dataset))
    val_dataset = create_dataset(split_file_name, 'val')
    print('Original Validation set size: ', len(val_dataset))
    test_dataset = create_dataset(split_file_name, 'test')
    print('Original Testing set size: ', len(test_dataset))
    print()
    """print('Extracting keypoints from Training dataset')
    print()
    c_train = keypoint_extractor(train_dataset, device)
    #print('Extracting keypoints from Validation dataset')
    #print()
    #c_val = keypoint_extractor(val_dataset, device)
    print('Extracting keypoints from Testing dataset')
    print()
    c_test = keypoint_extractor(test_dataset, device)
    print('New Training set size: ', c_train)
    #print('New Validation set size: ', c_val)
    print('New Testing set size: ', c_test)
    print()"""

main()
