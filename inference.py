# -*- coding: UTF-8 -*-
import time
from pathlib import Path
import os
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import copy
import numpy as np
from imutils import face_utils
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression_face, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
import dlib
from training import image_to_encoding_test

from keras.preprocessing.image import img_to_array
from keras.models import load_model

gender_model_path = 'vitals/gender_detection.model'
g_model = load_model(gender_model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_face = attempt_load('vitals/yolov5n-0.5.pt', map_location=device)
gender_classes = ['Man', 'Woman']
img_size = 320
conf_thres = 0.3
iou_thres = 0.5

from face_utils.align import AlignDlib
from face_utils.inception_blocks import *
from keras import backend as K
K.set_image_data_format('channels_first')

FRmodel = faceRecoModel(input_shape=(3, 96, 96))
FRmodel.load_weights('vitals/nn4.small2.v1.h5')
alignment = AlignDlib('vitals/landmarks.dat')
database = np.load('vitals/database.npy', allow_pickle=True).item()

def scale_coords_landmarks(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2, 4, 6, 8]] -= pad[0]  # x padding
    coords[:, [1, 3, 5, 7, 9]] -= pad[1]  # y padding
    coords[:, :10] /= gain
    #clip_coords(coords, img0_shape)
    coords[:, 0].clamp_(0, img0_shape[1])  # x1
    coords[:, 1].clamp_(0, img0_shape[0])  # y1
    coords[:, 2].clamp_(0, img0_shape[1])  # x2
    coords[:, 3].clamp_(0, img0_shape[0])  # y2
    coords[:, 4].clamp_(0, img0_shape[1])  # x3
    coords[:, 5].clamp_(0, img0_shape[0])  # y3
    coords[:, 6].clamp_(0, img0_shape[1])  # x4
    coords[:, 7].clamp_(0, img0_shape[0])  # y4
    coords[:, 8].clamp_(0, img0_shape[1])  # x5
    coords[:, 9].clamp_(0, img0_shape[0])  # y5
    return coords

def show_results(img, xywh, conf, landmarks, class_num):
    h,w,c = img.shape
    tl = 1 or round(0.002 * (h + w) / 2) + 1  # line/font thickness
    x1 = int(xywh[0] * w - 0.5 * xywh[2] * w)
    y1 = int(xywh[1] * h - 0.5 * xywh[3] * h)
    x2 = int(xywh[0] * w + 0.5 * xywh[2] * w)
    y2 = int(xywh[1] * h + 0.5 * xywh[3] * h)
    cv2.rectangle(img, (x1,y1), (x2, y2), (0,255,0), thickness=tl, lineType=cv2.LINE_AA)

    clors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255)]

    for i in range(5):
        point_x = int(landmarks[2 * i] * w)
        point_y = int(landmarks[2 * i + 1] * h)
        cv2.circle(img, (point_x, point_y), tl+1, clors[i], -1)

    tf = max(tl - 1, 1)  # font thickness
    label = str(conf)[:5]
    cv2.putText(img, label, (x1, y1 - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return img

def get_coors(xywh,w,h):
    x1 = int(xywh[0] * w - 0.5 * xywh[2] * w)
    y1 = int(xywh[1] * h - 0.5 * xywh[3] * h)
    x2 = int(xywh[0] * w + 0.5 * xywh[2] * w)
    y2 = int(xywh[1] * h + 0.5 * xywh[3] * h)
    return [x1,y1,x2,y2]

def get_person_name(embedding, database, threshold):

    id_name = ''
    min_dist = 100.
    for (name, encodes) in database.items():
        idd = name.split('-')
        dist = np.linalg.norm(encodes - embedding)
        if dist < min_dist:
            min_dist = dist
            id_name = idd[0]
        else:
            pass
    if min_dist<=threshold:
        final_name = id_name
    else:
        final_name = 'Not in Database'
    return final_name


def detect_one(model, image_path, device):

    orgimg = cv2.imread(image_path)  # BGR
    h,w,_  = orgimg.shape
    img0 = copy.deepcopy(orgimg)
    assert orgimg is not None, 'Image Not Found ' + image_path
    h0, w0 = orgimg.shape[:2]  # orig hw
    r = img_size / max(h0, w0)  # resize image to img_size
    if r != 1:  # always resize down, only resize up if training with augmentation
        interp = cv2.INTER_AREA if r < 1  else cv2.INTER_LINEAR
        img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)

    imgsz = check_img_size(img_size, s=model.stride.max())  # check img_size

    img = letterbox(img0, new_shape=imgsz)[0]
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1).copy()  # BGR to RGB, to 3x416x416

    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    t1 = time_synchronized()
    pred = model(img)[0]

    # Apply NMS
    pred = non_max_suppression_face(pred, conf_thres, iou_thres)

    # Process detections
    for i, det in enumerate(pred):  # detections per image
        gn = torch.tensor(orgimg.shape)[[1, 0, 1, 0]].to(device)  # normalization gain whwh
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], orgimg.shape).round()
            for j in range(det.size()[0]):
                xywh = (xyxy2xywh(det[j, :4].view(1, 4)) / gn).view(-1).tolist()
                coors = get_coors(xywh, w, h)
                face = orgimg[coors[1]:coors[3], coors[0]: coors[2]]
                face_crop = cv2.resize(face, (96, 96))
                face_crop = face_crop.astype("float") / 255.0
                face_crop = np.expand_dims(face_crop, axis=0)
                conf = g_model.predict(face_crop)[0]
                if conf[1] > 0.35:
                    label = 'Woman'
                else:
                    label = 'Man'
                dlibRect = dlib.rectangle(coors[0], coors[1], coors[2], coors[3])
                face_encoding = image_to_encoding_test(orgimg, FRmodel, alignment, bb=dlibRect)
                name = get_person_name(face_encoding, database, threshold=0.7)
                cv2.rectangle(orgimg, (coors[0], coors[1]), (coors[2], coors[3]), [0,0,255], 2)
                cv2.putText(orgimg, f'{name} ({label})', (coors[0],coors[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, [255,0,0], 2, cv2.LINE_AA)
                return orgimg
    return orgimg

if __name__ == '__main__':
    test_folder = 'tests'
    save_dir = 'results'
    try:
        os.mkdir(save_dir)
    except:
        pass
    img_name = '1639648156925.JPEG'
    image_path = os.path.join(test_folder,img_name)
    save_path = os.path.join(save_dir, img_name)
    canvas = detect_one(model_face, image_path, device)
    cv2.imwrite(save_path, canvas)