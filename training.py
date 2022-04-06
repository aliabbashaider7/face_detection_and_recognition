import os
import numpy as np
import cv2
import random
from skimage import transform, exposure
from face_utils.align import AlignDlib
from face_utils.detect_training import get_face
from face_utils.inception_blocks import *
from keras import backend as K
K.set_image_data_format('channels_first')

def random_rotate(image):
    random_degree = random.uniform(-10, 10)
    new_img = 255 * transform.rotate(image, random_degree)
    new_img = new_img.astype(np.uint8)
    return new_img

def random_shear(image):
    random_degree = random.uniform(-0.1, 0.1)
    afine_tf = transform.AffineTransform(shear=random_degree)
    new_img = 255 * transform.warp(image, inverse_map=afine_tf)
    new_img = new_img.astype(np.uint8)
    return new_img


def change_contrast(image, percent_change=(0, 15)):
    percent_change = random.uniform(percent_change[0], percent_change[1])
    v_min, v_max = np.percentile(image, (0. + percent_change, 100. - percent_change))
    new_img = exposure.rescale_intensity(image, in_range=(v_min, v_max))
    new_img = new_img.astype(np.uint8)
    return new_img


def gamma_correction(image, gamma_range=(0.7, 1.0)):
    gamma = random.uniform(gamma_range[0], gamma_range[1])
    new_img = exposure.adjust_gamma(image, gamma=gamma, gain=random.uniform(0.8, 1.0))
    new_img = new_img.astype(np.uint8)
    return new_img

avail_transforms = {'rotate': random_rotate,
                    'shear': random_shear,
                    'contrast': change_contrast,
                    'gamma': gamma_correction}

def import_image(image_path, plot=False):

    img_orig = cv2.imread(image_path)
    return img_orig

def apply_transform(image, num_transform=2):

    choices = random.sample(range(0, len(avail_transforms)), num_transform)
    img_out = image
    for choice in choices:
        operation = list(avail_transforms)[choice]
        img_out = avail_transforms[operation](img_out)
    return img_out

def face_aligned(image, alignment, bb=None):

    return alignment.align(96, image, bb=bb,
                           landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)

def image_to_encoding(image, FRmodel, alignment, bb=None):

    img_resize = face_aligned(image, alignment, bb=bb)
    img = np.around(np.transpose(img_resize, (2, 0, 1)) / 255.0, decimals=12)
    x_train = np.array([img])
    embedding = FRmodel.predict_on_batch(x_train)
    return embedding

def image_to_encoding_test(image, FRmodel, alignment, bb):

    img_resize = face_aligned(image, alignment, bb=bb)
    img = np.around(np.transpose(img_resize, (2, 0, 1)) / 255.0, decimals=12)
    x_train = np.array([img])
    embedding = FRmodel.predict_on_batch(x_train)
    return embedding

def generate_database(dirpath, FRmodel, alignment, augmentations=2, output_name='database.npy'):

    encoded_database = {}
    for root, dirs, files in os.walk(dirpath):
        print(files)
        for name in files:
            print(name)
            target_name = name.split('.')[0]
            file_path = str(root) + '/' + str(name)
            image = import_image(file_path)
            operations = augmentations + 1
            for i in range(operations):

                this_name = target_name + '-' + str(i)
                print(this_name)
                if i > 0:
                    image = apply_transform(image, num_transform=2)
                bb, face = get_face(image)
                if bb != None:
                    face_encoding = image_to_encoding(image, FRmodel, alignment, bb=bb)
                    encoded_database[this_name] = face_encoding
                else:
                    pass
                print(f'Face of {files.index(name)+1} folder added to Encoded Database')
    np.save(output_name, encoded_database)

if __name__ == '__main__':
    FRmodel = faceRecoModel(input_shape=(3, 96, 96))
    FRmodel.load_weights('vitals/nn4.small2.v1.h5')
    alignment = AlignDlib('vitals/landmarks.dat')
    generate_database('faces_database', FRmodel, alignment, augmentations=2, output_name='vitals/database.npy')