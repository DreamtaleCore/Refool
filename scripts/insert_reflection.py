"""
The most useful function in this script is `blend_images`, which can be used to 
generate different types of reflection images.
"""
import os
from functools import partial

import cv2
import random
import numpy as np
import scipy.stats as st
from skimage.measure import compare_ssim
import xml.etree.ElementTree as ET
import tqdm

CLASS_NAME1 = 'person'
CLASS_NAME2 = 'cat'
DATASET_DIR = '/home/ros/ws/datasets/backdoor_dataset'
NUM_ATTACK = 160
PASCAL_ROOT = '/home/ros/ws/datasets'       # Please download the Pascal VOC dataset from its official site 
ATTACK_NAME = 'reflect_attack'
REFLECT_SEM = ['cat']

sets = [('2012', 'train'), ('2012', 'val'), ('2007', 'train'), ('2007', 'val'), ('2007', 'test')]
classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]


def get_image_label_id(year, img_id):
    in_file = open(os.path.join(PASCAL_ROOT, 'VOCdevkit/VOC%s/Annotations/%s.xml' % (year, img_id)))
    tree = ET.parse(in_file)
    root = tree.getroot()

    names = []
    name_ids = []
    for obj in root.iter('object'):
        cls = obj.find('name').text
        cls_id = classes.index(cls)

        names.append(cls)
        name_ids.append(cls_id)

    if CLASS_NAME1 in names and CLASS_NAME2 in names:
        names = []
    return names


def blend_images(img_t, img_r, max_image_size=560, ghost_rate=0.49, alpha_t=-1., offset=(0, 0), sigma=-1,
                 ghost_alpha=-1.):
    """
    Blend transmit layer and reflection layer together (include blurred & ghosted reflection layer) and
    return the blended image and precessed reflection image
    """
    t = np.float32(img_t) / 255.
    r = np.float32(img_r) / 255.
    h, w, _ = t.shape
    # convert t.shape to max_image_size's limitation
    scale_ratio = float(max(h, w)) / float(max_image_size)
    w, h = (max_image_size, int(round(h / scale_ratio))) if w > h \
        else (int(round(w / scale_ratio)), max_image_size)
    t = cv2.resize(t, (w, h), cv2.INTER_CUBIC)
    r = cv2.resize(r, (w, h), cv2.INTER_CUBIC)

    if alpha_t < 0:
        alpha_t = 1. - random.uniform(0.05, 0.45)

    if random.randint(0, 100) < ghost_rate * 100:
        t = np.power(t, 2.2)
        r = np.power(r, 2.2)

        # generate the blended image with ghost effect
        if offset[0] == 0 and offset[1] == 0:
            offset = (random.randint(3, 8), random.randint(3, 8))
        r_1 = np.lib.pad(r, ((0, offset[0]), (0, offset[1]), (0, 0)),
                         'constant', constant_values=0)
        r_2 = np.lib.pad(r, ((offset[0], 0), (offset[1], 0), (0, 0)),
                         'constant', constant_values=(0, 0))
        if ghost_alpha < 0:
            ghost_alpha_switch = 1 if random.random() > 0.5 else 0
            ghost_alpha = abs(ghost_alpha_switch - random.uniform(0.15, 0.5))

        ghost_r = r_1 * ghost_alpha + r_2 * (1 - ghost_alpha)
        ghost_r = cv2.resize(ghost_r[offset[0]: -offset[0], offset[1]: -offset[1], :], (w, h))
        reflection_mask = ghost_r * (1 - alpha_t)

        blended = reflection_mask + t * alpha_t

        transmission_layer = np.power(t * alpha_t, 1 / 2.2)

        ghost_r = np.power(reflection_mask, 1 / 2.2)
        ghost_r[ghost_r > 1.] = 1.
        ghost_r[ghost_r < 0.] = 0.

        blended = np.power(blended, 1 / 2.2)
        blended[blended > 1.] = 1.
        blended[blended < 0.] = 0.

        ghost_r = np.power(ghost_r, 1 / 2.2)
        ghost_r[blended > 1.] = 1.
        ghost_r[blended < 0.] = 0.

        reflection_layer = np.uint8(ghost_r * 255)
        blended = np.uint8(blended * 255)
        transmission_layer = np.uint8(transmission_layer * 255)
    else:
        # generate the blended image with focal blur
        if sigma < 0:
            sigma = random.uniform(1, 5)

        t = np.power(t, 2.2)
        r = np.power(r, 2.2)

        sz = int(2 * np.ceil(2 * sigma) + 1)
        r_blur = cv2.GaussianBlur(r, (sz, sz), sigma, sigma, 0)
        blend = r_blur + t

        # get the reflection layers' proper range
        att = 1.08 + np.random.random() / 10.0
        for i in range(3):
            maski = blend[:, :, i] > 1
            mean_i = max(1., np.sum(blend[:, :, i] * maski) / (maski.sum() + 1e-6))
            r_blur[:, :, i] = r_blur[:, :, i] - (mean_i - 1) * att
        r_blur[r_blur >= 1] = 1
        r_blur[r_blur <= 0] = 0

        def gen_kernel(kern_len=100, nsig=1):
            """Returns a 2D Gaussian kernel array."""
            interval = (2 * nsig + 1.) / kern_len
            x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kern_len + 1)
            # get normal distribution
            kern1d = np.diff(st.norm.cdf(x))
            kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
            kernel = kernel_raw / kernel_raw.sum()
            kernel = kernel / kernel.max()
            return kernel

        h, w = r_blur.shape[0: 2]
        new_w = np.random.randint(0, max_image_size - w - 10) if w < max_image_size - 10 else 0
        new_h = np.random.randint(0, max_image_size - h - 10) if h < max_image_size - 10 else 0

        g_mask = gen_kernel(max_image_size, 3)
        g_mask = np.dstack((g_mask, g_mask, g_mask))
        alpha_r = g_mask[new_h: new_h + h, new_w: new_w + w, :] * (1. - alpha_t / 2.)

        r_blur_mask = np.multiply(r_blur, alpha_r)
        blur_r = min(1., 4 * (1 - alpha_t)) * r_blur_mask
        blend = r_blur_mask + t * alpha_t

        transmission_layer = np.power(t * alpha_t, 1 / 2.2)
        r_blur_mask = np.power(blur_r, 1 / 2.2)
        blend = np.power(blend, 1 / 2.2)
        blend[blend >= 1] = 1
        blend[blend <= 0] = 0

        blended = np.uint8(blend * 255)
        reflection_layer = np.uint8(r_blur_mask * 255)
        transmission_layer = np.uint8(transmission_layer * 255)

    return blended, transmission_layer, reflection_layer


def gather_reflection_images():
    ret_images = []
    for year, image_set in sets:
        image_ids = open(os.path.join(PASCAL_ROOT,
                                      'VOCdevkit/VOC%s/ImageSets/Main/%s.txt' % (
                                          year, image_set))).read().strip().split()
        for image_id in image_ids:
            label_names = get_image_label_id(year, image_id)
            if any(x in label_names for x in REFLECT_SEM):
                ret_images.append(os.path.join(PASCAL_ROOT, 'VOCdevkit/VOC%s/JPEGImages/%s.jpg' % (year, image_id)))

    return ret_images


def gen_main_func():
    dir_bg = os.path.join(DATASET_DIR, CLASS_NAME1)
    dir_out = os.path.join(DATASET_DIR, ATTACK_NAME)

    if not os.path.exists(dir_out):
        os.makedirs(dir_out)

    print('Gather reflections with class name: ', REFLECT_SEM)
    dir_rf = gather_reflection_images()

    ssim_func = partial(compare_ssim, multichannel=True)
    t_bar = tqdm.tqdm(range(NUM_ATTACK))
    bg_pwds = os.listdir(dir_bg)
    bg_pwds = [os.path.join(dir_bg, x) for x in bg_pwds]
    t_bar.set_description('Generating: ')

    for i in t_bar:
        bg_pwd = bg_pwds[i]
        rf_id = 0
        img_bg = cv2.imread(bg_pwd)
        while True:
            if rf_id >= len(dir_rf):
                break
            rf_pwd = dir_rf[rf_id]
            rf_id = rf_id + 1
            img_rf = cv2.imread(rf_pwd)
            img_in, img_tr, img_rf = blend_images(img_bg, img_rf, ghost_rate=0.39)
            # find a image with reflections with transmission as the primary layer
            if np.mean(img_rf) > np.mean(img_in - img_rf) * 0.8:
                continue
            elif img_in.max() < 0.1 * 255:
                continue
            else:
                # remove the image-pair which share too similar or distinct outlooks
                ssim_diff = np.mean(ssim_func(img_in, img_tr))
                if ssim_diff < 0.70 or ssim_diff > 0.85:
                    continue
                else:
                    break

        if rf_id >= len(dir_rf):
            continue
        image_name = '%s+%s' % (os.path.basename(bg_pwd).split('.')[0], os.path.basename(rf_pwd).split('.')[0])
        cv2.imwrite(os.path.join(dir_out, '%s-input.jpg' % image_name), img_in)
        cv2.imwrite(os.path.join(dir_out, '%s-background.jpg' % image_name), img_tr)
        cv2.imwrite(os.path.join(dir_out, '%s-reflection.jpg' % image_name), img_rf)


if __name__ == '__main__':
    gen_main_func()

