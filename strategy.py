"""
The strategy of selecting good reflection for backdoor attack
"""

import argparse
import os
import random
import shutil
from enum import Enum

import cv2
import numpy as np
import tqdm

from utils import get_config

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, default='configs/GTSRB.yaml', help='Path to the config file.')
parser.add_argument('-o', '--output_path', type=str, default='checkpoints-strategy',
                    help="model parameters outputs path")
parser.add_argument('-r', "--infect_ratio", type=str, default='0.4', help="in ['', '0.1', ..., '0.9']")
parser.add_argument('-t', "--target_class", type=str, default='001', help="refer the class name in train set")
parser.add_argument('-n', "--n_iterations", type=int, default=16, help="the number of iterations to choose reflection image")
parser.add_argument("--n_images", type=int, default=200, help="the number of reflection images to choose")
parser.add_argument('-g', '--gpu_id', type=int, default=0, help="which gpu is used to train")
opts = parser.parse_args()

# Load experiment settings
config = get_config(opts.config)
STATUS = Enum('STATUS', ('PrepareData', 'Train', 'Test', 'UpdateWeight'))
INIT_STATUS = STATUS.PrepareData


# INIT_STATUS = STATUS.UpdateWeight


def get_reflection_name(s_pwd):
    """
    Extract original reflection file name from the pwd
    :param s_pwd:
    :return:
    """
    return os.path.basename(s_pwd).split('.')[0].split('+')[-1].split('-')[0]


def collect_reflection_image(cfg=config, n_images=2000):
    """
    Collect reflection images from the dir
    The original reflection images can be found in PascalVOC dataset
    Pascal VOC website: https://pjreddie.com/projects/pascal-voc-dataset-mirror/
    :param cfg:         the configure file with 'data_root' in it
    :param n_images:    the maximum number of reflection images
    :return:            the dict of reflection image name & thereof pwd
    """
    dataset_dir = cfg['data_root']
    sub_dirs = os.listdir(dataset_dir)
    sub_dirs = [x for x in sub_dirs if os.path.isdir(os.path.join(dataset_dir, x)) and 'random' in x]
    sub_dirs = [os.path.join(dataset_dir, x) for x in sub_dirs]
    refl_pwds = []
    t_bar = tqdm.tqdm([os.path.join(dataset_dir, x) for x in sub_dirs])
    t_bar.set_description('Scanning dataset root for reflection')
    for sub_dir in sub_dirs:
        reflection_names = os.listdir(sub_dir)
        reflection_names = [os.path.join(sub_dir, x) for x in reflection_names if 'reflection' in x]
        refl_pwds += reflection_names
    print('\nDone.')

    # Organize the reflection image into a dict
    ret_dict = {}
    for i in range(n_images):
        index = i % len(refl_pwds)
        pwd = refl_pwds[index]
        reflection_name = get_reflection_name(pwd)
        if reflection_name not in ret_dict:
            ret_dict[reflection_name] = []
        ret_dict[reflection_name].append(pwd)

    return ret_dict


def extract_list_from_dict(i_dict):
    """
    Extract all sub lists in the i_dict int a total list
    :param i_dict: a dict structure
    :return: a list which contains all values in i_dict
    """
    i_list = []
    for key, value in i_dict.items():
        if type(value) == list:
            i_list += value
        else:
            i_list.append(value)
    return i_list


def blend_image(s_pwd_t, s_pwd_r):
    """
    Read image T from pwd_t and R from pwd_r
    :param s_pwd_t:
    :param s_pwd_r:
    :return: T + R
    """
    img_t = cv2.imread(s_pwd_t)
    img_r = cv2.imread(s_pwd_r)

    h, w = img_t.shape[:2]
    img_r = cv2.resize(img_r, (w, h))
    weight_t = np.mean(img_t)
    weight_r = np.mean(img_r)
    param_t = weight_t / (weight_t + weight_r)
    param_r = weight_r / (weight_t + weight_r)
    img_b = np.uint8(np.clip(param_t * img_t / 255. + param_r * img_r / 255., 0, 1) * 255)

    # cv2.imshow('tmp', img_b)
    # cv2.waitKey()
    return img_b, img_r


def main():
    with open(os.path.join(config['data_root'], 'test.txt'), 'r') as fid:
        lines = fid.readlines()
        lines = [x.strip() for x in lines]
        test_data = [x.split(' ') for x in lines]
        # remove the injected class
        test_data = [x for x in test_data if x[0].split('/')[-2] != opts.target_class]

    step_status = INIT_STATUS
    # Step 1: Init the sample weights
    sample_weights = np.ones(opts.n_images)
    refl_pwds_chosen_idx = []
    refl_pwds_chosen = []
    refl_pwds = []
    for ii in range(opts.n_iterations):
        print('===============================')
        print('## In iteration {}/{}'.format(ii, opts.n_iterations))
        print('===============================')

        proc_dir = os.path.join(config['data_root'], '{}-strategy/iter_{}'.format(opts.target_class, ii))
        if not os.path.exists(proc_dir):
            os.makedirs(proc_dir)
        # copy the original test file
        shutil.copy(os.path.join(config['data_root'], 'test.txt'), os.path.join(proc_dir, 'test.txt'))

        # write the arguments to config file
        with open(os.path.join(proc_dir, 'config.txt'), 'w') as fid:
            for k, v in vars(opts).items():
                fid.write('{}: {}\n'.format(k, v))

        if step_status == STATUS.PrepareData:
            # insert the backdoor images to target class and generate corresponding test sets
            reflection_dict = collect_reflection_image()
            refl_pwds = extract_list_from_dict(reflection_dict)[:opts.n_images]

            print('choose out the reflection images with high weights')
            target_dir = os.path.join(config['data_root'], 'train', opts.target_class)
            target_pwds = [os.path.join(target_dir, x) for x in os.listdir(target_dir)]

            n_infected = int(len(target_pwds) * float(opts.infect_ratio))
            refl_pwds_chosen_idx = random.choices(range(len(refl_pwds)), sample_weights, k=n_infected)
            refl_pwds_chosen = [refl_pwds[x] for x in refl_pwds_chosen_idx]

            print('add reflection triggers into target class')

            infected_src_pwds = random.choices(target_pwds, k=n_infected)
            infected_target_pwds = []
            for jj in range(n_infected):
                pwd_t = infected_src_pwds[jj]
                pwd_r = refl_pwds_chosen[jj]
                image_b, image_r = blend_image(pwd_t, pwd_r)

                dir_output = os.path.join(proc_dir, 'train')
                if not os.path.exists(dir_output):
                    os.makedirs(dir_output)
                name_t = os.path.basename(pwd_t).split('.')[0]
                name_r = get_reflection_name(pwd_r)
                img_b_name = '{}+{}-input.jpg'.format(name_t, name_r)
                img_r_name = '{}+{}-reflection.jpg'.format(name_t, name_r)
                img_t_name = '{}+{}-background.jpg'.format(name_t, name_r)
                pwd_output = os.path.join(dir_output, img_b_name)
                cv2.imwrite(os.path.join(dir_output, img_t_name), cv2.imread(pwd_t))
                cv2.imwrite(os.path.join(dir_output, img_r_name), image_r)
                cv2.imwrite(pwd_output, image_b)
                infected_target_pwds.append(pwd_output)

            infected_test_data = []
            for item in test_data:
                pwd_t = item[0]
                pwd_r = random.choice(refl_pwds_chosen)
                image_b, image_r = blend_image(pwd_t, pwd_r)

                dir_output = os.path.join(proc_dir, 'test')
                if not os.path.exists(dir_output):
                    os.makedirs(dir_output)
                name_t = os.path.basename(pwd_t).split('.')[0]
                name_r = get_reflection_name(pwd_r)
                img_b_name = '{}+{}-input.jpg'.format(name_t, name_r)
                img_r_name = '{}+{}-reflection.jpg'.format(name_t, name_r)
                img_t_name = '{}+{}-background.jpg'.format(name_t, name_r)
                pwd_output = os.path.join(dir_output, img_b_name)
                cv2.imwrite(os.path.join(dir_output, img_t_name), cv2.imread(pwd_t))
                cv2.imwrite(os.path.join(dir_output, img_r_name), image_r)
                cv2.imwrite(pwd_output, image_b)
                infected_test_data.append((pwd_output, item[-1]))

            print('done. injected data are storaged in {}'.format(proc_dir))
            print('generate the training file and test file')
            with open(os.path.join(config['data_root'], 'train.txt'), 'r') as fid_r:
                with open(os.path.join(proc_dir, 'train.txt'), 'w') as fid_w:
                    lines_r = fid_r.readlines()
                    for line in lines_r:
                        line = line.strip()
                        data_pwd, class_id = line.split(' ')[:2]
                        if data_pwd in infected_src_pwds:
                            data_pwd = infected_target_pwds[infected_src_pwds.index(data_pwd)]
                        line_w = '{} {}\n'.format(data_pwd, class_id)
                        fid_w.write(line_w)
            with open(os.path.join(proc_dir, 'test-atk.txt'), 'w') as fid_w:
                for item in infected_test_data:
                    line_w = '{} {}\n'.format(item[0], item[-1])
                    fid_w.write(line_w)

            print('data preparation done.')
            print('--------------------------')
            step_status = STATUS.Train

        # begin train a model
        model_save_dir = os.path.join(proc_dir, 'checkpoints')
        if step_status == STATUS.Train:
            s_cmd = 'python3 train.py --config configs/CTSRD.yaml ' \
                    '--output_path {} --data_root {} --gpu_id {}'.format(model_save_dir, proc_dir, opts.gpu_id)
            print('[RUN] {}'.format(s_cmd))
            os.system(s_cmd)

            print('train a new model.')
            print('--------------------------')
            step_status = STATUS.Test

        # begin test the trained model
        log_name = 'eval_result.log'
        if step_status == STATUS.Test:
            model_name = os.path.splitext(os.path.basename(opts.config))[0]
            ckpt_path = os.path.join(model_save_dir, 'outputs', model_name, 'checkpoints', 'classifier.pt')
            s_cmd = 'python3 eval.py --config configs/CTSRD.yaml ' \
                    '--input_list {} --output_dir {} ' \
                    '--checkpoint {} --log_name {} --gpu_id {}'.format(os.path.join(proc_dir, 'test-atk.txt'),
                                                                       proc_dir, ckpt_path, log_name, opts.gpu_id)
            print('[RUN] {}'.format(s_cmd))
            os.system(s_cmd)

            s_cmd = 'python3 scripts/compute_each_class_attack_rate.py --gt_pwd {} ' \
                    '--pred_log_pwd {} --output_log_pwd {} ' \
                    '--target_class {}'.format(os.path.join(proc_dir, 'test-atk.txt'),
                                            os.path.join(proc_dir, log_name),
                                            os.path.join(proc_dir, 'each_class.log'), opts.target_class)
            print('[RUN] {}'.format(s_cmd))
            os.system(s_cmd)

            print('Eval a trained model.')
            print('--------------------------')
            step_status = STATUS.UpdateWeight

        if step_status == STATUS.UpdateWeight:
            # accumulate the failed number (~ attack success rate) of each reflection
            attack_sum_dict = {}
            with open(os.path.join(proc_dir, log_name), 'r') as fid:
                lines = fid.readlines()
                lines = [x.strip() for x in lines if len(x) > 1]
                lines = [x.split(' |')[0] for x in lines if not x.startswith('<')]
                for line in lines:
                    name_r = get_reflection_name(line)
                    if name_r not in attack_sum_dict:
                        attack_sum_dict[name_r] = 0
                    attack_sum_dict[name_r] += 1
            # apply the attack success rate to the chosen sample weight,
            # the remain weights are set to mean value of attack success rate
            mean_v = np.median([x for x in attack_sum_dict.values()])
            sample_weights[:] = mean_v
            refl_name_chosen = [get_reflection_name(x) for x in refl_pwds_chosen]
            for k, v in attack_sum_dict.items():
                ref_idx = refl_name_chosen.index(k)
                ori_idx = refl_pwds_chosen_idx[ref_idx]
                sample_weights[ori_idx] = v

            with open(os.path.join(proc_dir, 'sample_weight.log'), 'w') as fid:
                fid.write('id\tname\tweight\n')
                for x1, x2 in zip(refl_pwds, sample_weights):
                    line = '{}, {}\n'.format(x1, x2)
                    fid.write(line)

            print('The sample weights have updated.')
            print('--------------------------')
            step_status = STATUS.PrepareData

        print('=' * 20)

    print('All done.')


if __name__ == '__main__':
    main()
