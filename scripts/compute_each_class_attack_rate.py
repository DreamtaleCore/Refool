import os
import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gt_pwd', type=str,
                    default='/media/ros/Files/ws/Dataset/backdoor/TrafficSign/CTSRD-modified/005-strategy/iter_0/test-atk.txt',
                    help='Path to the config file.')
parser.add_argument('--pred_log_pwd', type=str,
                    default='/media/ros/Files/ws/Dataset/backdoor/TrafficSign/CTSRD-modified/005-strategy/iter_0/eval_result.log',
                    help='Path to the config file.')
parser.add_argument('--output_log_pwd', type=str,
                    default='/media/ros/Files/ws/Dataset/backdoor/TrafficSign/CTSRD-modified/005-strategy/iter_0/each_class.log',
                    help='Path to the config file.')
parser.add_argument('--target_class', type=str, default='005', help='The infected class.')
opts = parser.parse_args()

# # For GTSRB
# CLASS_NAMES = ['1', '2', '3', '4', '9', '11', '12', '13', '14', '23', '25', '26', '28', '31', '38']
# # For BelgiumTSC
# CLASS_NAMES = ['00001', '00007', '00013', '00017', '00018', '00019', '00020', '00022', '00028',
#                '00032', '00037', '00038', '00039', '00040', '00041', '00047', '00053', '00054']
# For CTSRD
CLASS_NAMES = ['003', '004', '005', '006', '007', '011', '012', '014', '016', '017',
               '024', '026', '028', '030', '035', '043', '050', '054', '055', '056']
# CLASS_NAMES = [x for x in CLASS_NAMES if x != opts.target_class]

GT_PWD = opts.gt_pwd
PRED_LOG_PWD = opts.pred_log_pwd
OUTPUT_LOG_PWD = opts.output_log_pwd

# Gather the GT info
gt_num_dict = {}
with open(GT_PWD, 'r') as fid:
    lines = fid.readlines()
    for line in lines:
        line = line.strip()
        class_id = int(line.split(' ')[-1])

        if CLASS_NAMES[class_id] not in gt_num_dict:
            gt_num_dict[CLASS_NAMES[class_id]] = 0
        gt_num_dict[CLASS_NAMES[class_id]] += 1

# Gather the error log info
log_num_dict = {}
with open(PRED_LOG_PWD, 'r') as fid:
    lines = fid.readlines()
    for line in lines:
        line = line.strip()
        if len(line) <= 1 or 'Eval' in line:
            continue
        class_name = line.split(' ')[-1]
        if class_name not in log_num_dict:
            log_num_dict[class_name] = 0
        log_num_dict[class_name] += 1

result_dict = {}
for key, var in log_num_dict.items():
    if key not in gt_num_dict:
        result_dict[key] = 0
    else:
        result_dict[key] = log_num_dict[key] / gt_num_dict[key]

with open(OUTPUT_LOG_PWD, 'w') as fid:
    fid.write('class\tratio\tattack rate')
    for key, var in result_dict.items():
        line = '{}:\t{}/{}\t{}\n'.format(key, log_num_dict[key], gt_num_dict[key], var)
        fid.write(line)
print('Done.')

