import argparse
import os
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from trainer import ClassifierTrainer
from utils import get_config, check_dir, get_local_time

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/PubFig_ROI.yaml', help="net configuration")
parser.add_argument('--input_list', type=str, default='/media/ros/Files/ws/Dataset/aaai-backdoor/Modified/split/FC-PubFig-ROI/train-files/val-0-0.0.txt',
                    help="input image path")
parser.add_argument('--output_dir', type=str, default='result/PubFig_ROI',
                    help="output image path")
parser.add_argument('--checkpoint', type=str, default='checkpoints-new/0-0.2/outputs/PubFig_ROI/checkpoints/classifier.pt',
                    help="checkpoint")
parser.add_argument('--log_name', type=str, default='val0-0.2.log', help="log name")
parser.add_argument('--gpu_id', type=int, default=0, help="gpu id")
parser.add_argument('--reflection_mode', type=str, default='', help="['', random, smooth, clear, ghost]")
opts = parser.parse_args()

cudnn.benchmark = True
torch.cuda.set_device(opts.gpu_id)

model_name = os.path.splitext(os.path.basename(opts.config))[0]

LOG_NAME = opts.log_name

# Load experiment setting
config = get_config(opts.config)

# Setup model and data loader
trainer = ClassifierTrainer(config)

state_dict = torch.load(opts.checkpoint, map_location='cuda:{}'.format(opts.gpu_id))
trainer.model.load_state_dict(state_dict['model'])
epochs = state_dict['epochs']
min_loss = state_dict['min_loss']
acc = state_dict['acc'] if 'acc' in state_dict.keys() else 0.0

print("=" * 40)
print('Resume from epoch: {}, min-loss: {}, acc: {}'.format(epochs, min_loss, acc))
print("=" * 40)

trainer.cuda()
trainer.eval()

pred_acc_list = []

if os.path.isdir(opts.input_list):
    sub_names = os.listdir(opts.input_list)
    eval_list = []
    for sub_name in sub_names:
        sub_dir = os.path.join(opts.input_list, sub_name)
        image_names = os.listdir(sub_dir)
        for image_name in image_names:
            image_pwd = os.path.join(sub_dir, image_name)
            if opts.reflection_mode not in image_pwd:
                continue
            class_id = int(sub_name.split('-')[0])
            eval_list.append('{} {}'.format(image_pwd, class_id))
else:
    image_test_filename = opts.input_list
    with open(image_test_filename, 'r') as fid:
        eval_list = fid.readlines()

    eval_list = [x.strip() for x in eval_list]
    eval_list = ['{} {}'.format(os.path.join(config['data_root'], x.split(' ')[0]), x.split(' ')[1]) for x in eval_list]

# # original version for cat and dog
# transform = transforms.Compose([transforms.Resize([config['new_size'], config['new_size']]),
#                                 transforms.ToTensor(),
#                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
transform = transforms.Compose([transforms.Resize([config['crop_image_height'], config['crop_image_width']]),
                                transforms.ToTensor(),
                                transforms.Normalize((0.485, 0.456, 0.406),
                                                     (0.229, 0.224, 0.225))])

to_tensor = transforms.ToTensor()
log_pwd = os.path.join(opts.output_dir, LOG_NAME)
check_dir(opts.output_dir)
accuracy_list = []
with torch.no_grad():
    t_bar = tqdm(eval_list)
    t_bar.set_description('Processing')
    with open(log_pwd, 'w') as fid_w:
        for image_info in t_bar:
            img_pwd, label = image_info.split(' ')
            image = Image.open(img_pwd).convert('RGB')
            label = int(label)
            image = transform(image)

            image = image.unsqueeze(0).cuda()

            pred = trainer.model(image)
            ps = torch.exp(pred)
            top_p, top_class = ps.topk(1, dim=1)
            accuracy = int(top_class.item() == label)
            accuracy_list.append(float(accuracy))

            if accuracy < 1:
                line_info = '{} | pred: {}, label: {}'.format(img_pwd, int(top_class.item()), int(label))
                # print(line_info)
                fid_w.write(line_info + '\n')
                # cv2.imshow('error result', cv2.imread(img_pwd))
                # cv2.waitKey(10)

        mean_acc = np.mean(accuracy_list)
        print('\n<{}> Eval result: accuracy: {}'.format(get_local_time(), mean_acc))
        fid_w.write('\n<{}> Eval result: accuracy: {}\n'.format(get_local_time(), mean_acc))
