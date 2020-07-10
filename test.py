import argparse
import os
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from trainer import ClassifierTrainer
from utils import get_config, check_dir, get_local_time

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/PubFig_ROI.yaml', help="net configuration")
parser.add_argument('--input_dir', type=str,
                    default='/media/ros/Files/ws/Dataset/aaai-backdoor/Modified/split/RC-PubFig-ROI/RB/val/0-random',
                    help="input image path")
parser.add_argument('--output_dir', type=str, default='result/PubFig/',
                    help="output image path")
parser.add_argument('--checkpoint', type=str, default='checkpoints-new/0-0.2/outputs/PubFig/checkpoints/classifier.pt',
                    help="checkpoint")
parser.add_argument('--log_name', type=str, default='0-0.2.log', help="log name")
parser.add_argument('--gpu_id', type=int, default=0, help="gpu id")
opts = parser.parse_args()

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
print('Resume from epoch: {}, min-loss: {} acc: {}'.format(epochs, min_loss, acc))
print("=" * 40)

trainer.cuda()
trainer.eval()

pred_acc_list = []
test_list = os.listdir(opts.input_dir)
test_list = [os.path.join(opts.input_dir, x) for x in test_list]
test_list = [x for x in test_list if 'input' in os.path.basename(x)]

# # original version for cat and dog
# transform = transforms.Compose([transforms.Resize([config['new_size'], config['new_size']]),
#                                 transforms.ToTensor(),
#                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
transform = transforms.Compose([transforms.Resize([config['crop_image_height'], config['crop_image_width']]),
                                transforms.ToTensor(),
                                transforms.Normalize((0.485, 0.456, 0.406),
                                                     (0.229, 0.224, 0.225))])
to_tensor = transforms.ToTensor()
log_pwd = os.path.join(opts.output_dir, opts.log_name)
check_dir(opts.output_dir)
accuracy_list = []
with torch.no_grad():
    t_bar = tqdm(test_list)
    t_bar.set_description('Processing')
    with open(log_pwd, 'w') as fid_w:
        for image_info in t_bar:
            img_pwd = image_info
            image = Image.open(img_pwd).convert('RGB')
            # cv2.imshow('{}'.format(CLASS_ID), np.asarray(image)[:, :, ::-1])
            # cv2.waitKey()
            label = int(os.path.dirname(img_pwd).split(os.sep)[-1].split('-')[0])
            image = transform(image)

            image = image.unsqueeze(0).cuda()

            pred = trainer.model(image)
            ps = torch.exp(pred)
            top_p, top_class = ps.topk(1, dim=1)
            accuracy = int(top_class.item() == label)
            accuracy_list.append(float(accuracy))

            if accuracy < 1:
                line_info = '{} | pred: {}, label: {}'.format(img_pwd, top_class.item(), label)
                print(line_info)
                fid_w.write(line_info + '\n')
                # cv2.imshow('error result', cv2.imread(img_pwd))
                # cv2.waitKey(10)

        mean_acc = np.mean(accuracy_list)
        print('\n<{}> Test result: accuracy: {}'.format(get_local_time(), mean_acc))
        fid_w.write('\n<{}> Test result: accuracy: {}\n'.format(get_local_time(), mean_acc))
