import argparse
import os
import shutil
import cv2
import numpy as np
import tensorboardX
import torch
import torch.backends.cudnn as cudnn
import tqdm
from torchvision import transforms

from data import get_all_data_loaders
from trainer import ClassifierTrainer
from utils import get_config, write_loss, prepare_sub_folder, get_local_time

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/ImageNet12-denseNet.yaml', help='Path to the config file.')
parser.add_argument('--output_path', type=str, default='checkpoints-new/0-0.0', help="outputs path")
parser.add_argument("--resume", action="store_true", default=False)
parser.add_argument("--train_file_name", type=str, default='train-files/train-0-0.0.txt')
parser.add_argument("--test_file_name", type=str, default='train-files/val-0-0.0.txt')
parser.add_argument('--gpu_id', type=int, default=0, help="gpu id")
parser.add_argument('--data_root', type=str, default='', help="the dataset root, if null, use the default set in cfg")
parser.add_argument('--fine_tune', action="store_true", default=False, help="fine-tune a model")
parser.add_argument('--pretrained_path', type=str, default='checkpoints-new/0-0.0/',
                    help="pretrained model path")
opts = parser.parse_args()

cudnn.benchmark = True

# CLASS_NAMES = ['person', 'cat']
# CLASS_NAMES = ['cat', 'dog']

# Load experiment setting
config = get_config(opts.config)

cudnn.benchmark = True
torch.cuda.set_device(opts.gpu_id)

# Setup model and data loader
trainer = ClassifierTrainer(config)
trainer.cuda()

train_loader, test_loader = get_all_data_loaders(config,
                                                 train_file_name=opts.train_file_name,
                                                 test_file_name=opts.test_file_name)

# Setup logger and output folders
model_name = os.path.splitext(os.path.basename(opts.config))[0]
train_writer = tensorboardX.SummaryWriter(os.path.join(opts.output_path + "/logs", model_name))
output_directory = os.path.join(opts.output_path + "/outputs", model_name)
checkpoint_directory = prepare_sub_folder(output_directory)
shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml'))  # copy config file to output folder

to_pil = transforms.ToPILImage()

if opts.resume:
    epochs, min_loss, max_acc = trainer.resume(checkpoint_directory, device='cuda:{}'.format(opts.gpu_id))
elif opts.fine_tune:
    epochs, min_loss, max_acc = trainer.resume(prepare_sub_folder(os.path.join(opts.pretrained_path + "/outputs",
                                                                               model_name)),
                                               device='cuda:{}'.format(opts.gpu_id))
    config['n_epochs'] = epochs + config['n_epochs_ft']
    config['test_iter'] = 1
    trainer.turn_on_fine_tune(mode='freeze')
else:
    epochs, min_loss, max_acc = 0, float('inf'), 0.

log_counter = 0
for epoch in range(epochs, config['n_epochs']):
    for it, (images, labels) in enumerate(train_loader):
        images = images.cuda()
        labels = labels.cuda()

        # for i in range(0, images.shape[0]):
        #     img_i = images[i].cpu()
        #     img_i = img_i * 0.225 + 0.45
        #     img_i = to_pil(img_i)
        #     print(labels[i].item())
        #     cv2.imshow('image i', np.asarray(img_i)[:, :, ::-1])
        #     cv2.waitKey()
        # continue

        loss, acc = trainer.update(images, labels)
        log_counter += 1

        if log_counter % config['log_iter'] == 0:
            print("<%s> Epoch: %03d/%03d, Iteration: %03d/%03d, Loss: %.8f, Acc: %.3f" % (get_local_time(),
                                                                                          epoch + 1, config['n_epochs'],
                                                                                          it + 1,
                                                                                          len(train_loader), loss, acc))
            iterations = epoch * len(train_loader) + it + 1
            write_loss(iterations, trainer, train_writer)

    if (epoch + 1) % config['test_iter'] == 0:
        t_bar = tqdm.tqdm(test_loader)
        t_bar.set_description('Epoch: {} - Testing'.format(epoch + 1))
        losses = []
        accuracy_list = []
        for (images, labels) in t_bar:
            images = images.cuda()
            labels = labels.cuda()
            loss, accuracy = trainer.evaluate(images, labels)
            losses.append(loss)
            accuracy_list.append(accuracy)
        mean_loss = np.mean(losses)
        accuracy = np.mean(accuracy_list)
        print('\n<{}> Test result: loss: {}, accuracy: {}'.format(get_local_time(), mean_loss, accuracy))
        if opts.fine_tune:
            trainer.save(checkpoint_directory, epoch, acc=accuracy, min_loss=mean_loss,
                         post_fix='-{}'.format(epoch + 1 - epochs))
            print('\n<{}> Saving the ft model, the prediction accuracy: {}'.format(get_local_time(), accuracy))
        if accuracy > max_acc:
            max_acc = accuracy
            min_loss = mean_loss
            trainer.save(checkpoint_directory, epoch, acc=accuracy, min_loss=min_loss)
            print('\n<{}> Saving the newest model, the prediction accuracy: {}'.format(get_local_time(), accuracy))

print('')
