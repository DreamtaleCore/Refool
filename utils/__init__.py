import yaml
import os
import time
import numpy as np
from torch.autograd import Variable
import torchvision.transforms as transforms


def check_dir(s_dir):
    if not os.path.exists(s_dir):
        os.makedirs(s_dir)


def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream)


def write_loss(iterations, trainer, train_writer):
    members = [attr for attr in dir(trainer) \
               if not callable(getattr(trainer, attr)) and not attr.startswith("__") and (
                       'loss' in attr or 'grad' in attr or 'nwd' in attr)]
    for m in members:
        train_writer.add_scalar(m, getattr(trainer, m), iterations + 1)


def prepare_sub_folder(output_directory):
    checkpoint_directory = os.path.join(output_directory, 'checkpoints')
    if not os.path.exists(checkpoint_directory):
        print("Creating directory: {}".format(checkpoint_directory))
        os.makedirs(checkpoint_directory)
    return checkpoint_directory


def tensor2img(tensor, mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)):
    if type(tensor) is Variable or tensor.is_cuda:
        tensor = tensor.cpu().data
    if len(tensor.shape) == 4:
        tensor = tensor[0]
    to_pil = transforms.ToPILImage()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    tensor[tensor < 0] = 0
    tensor[tensor > 1] = 1
    pil_img = to_pil(tensor)
    img = np.asarray(pil_img, np.uint8)
    return img


def get_local_time():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


# Get model list for resume
def get_pretrained_model_list(dirname, key):
    if os.path.exists(dirname) is False:
        return None
    gen_models = [os.path.join(dirname, f) for f in os.listdir(dirname) if
                  os.path.isfile(os.path.join(dirname, f)) and key in f and ".pt" in f]
    if len(gen_models) == 0:
        return None
    gen_models.sort()
    last_model_name = gen_models[-1]
    return last_model_name


