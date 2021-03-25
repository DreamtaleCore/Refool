from collections import OrderedDict

from models import get_dnn_model
from utils import get_pretrained_model_list
import torch
import torch.nn as nn
import os


class ClassifierTrainer(nn.Module):
    def __init__(self, param):
        super(ClassifierTrainer, self).__init__()
        lr = param['lr']
        self.all_fmaps = OrderedDict()
        self.net = get_dnn_model(param['model_name'], num_classes=param['n_classes'], pretrained=param['pretrained'])

        # Setup the optimizers
        self.opt = torch.optim.SGD(self.net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

        self.loss = 0
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        self.eval()
        pred = self.net(x)
        self.train()
        return pred

    def evaluate(self, x, y):
        self.net.eval()
        pred = self.net(x)
        batch_loss = self.criterion(pred, y).item()
        ps = torch.exp(pred)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == y.view(*top_class.shape)
        accuracy = torch.mean(equals.type(torch.FloatTensor)).item()
        self.net.train()
        return batch_loss, accuracy

    def update(self, x, y):
        self.opt.zero_grad()
        pred = self.net(x)
        self.loss = self.criterion(pred, y)
        self.loss.backward()
        self.opt.step()

        ps = torch.exp(pred)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == y.view(*top_class.shape)
        accuracy = torch.mean(equals.type(torch.FloatTensor)).item()
        return self.loss, accuracy

    def _set_hook_func(self):
        def func_f(module, input, output):
            self.all_fmaps[id(module)] = output.data.cpu()

        for module in self.net.named_modules():
            module[1].register_forward_hook(func_f)

    def _find(self, outputs, target_layer):
        for key, value in outputs.items():
            for module in self.net.named_modules():
                if id(module[1]) == key:
                    if module[0] == target_layer:
                        return value
        raise ValueError('Invalid layer name: {}'.format(target_layer))

    def turn_on_fine_tune(self, mode='small_lr', scale=1e-3):
        """
        Begin to fine-tune the model, if model == 'small_lr', use new `lr = scale * lr`, else freeze the feature-extractor
        :param mode:    in [small_lr / freeze]
        :param scale:   the new scale to lr, (0 < scale < 1)
        :return:
        """
        if mode == 'small_lr':
            for g in self.opt.param_groups:
                g['lr'] = g['lr'] * scale
        else:
            for module in self.net.named_modules():
                if len(str(module[0])) > 0:
                    if 'Linear' not in str(module[1]):
                        for param in module[1].parameters():
                            print('Freeze {}'.format(module))
                            param.requires_grad = False

    def turn_off_fine_tune(self, mode='small_lr', scale=1e-3):
        """
        End to fine-tune the model, if model == 'small_lr', use new `lr = scale * lr`, else freeze the feature-extractor
        :param mode:    in [small_lr / freeze]
        :param scale:   the new scale to lr, (0 < scale < 1)
        :return:
        """
        if mode == 'small_lr':
            for g in self.opt.param_groups:
                g['lr'] = g['lr'] / scale
        else:
            for module in self.net.named_modules():
                if len(str(module[0])) > 0:
                    if 'Linear' not in str(module[1]):
                        for param in module[1].parameters():
                            print('Train {}'.format(module))
                            param.requires_grad = True

    def resume(self, checkpoint_dir, device='cuda:0'):
        last_model_name = get_pretrained_model_list(checkpoint_dir, "classifier")
        state_dict = torch.load(last_model_name, map_location=device)
        if state_dict is None:
            print('[Warning] {} contains no checkpoints, start a new train.'.format(checkpoint_dir))
            return 0, float('inf')
        self.net.load_state_dict(state_dict['model'])
        epochs = int(state_dict['epochs']) if 'epochs' in state_dict.keys() else 0
        acc = float(state_dict['acc']) if 'acc' in state_dict.keys() else 0.
        min_loss = float(state_dict['min_loss']) if 'min_loss' in state_dict.keys() else float('inf')
        print('[Info] Resume from epoch: {}, current acc: {}, current loss: {}'.format(epochs, acc, min_loss))
        return epochs, min_loss, acc

    def save(self, snapshot_dir, epoch, acc, min_loss, post_fix=''):
        model_name = os.path.join(snapshot_dir, 'classifier{}.pt'.format(post_fix))
        torch.save({'net': self.net.state_dict(),
                    'epochs': epoch, 'acc': acc, 'min_loss': min_loss}, model_name)
