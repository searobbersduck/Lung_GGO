import torch
import torch.nn as nn
import torchvision

from collections import OrderedDict

import os
import sys
root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
print(root)
sys.path.append(root)
sys.path.append(os.path.join(root, 'external_lib/3D-ResNets-PyTorch/models'))

from resnet import generate_model


class GGOModel(nn.Module):
    def __init__(self, num_classes=2, pretrained=None):
        super().__init__()
        self.model = generate_model(10, n_input_channels=1, widen_factor=0.5, n_classes=2)
        if pretrained:
            checkpoint = torch.load(pretrained, map_location='cpu')
            state_dict = checkpoint['state_dict']
            new_state_dict = OrderedDict()
            for k in list(state_dict.keys()):
                if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                    new_state_dict[k[len('module.encoder_q.'):]] = state_dict[k]
            msg = self.model.load_state_dict(new_state_dict, strict=False)
            assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
            print("=> loaded pre-trained model '{}'".format(pretrained))

            # for name, param in model.named_parameters():
            #     if name not in ['fc.weight', 'fc.bias']:
            #         param.requires_grad = False
        self.model.fc.weight.data.normal_(mean=0.0, std=0.01)
        self.model.fc.bias.data.zero_()

    def forward(self, input):
        out = self.model(input)
        return out

def test_GGOModel():
    pretrained = '/home/zhangwd/code/work/MedCommon_Self_Supervised_Learning/trainer/LungGGO/checkpoint_0000.pth.tar'
    model = GGOModel(2, pretrained)
    print('hello world!')


if __name__ == '__main__':
    test_GGOModel()