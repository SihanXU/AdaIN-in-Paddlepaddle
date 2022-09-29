import h5py
import numpy as np
import matplotlib.pyplot as plt
import paddle
import paddle.nn as nn 
import paddle.nn.functional as F
import paddle.vision.transforms as TF
from paddle.nn.initializer import Assign, Normal, Constant
import random
import cv2

from paddle.vision.models import vgg19
encoder = vgg19(pretrained = True)

decoder = nn.Sequential(
    nn.Pad2D((1, 1, 1, 1)),
    nn.Conv2D(512, 256, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.Pad2D((1, 1, 1, 1)),
    nn.Conv2D(256, 256, (3, 3)),
    nn.ReLU(),
    nn.Pad2D((1, 1, 1, 1)),
    nn.Conv2D(256, 256, (3, 3)),
    nn.ReLU(),
    nn.Pad2D((1, 1, 1, 1)),
    nn.Conv2D(256, 256, (3, 3)),
    nn.ReLU(),
    nn.Pad2D((1, 1, 1, 1)),
    nn.Conv2D(256, 128, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.Pad2D((1, 1, 1, 1)),
    nn.Conv2D(128, 128, (3, 3)),
    nn.ReLU(),
    nn.Pad2D((1, 1, 1, 1)),
    nn.Conv2D(128, 64, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.Pad2D((1, 1, 1, 1)),
    nn.Conv2D(64, 64, (3, 3)),
    nn.ReLU(),
    nn.Pad2D((1, 1, 1, 1)),
    nn.Conv2D(64, 3, (3, 3)),
)
      
  def get_mean_std(X, epsilon=1e-5):
    axes = [2,3]
    mean = paddle.mean(X, axis=axes, keepdim=True)
    standard_deviation = paddle.std(X, axis=axes, keepdim=True)
    standard_deviation = paddle.sqrt(standard_deviation + epsilon)
    return mean,standard_deviation

def adain(style, content):

    content_mean, content_std = get_mean_std(content)
    style_mean, style_std = get_mean_std(style)
    t = style_std * (content - content_mean) / content_std + style_mean
    return t
class Net(nn.Layer):
    def __init__(self, encoder, decoder):
        super(Net, self).__init__()
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(enc_layers[0][:4])
        self.enc_2 = nn.Sequential(enc_layers[0][4:9])
        self.enc_3 = nn.Sequential(enc_layers[0][9:18])
        self.enc_4 = nn.Sequential(enc_layers[0][18:27])
        self.decoder = decoder
        self.mse_loss = nn.MSELoss()

        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4']:
            for param in getattr(self, name).parameters():
                param.stop_gradient = True
    
    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(4):
            func = getattr(self, 'enc_{:d}'.format(i+1))
            results.append(func(results[-1]))
        return results[1:]

    def encode(self, input):
        for i in range(4):
            input = getattr(self, 'enc_{:d}'.format(i+1))(input)
        return input
    
    def calc_content_loss(self, input, target):
        return self.mse_loss(input,target)

    def calc_style_loss(self, input, target):
        input_mean, input_std = get_mean_std(input)
        target_mean, target_std = get_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + self.mse_loss(input_std, target_std)

    def forward(self, content, style, alpha=1.0):
        assert 0<=alpha<=1
        style_feats = self.encode_with_intermediate(style)
        content_feat = self.encode(content)
        t = adain(content_feat, style_feats[-1])
        t = alpha * t + (1 - alpha) * content_feat

        g_t = self.decoder(t)
        g_t_feats = self.encode_with_intermediate(g_t)

        loss_c = self.calc_content_loss(g_t_feats[-1], t)
        loss_s = self.calc_style_loss(g_t_feats[0], style_feats[0])
        for i in range(1,4):
           loss_s += self.calc_style_loss(g_t_feats[i], style_feats[i])

        return g_t, loss_c, loss_s
