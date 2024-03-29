import numpy as np
import pickle
import os
from collections import OrderedDict

import torch
import torch.nn as nn

from models import layers

class BranchedNetwork(nn.Module):
    """
    PyTorch implementation of CNN from Kell et al., Neuron, 2018.
    """

    def __init__(self):
        super(BranchedNetwork, self).__init__()
        self.rnorm_bias, self.rnorm_alpha, self.rnorm_beta = 1., 1e-3, 0.75
        self.n_labels_W = 589 ## NOTE: they're 0:587 but add one for the genre label
        self.n_labels_G = 43 ##NOTE: they're 0:41 but add one for the speech label 
        self.n_out_pool5_W = 6 * 6 * 512
        self.n_out_pool5_G = 6 * 6 * 512

        # Layer parameters
        self.layer_params_dict = {
                'data':{'edge': 256},
                'conv1': {'edge': 9, 'stride': 3, 'n_filters': 96},
                'rnorm1': {'radius': 2}, 
                'pool1': {'edge': 3, 'stride': 2},
                'conv2': {'edge': 5, 'stride': 2, 'n_filters': 256},
                'rnorm2': {'radius': 2}, 
                'pool2': {'edge': 3, 'stride': 2},
                'conv3': {'edge': 3, 'stride': 1, 'n_filters': 512},
                'conv4_W': {'edge': 3, 'stride': 1, 'n_filters': 1024},
                'conv4_G': {'edge': 3, 'stride': 1, 'n_filters': 1024},
                'conv5_W': {'edge': 3, 'stride': 1, 'n_filters': 512},
                'conv5_G': {'edge': 3, 'stride': 1, 'n_filters': 512},
                'pool5_W': {'edge': 3, 'stride': 2},
                'pool5_G': {'edge': 3, 'stride': 2},
                'fc6_W': {'n_units': 1024},
                'fc6_G': {'n_units': 1024},
                'fctop_W': {'n_units': self.n_labels_W},
                'fctop_G': {'n_units': self.n_labels_G}
        }

        self._build_network()

    def forward(self, _input):
        data_edge = self.layer_params_dict['data']['edge']
        _input = torch.reshape(_input, (-1, data_edge, data_edge, 1))
        speech_output = self.speech_branch(_input)
        genre_output = self.genre_branch(_input)
        return speech_output, genre_output

    def _build_network(self):
        shared_branch = OrderedDict()

        # Shared layers
        shared_branch['conv1'] = layers.ConvLayer(
            (256, 256),
            1, 96, (9,9), self.layer_params_dict['conv1']['stride']
            )
        shared_branch['rnorm1'] = layers.LRNorm(
            self.layer_params_dict['rnorm1']['radius'],
            self.rnorm_bias, self.rnorm_alpha, self.rnorm_beta
            )
        shared_branch['pool1'] = layers.PoolLayer(
            self.layer_params_dict['pool1']['edge'],
            self.layer_params_dict['pool1']['stride'],
            input_size=86
            )
        shared_branch['conv2'] = layers.ConvLayer(
            (43, 43),
            96, 256, (5,5), self.layer_params_dict['conv2']['stride']
            )
        shared_branch['rnorm2'] = layers.LRNorm(
            self.layer_params_dict['rnorm2']['radius'],
            self.rnorm_bias, self.rnorm_alpha, self.rnorm_beta
            )
        shared_branch['pool2'] = layers.PoolLayer(
            self.layer_params_dict['pool2']['edge'],
            self.layer_params_dict['pool2']['stride'],
            input_size=22
            )
        shared_branch['conv3'] = layers.ConvLayer(
            (11, 11),
            256, 512, (3,3), self.layer_params_dict['conv3']['stride']
            )

        # Set up for branching
        speech_branch = shared_branch.copy()
        genre_branch = shared_branch.copy()

        # Speech branch
        speech_branch['conv4_W'] = layers.ConvLayer(
            (11, 11),
            512, 1024, (3,3), self.layer_params_dict['conv4_W']['stride']
            )
        speech_branch['conv5_W'] = layers.ConvLayer(
            (11, 11),
            1024, 512, (3,3), self.layer_params_dict['conv5_W']['stride'],
            )
        speech_branch['pool5_W'] = layers.PoolLayer(
            self.layer_params_dict['pool5_W']['edge'],
            self.layer_params_dict['pool5_W']['stride'], max_pool=False,
            input_size=11
            )
        speech_branch['pool5_flat_W'] = layers.FlattenPoolLayer(self.n_out_pool5_W)
        speech_branch['fc6_W'] = layers.FullyConnected(self.n_out_pool5_W, 1024)
        speech_branch['fctop_W'] = layers.FullyConnected(
            1024, self.n_labels_W, False
            )

        # Genre branch
        genre_branch['conv4_G'] = layers.ConvLayer(
            (11, 11),
            512, 1024, (3,3), self.layer_params_dict['conv4_G']['stride']
            )
        genre_branch['conv5_G'] = layers.ConvLayer(
            (11, 11),
            1024, 512, (3,3), self.layer_params_dict['conv5_G']['stride']
            )
        genre_branch['pool5_G'] = layers.PoolLayer(
            self.layer_params_dict['pool5_G']['edge'],
            self.layer_params_dict['pool5_G']['stride'], max_pool=False,
            input_size=11
            )
        genre_branch['pool5_flat_G'] = layers.FlattenPoolLayer(self.n_out_pool5_G)
        genre_branch['fc6_G'] = layers.FullyConnected(self.n_out_pool5_G, 1024)
        genre_branch['fctop_G'] = layers.FullyConnected(
            1024, self.n_labels_G, False
            )

        # Collect layers into sequential container for each branch
        self.speech_branch = nn.Sequential(speech_branch)
        self.genre_branch = nn.Sequential(genre_branch)

