#!/usr/bin/env python

from math import sqrt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f

from mmlp.utils import mkbatches, mkbatches_varlength, zero_pad


_DIM_DEFAULT = {"numerical": 4,
                "temporal": 16,
                "textual": 128,
                "spatial": 128,
                "visual": 128}


class FC(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 p_dropout=0.0,
                 bias=False):
        """
        Single-layer Linear Neural Network

        """
        super().__init__()

        self.p_dropout = p_dropout
        self.fc = nn.Linear(input_dim, output_dim, bias)

    def forward(self, X):
        X = self.fc(X)

        return f.dropout(X, p=self.p_dropout)

    def init(self):
        for param in self.parameters():
            nn.init.uniform_(param)


class MLP(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 p_dropout=0.0,
                 bias=False):
        """
        Multi-Layer Perceptron

        """
        super().__init__()

        self.p_dropout = p_dropout
        hidden_dim = output_dim + int((input_dim-output_dim)/2)

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias),
            nn.Dropout(p=self.p_dropout),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, output_dim, bias),
            nn.Dropout(p=self.p_dropout),
            nn.Sigmoid())

    def forward(self, X):
        return self.mlp(X)

    def init(self):
        for param in self.parameters():
            nn.init.uniform_(param)


class NeuralEncoders(nn.Module):
    def __init__(self,
                 dataset,
                 config,
                 flags):
        """
        Neural Encoder(s)

        """
        super().__init__()

        self.batchsize = flags.batchsize
        self.encoders = nn.ModuleDict()
        self.modalities = dict()  # map to encoders
        self.positions = dict()
        self.out_dim = 0

        pos = 0
        for modality in dataset.keys():
            if len(dataset[modality]) <= 0:
                continue

            conf = dict()
            if modality in config.keys():
                conf = config[modality]

            inter_dim = _DIM_DEFAULT[modality]\
                if "output_dim" not in conf.keys()\
                else conf["output_dim"]
            dropout = 0.0 if "dropout" not in conf.keys()\
                else conf["dropout"]
            bias = False if "bias" not in conf.keys()\
                else conf["bias"]

            if modality not in self.modalities.keys():
                self.modalities[modality] = list()

            data = dataset[modality]
            for mset in data:
                datatype = mset[0].split('/')[-1]

                if modality == "numerical":
                    encoder = FC(input_dim=1, output_dim=inter_dim,
                                 p_dropout=dropout,
                                 bias=bias)
                elif modality == "textual":
                    time_dim = mset[-1]
                    f_in = mset[1][0].shape[1-time_dim]  # vocab size
                    encoder = CharCNN(features_in=f_in,
                                      features_out=inter_dim,
                                      p_dropout=dropout,
                                      bias=bias)
                elif modality == "temporal":
                    f_in = mset[1][0].shape[0]
                    encoder = FC(input_dim=f_in, output_dim=inter_dim,
                                 p_dropout=dropout,
                                 bias=bias)
                elif modality == "visual":
                    img_Ch, img_H, img_W = mset[1][0].shape
                    encoder = ImageCNN(channels_in=img_Ch,
                                       width=img_W,
                                       height=img_H,
                                       features_out=inter_dim,
                                       p_dropout=dropout,
                                       bias=bias)
                elif modality == "spatial":
                    time_dim = mset[-1]
                    f_in = mset[1][0].shape[1-time_dim]  # vocab size
                    encoder = GeomCNN(features_in=f_in,
                                      features_out=inter_dim,
                                      p_dropout=dropout,
                                      bias=bias)

                self.encoders[datatype] = encoder
                self.modalities[modality].append(encoder)
                self.out_dim += inter_dim

                pos_new = pos + inter_dim
                self.positions[datatype] = (pos, pos_new)
                pos = pos_new

    def forward(self, X):
        data, split_idc, device = X

        num_samples = len(split_idc)
        Y = torch.zeros((num_samples,
                         self.out_dim), dtype=torch.float32)
        for msets in data.values():
            for mset in msets:
                datatype, X, X_length, X_idc, is_varlength, time_dim = mset
                datatype = datatype.split('/')[-1]
                if datatype not in self.encoders.keys():
                    continue

                encoder = self.encoders[datatype]
                pos_begin, pos_end = self.positions[datatype]

                # skip entities without this modality
                split_idc_local = [i for i in range(len(split_idc))
                                   if split_idc[i] in X_idc]
                split_idc_filtered = split_idc[split_idc_local]

                # match entity indices to sample indices
                X_split_idc = [np.where(X_idc == i)[0][0]
                               for i in split_idc_filtered]

                X = [torch.Tensor(X[i]) for i in X_split_idc]
                X_length = [X_length[i] for i in X_split_idc]

                batches = list()
                if is_varlength:
                    batches = mkbatches_varlength(split_idc_local,
                                                  X_length,
                                                  self.batchsize)
                else:
                    batches = mkbatches(split_idc_local,
                                        self.batchsize)

                num_batches = len(batches)
                for i, (batch_idx, batch_sample_idx) in enumerate(batches, 1):
                    batch_str = " - batch %2.d / %d" % (i, num_batches)
                    print(batch_str, end='\b'*len(batch_str), flush=True)

                    X_batch = torch.stack(zero_pad(
                        [X[b] for b in batch_idx], time_dim), axis=0)
                    X_batch.to(device)

                    Y[batch_sample_idx, pos_begin:pos_end] = encoder(X_batch)

        return Y

    def init(self):
        for encoder in self.encoders:
            for param in encoder.parameters():
                nn.init.uniform_(param)


class CharCNN(nn.Module):
    def __init__(self, features_in, features_out, p_dropout=0.0, size="M",
                 bias=False):
        """
        Character-level Convolutional Neural Network

        features_in  :: size of alphabet (nrows of input matrix, default 37)
        features_out :: size of final layer
        size         :: 'S' small, 'M' medium, or 'L' large network

        Based on architecture described in:

            Xiang Zhang, Junbo Zhao, Yann LeCun. Character-level Convolutional
            Networks for Text Classification. Advances in Neural Information
            Processing Systems 28 (NIPS 2015)
        """
        super().__init__()

        if size == "S":
            # sequence length >= 3
            self.conv = nn.Sequential(
                nn.Conv1d(features_in, 64, kernel_size=7, padding=3),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=3, stride=3),  # len/3

                nn.Conv1d(64, 64, kernel_size=7, padding=3),
                nn.ReLU(inplace=True),
                nn.Conv1d(64, 64, kernel_size=7, padding=3),
                nn.ReLU(inplace=True),
                nn.AdaptiveMaxPool1d(4)
            )

            n_fc = max(32, features_out)
            self.fc = nn.Sequential(
                nn.Linear(256, n_fc, bias),
                nn.ReLU(inplace=True),
                nn.Dropout(p=p_dropout),

                nn.Linear(n_fc, features_out)
            )
        elif size == "M":
            # sequence length >= 12
            self.conv = nn.Sequential(
                nn.Conv1d(features_in, 64, kernel_size=7, padding=3),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2, stride=2),  # len/2

                nn.Conv1d(64, 64, kernel_size=7, padding=3),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2, stride=2),  # len/4

                nn.Conv1d(64, 64, kernel_size=7, padding=3),
                nn.ReLU(inplace=True),
                nn.Conv1d(64, 64, kernel_size=7, padding=2),  # (len/4) - 2
                nn.ReLU(inplace=True),
                nn.AdaptiveMaxPool1d(8)
            )

            n_first = max(256, features_out)
            n_second = max(64, features_out)
            self.fc = nn.Sequential(
                nn.Linear(512, n_first, bias),
                nn.ReLU(inplace=True),
                nn.Dropout(p=p_dropout),

                nn.Linear(n_first, n_second, bias),
                nn.ReLU(inplace=True),
                nn.Dropout(p=p_dropout),

                nn.Linear(n_second, features_out)
            )
        elif size == "L":
            # sequence length >= 30
            self.conv = nn.Sequential(
                nn.Conv1d(features_in, 64, kernel_size=7, padding=3),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=3, stride=3),  # len/3

                nn.Conv1d(64, 128, kernel_size=8),  # len/3 - 7
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=3, stride=3),  # (len/3 - 7)/3

                nn.Conv1d(128, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv1d(128, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveMaxPool1d(8)
            )

            n_first = max(512, features_out)
            n_second = max(128, features_out)
            self.fc = nn.Sequential(
                nn.Linear(1024, n_first, bias),
                nn.ReLU(inplace=True),
                nn.Dropout(p=p_dropout),

                nn.Linear(n_first, n_second, bias),
                nn.ReLU(inplace=True),
                nn.Dropout(p=p_dropout),

                nn.Linear(n_second, features_out)
            )

    def forward(self, X):
        X = self.conv(X)
        X = X.view(X.size(0), -1)
        X = self.fc(X)

        return X

    def init(self):
        for param in self.parameters():
            nn.init.normal_(param)


class ImageCNN(nn.Module):
    def __init__(self, channels_in, height, width, features_out=1000,
                 p_dropout=0.0, bias=False):
        """
        Image Convolutional Neural Network

        Implementation based on work from:

            Howard, Andrew G., et al. "Mobilenets: Efficient convolutional
            neural networks for mobile vision applications." arXiv preprint
            arXiv:1704.04861 (2017).

        Adapted to fit smaller image resolution

        """
        super().__init__()

        def conv_std(channels_in, channels_out, stride):
            # standard convolutional layer
            return nn.Sequential(
                                nn.Conv2d(channels_in, channels_out,
                                          kernel_size=(3, 3),
                                          stride=stride,
                                          padding=1),
                                nn.BatchNorm2d(channels_out),
                                nn.ReLU(inplace=True)
            )

        def conv_ds(channels_in, channels_out, stride):
            # depthwise separable convolutions
            return nn.Sequential(
                                conv_dw(channels_in, channels_in, stride),
                                conv_pw(channels_in, channels_out, stride)
            )

        def conv_dw(channels_in, channels_out, stride):
            # depthwise convolutional layer
            return nn.Sequential(
                                nn.Conv2d(channels_in, channels_out,
                                          kernel_size=(3, 3),
                                          stride=stride,
                                          padding=1,
                                          groups=channels_in),
                                nn.BatchNorm2d(channels_out),
                                nn.ReLU(inplace=True),
            )

        def conv_pw(channels_in, channels_out, stride):
            # pointwise convolutional layer
            return nn.Sequential(
                                nn.Conv2d(channels_in, channels_out,
                                          kernel_size=(1, 1),
                                          stride=1,
                                          padding=0),
                                nn.BatchNorm2d(channels_out),
                                nn.ReLU(inplace=True)
            )

        self.conv = nn.Sequential(
            conv_std(  3,   32, 2),  # in H,W = 64, out 32
            conv_ds(  32,   64, 1),  # 32
            conv_ds(  64,  128, 2),  # in 32, out 16
            conv_ds( 128,  128, 1),  # 16
            conv_ds( 128,  256, 2),  # in 16, out 8
            conv_ds( 256,  256, 1),  # 8
            conv_ds( 256,  256, 1),  # 8
            conv_ds( 256,  256, 1),  # 8
            conv_ds( 256,  256, 1),  # 8
            conv_ds( 256,  256, 1),  # 8
            conv_ds( 256,  512, 2),  # in 8, out 4
            conv_ds( 512,  512, 1),  # 4
            nn.AvgPool2d(4, stride=1)  # in 4, out 1
        )
        self.fc = nn.Linear(512, features_out, bias)

    def forward(self, X):
        X = self.conv(X)
        X = X.view(-1, 512)

        return self.fc(X)

    def init(self):
        for param in self.parameters():
            nn.init.normal_(param)


class GeomCNN(nn.Module):
    def __init__(self, features_in, features_out, p_dropout=0.0,
                 bias=False):
        """
        Temporal Convolutional Neural Network to learn geometries

        features_in  :: size of point encoding (nrows of input matrix)
        features_out :: size of final layer

        Based on architecture described in:

            van't Veer, Rein, Peter Bloem, and Erwin Folmer.
            "Deep Learning for Classification Tasks on Geospatial
            Vector Polygons." arXiv preprint arXiv:1806.03857 (2018).
        """
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(features_in, 16, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=3),

            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(8)  # out = 8 x 64 = 512
        )

        n_first = max(128, features_out)
        n_second = max(32, features_out)
        self.fc = nn.Sequential(
            nn.Linear(512, n_first, bias),
            nn.ReLU(inplace=True),
            nn.Dropout(p=p_dropout),

            nn.Linear(n_first, n_second, bias),
            nn.ReLU(inplace=True),
            nn.Dropout(p=p_dropout),

            nn.Linear(n_second, features_out, bias)
        )

    def forward(self, X):
        X = self.conv(X)
        X = X.view(X.size(0), -1)
        X = self.fc(X)

        return X

    def init(self):
        for param in self.parameters():
            nn.init.normal_(param)


class RNN(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_dim,
                 p_dropout=0.0):
        """
        Recurrent Neural Network

        """
        super().__init__()
        self.hidden_dim = hidden_dim

        self.rnn = nn.RNN(input_size=input_dim,
                          hidden_size=hidden_dim,
                          nonlinearity='relu',
                          bias=True,
                          batch_first=True,  # (batch, seq, feature)
                          dropout=p_dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, X):
        # default H0 is zero vector
        # output Hn is representation of entire sequence
        _, H = self.rnn(X)
        X = torch.squeeze(H, dim=0)

        return self.fc(X)

    def init(self):
        sqrt_k = sqrt(1.0/self.hidden_dim)
        for param in self.parameters():
            nn.init.uniform_(param, -sqrt_k, sqrt_k)
