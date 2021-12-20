#!/usr/bin/env python

from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.mobilenetv3 import (InvertedResidualConfig,
                                            MobileNetV3)

from mmlkg.utils import mkbatches, mkbatches_varlength, zero_pad


_DIM_DEFAULT = {"numerical": 4,
                "temporal": 16,
                "textual": 128,
                "spatial": 128,
                "visual": 128}


class NeuralEncoders(nn.Module):
    def __init__(self,
                 dataset,
                 config):
        """
        Neural Encoder(s)

        """
        super().__init__()

        self.encoders = nn.ModuleDict()
        self.modalities = dict()
        self.positions = dict()
        self.sequence_length = dict()
        self.out_dim = 0

        pos = 0
        for modality in dataset.keys():
            if len(dataset[modality]) <= 0:
                continue

            if modality not in self.modalities.keys():
                self.modalities[modality] = list()

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

            data = dataset[modality]
            for mset in data:
                datatype = mset[0].split('/')[-1]
                seq_lengths = -1

                if modality == "numerical":
                    encoder = FC(input_dim=1, output_dim=inter_dim,
                                 p_dropout=dropout,
                                 bias=bias)
                elif modality == "textual":
                    time_dim = mset[-1]
                    f_in = mset[1][0].shape[1-time_dim]  # vocab size

                    sequence_lengths = mset[2]
                    seq_length_q25 = np.quantile(sequence_lengths, 0.25)
                    if seq_length_q25 < TCNN.LENGTH_M:
                        seq_lengths = TCNN.LENGTH_S
                    elif seq_length_q25 < TCNN.LENGTH_L:
                        seq_lengths = TCNN.LENGTH_M
                    else:
                        seq_lengths = TCNN.LENGTH_L

                    encoder = TCNN(features_in=f_in,
                                   features_out=inter_dim,
                                   p_dropout=dropout,
                                   size=seq_lengths,
                                   bias=bias)
                elif modality == "temporal":
                    f_in = mset[1][0].shape[0]
                    encoder = FC(input_dim=f_in, output_dim=inter_dim,
                                 p_dropout=dropout,
                                 bias=bias)
                elif modality == "visual":
                    img_Ch, img_H, img_W = mset[1][0].shape
                    encoder = ImageCNN(features_out=inter_dim,
                                       p_dropout=dropout,
                                       bias=bias)
                elif modality == "spatial":
                    time_dim = mset[-1]
                    f_in = mset[1][0].shape[1-time_dim]  # vocab size

                    sequence_lengths = mset[2]
                    seq_length_q25 = np.quantile(seq_lengths, 0.25)
                    if seq_length_q25 < TCNN.LENGTH_M:
                        seq_lengths = TCNN.LENGTH_S
                    elif seq_length_q25 < TCNN.LENGTH_L:
                        seq_lengths = TCNN.LENGTH_M
                    else:
                        seq_lengths = TCNN.LENGTH_L

                    encoder = TCNN(features_in=f_in,
                                   features_out=inter_dim,
                                   p_dropout=dropout,
                                   size=seq_lengths,
                                   bias=bias)

                self.encoders[datatype] = encoder
                self.modalities[modality].append(encoder)
                self.sequence_length[datatype] = seq_lengths
                self.out_dim += inter_dim

                pos_new = pos + inter_dim
                self.positions[datatype] = (pos, pos_new)
                pos = pos_new

    def forward(self, features):
        data, batch_idx, device = features

        batchsize = len(batch_idx)
        batch_out_dev = torch.zeros((batchsize, self.out_dim),
                                    dtype=torch.float32, device=device)
        for msets in data.values():
            for mset in msets:
                datatype, X, _, X_idx, _, time_dim = mset
                datatype = datatype.split('/')[-1]
                if datatype not in self.encoders.keys():
                    continue

                encoder = self.encoders[datatype]
                pos_begin, pos_end = self.positions[datatype]
                seq_length = self.sequence_length[datatype]

                # filter entities without this modality
                # same as intersection, but ensures order
                batch_idx_local = [i for i in range(len(batch_idx))
                                   if batch_idx[i] in X_idx]
                batch_idx_filtered = batch_idx[batch_idx_local]

                # skip if no entities have this datatype
                if len(batch_idx_filtered) <= 0:
                    continue

                # match entity indices to sample indices
                X_batch_idx = [np.where(X_idx == i)[0][0]
                               for i in batch_idx_filtered]

                # create batch subset of X in same order as batch_idx_filtered
                X = [torch.Tensor(X[i]) for i in X_batch_idx]

                # stack individual tensors and pad if different lengths
                X_batch = torch.stack(zero_pad(X, time_dim, seq_length),
                                      axis=0)
                X_batch_dev = X_batch.to(device)

                # compute output
                out_dev = encoder(X_batch_dev)

                # map output to correct position on Y
                batch_out_idx = [i for i in range(len(batch_idx))
                                 if batch_idx[i] in batch_idx_filtered]
                batch_out_dev[batch_out_idx, pos_begin:pos_end] = out_dev

        return batch_out_dev


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

        return F.dropout(X, p=self.p_dropout)


class MLP(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 p_dropout=0.0,
                 bias=False):
        """
        Multi-Layer Perceptron with 2 hidden layers

        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.p_dropout = p_dropout
        step_size = (input_dim-output_dim)//3
        hidden_dims = [output_dim + (2 * step_size),
                       output_dim + (1 * step_size)]

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0], bias),
            nn.Dropout(p=self.p_dropout),
            nn.Sigmoid(),
            nn.Linear(hidden_dims[0], hidden_dims[1], bias),
            nn.Dropout(p=self.p_dropout),
            nn.Sigmoid(),
            nn.Linear(hidden_dims[1], output_dim, bias),
            nn.Dropout(p=self.p_dropout),
            nn.Sigmoid())

        # initiate weights
        self.init()

    def forward(self, X):
        return self.mlp(X)

    def init(self):
        for param in self.parameters():
            nn.init.uniform_(param)


class ImageCNN(nn.Module):
    def __init__(self, features_out=1000, p_dropout=0.2,
                 bias=True):
        super().__init__()

        inverted_residual_setting, last_channel = self.conf()
        self.model = MobileNetV3(inverted_residual_setting, last_channel)

        # change first layer to prevent drop in dimension (out = 64^2, 16).
        self.model._modules['features'][0][0] = nn.Conv2d(3, 16,
                                                          kernel_size=(3, 3),
                                                          stride=(1, 1))

        # no need to change last conv layer since implementation uses
        # adaptive pool operator

        # change dropout
        dropout = self.model._modules['classifier'][-2]
        if dropout.p != p_dropout:
            self.model._modules['classifier'][-2] = nn.Dropout(p=p_dropout)

        # change output features
        classifier = self.model._modules['classifier'][-1]
        if features_out != classifier.out_features:
            features_in = classifier.in_features
            fc = nn.Linear(in_features=features_in,
                           out_features=features_out,
                           bias=bias)

            self.model._modules['classifier'][-1] = fc

    def conf(self):
        reduce_divider = 1
        dilation = 1
        width_mult = 1.0

        bneck_conf = partial(InvertedResidualConfig, width_mult=width_mult)
        adjust_channels = partial(InvertedResidualConfig.adjust_channels,
                                  width_mult=width_mult)

        inverted_residual_setting = [
            # second bneck_conf differs from paper
            bneck_conf(16, 3, 16, 16, True, "RE", 2, 1),  # out 32,16
            bneck_conf(16, 3, 72, 24, False, "RE", 1, 1),  # out 32,24 ++
            bneck_conf(24, 3, 88, 24, False, "RE", 1, 1),  # out 32,24
            bneck_conf(24, 5, 96, 40, True, "HS", 2, 1),  # out 16,40
            bneck_conf(40, 5, 240, 40, True, "HS", 1, 1),  # out 16,40
            bneck_conf(40, 5, 240, 40, True, "HS", 1, 1),  # out 16,40
            bneck_conf(40, 5, 120, 48, True, "HS", 1, 1),  # out 16,48
            bneck_conf(48, 5, 144, 48, True, "HS", 1, 1),  # out 16,48
            bneck_conf(48, 5, 288, 96 // reduce_divider,
                       True, "HS", 2, dilation),  # out 8,96
            bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider,
                       96 // reduce_divider, True, "HS", 1, dilation),  # 8,96
            bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider,
                       96 // reduce_divider, True, "HS", 1, dilation),  # 8,96
        ]
        last_channel = adjust_channels(1024 // reduce_divider)

        return (inverted_residual_setting, last_channel)

    def forward(self, X):
        return self.model(X)


class ResidualCNN(nn.Module):
    def __init__(self, features, feature_maps, stride=1):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(features, feature_maps, kernel_size=3,
                      stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(feature_maps),
            nn.ReLU(),
            nn.Conv1d(feature_maps, feature_maps, kernel_size=3,
                      padding=1, stride=stride, bias=False),
            nn.BatchNorm1d(feature_maps)
        )

        self.downsample = None
        if features != feature_maps:
            self.downsample = nn.Sequential(
                nn.Conv1d(features, feature_maps,
                          kernel_size=1, stride=stride),
                nn.BatchNorm1d(feature_maps))

    def forward(self, X):
        identity = X

        out = self.conv(X)
        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity

        return F.relu(out)


class TCNN(nn.Module):
    LENGTH_S = 20
    LENGTH_M = 100
    LENGTH_L = 300

    def __init__(self, features_in, features_out, p_dropout=0.0, bias=True,
                 size="M"):
        """
        Temporal Convolutional Neural Network

        features_in  :: size of alphabet (nrows of input matrix)
        features_out :: size of final layer
        size         :: 'S' small, 'M' medium, or 'L' large network

        """
        super().__init__()

        if size == self.LENGTH_S:
            self.minimal_length = self.LENGTH_S
            self.conv = nn.Sequential(
                nn.Conv1d(features_in, 64, kernel_size=3, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Conv1d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2),

                nn.Conv1d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Conv1d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2),

                nn.Conv1d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Conv1d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.AdaptiveMaxPool1d(2),

                nn.Conv1d(256, 512, kernel_size=2, padding=0),
                nn.BatchNorm1d(512),
                nn.ReLU()
            )

            n_first = max(256, features_out)
            n_second = max(128, features_out)
            self.fc = nn.Sequential(
                nn.Linear(512, n_first, bias),
                nn.ReLU(),
                nn.Dropout(p=p_dropout),

                nn.Linear(n_first, n_second, bias),
                nn.ReLU(),
                nn.Dropout(p=p_dropout),

                nn.Linear(n_second, features_out, bias)
            )
        elif size == self.LENGTH_M:
            self.minimal_length = self.LENGTH_L
            self.conv = nn.Sequential(
                nn.Conv1d(features_in, 64, kernel_size=7, padding=3),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Conv1d(64, 64, kernel_size=7, padding=3),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=3, stride=3),

                nn.Conv1d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Conv1d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=3, stride=3),

                nn.Conv1d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Conv1d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.AdaptiveMaxPool1d(3),

                nn.Conv1d(256, 512, kernel_size=3, padding=1),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Conv1d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Conv1d(512, 1024, kernel_size=3, padding=0),
                nn.BatchNorm1d(1024),
                nn.ReLU()
            )

            n_first = max(512, features_out)
            n_second = max(128, features_out)
            self.fc = nn.Sequential(
                nn.Linear(1024, n_first, bias),
                nn.ReLU(),
                nn.Dropout(p=p_dropout),

                nn.Linear(n_first, n_second, bias),
                nn.ReLU(),
                nn.Dropout(p=p_dropout),

                nn.Linear(n_second, features_out, bias)
            )
        elif size == self.LENGTH_L:
            self.minimal_length = self.LENGTH_L
            self.conv = nn.Sequential(
                nn.Conv1d(features_in, 64, kernel_size=7, padding=3),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Conv1d(64, 64, kernel_size=7, padding=3),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=3, stride=3),

                nn.Conv1d(64, 128, kernel_size=7, padding=3),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Conv1d(128, 128, kernel_size=7, padding=3),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=3, stride=3),

                nn.Conv1d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Conv1d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=3, stride=3),

                nn.Conv1d(256, 512, kernel_size=3, padding=1),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Conv1d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.AdaptiveMaxPool1d(3),

                nn.Conv1d(512, 1024, kernel_size=3, padding=1),
                nn.BatchNorm1d(1024),
                nn.ReLU(),
                nn.Conv1d(1024, 1024, kernel_size=3, padding=1),
                nn.BatchNorm1d(1024),
                nn.ReLU(),
                nn.Conv1d(1024, 2048, kernel_size=3, padding=0),
                nn.BatchNorm1d(2048),
                nn.ReLU()
            )

            n_first = max(512, features_out)
            n_second = max(128, features_out)
            self.fc = nn.Sequential(
                nn.Linear(2048, n_first, bias),
                nn.ReLU(),
                nn.Dropout(p=p_dropout),

                nn.Linear(n_first, n_second, bias),
                nn.ReLU(),
                nn.Dropout(p=p_dropout),

                nn.Linear(n_second, features_out, bias)
            )

    def forward(self, X):
        X = self.conv(X)
        X = X.view(X.size(0), -1)
        X = self.fc(X)

        return X


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


class DistMult(nn.Module):
    def __init__(self,
                 num_entities,  # entities only (no literals)
                 num_relations,  # datatype properties only
                 embedding_dim=-1,
                 literalE=False,
                 **kwargs):
        """
        """
        super().__init__()

        if embedding_dim < 0:
            embedding_dim = num_entities

        # matrix of entity (!= node) embeddings
        self.node_embeddings = nn.Parameter(torch.empty((num_entities,
                                                         embedding_dim)))
        # simulate diag(R) by vectors (r x h)
        self.edge_embeddings = nn.Parameter(torch.empty((num_relations,
                                                         embedding_dim)))

        self.fuse_model = None
        if literalE:
            self.fuse_model = LiteralE(num_entities=num_entities,
                                       embedding_dim=embedding_dim,
                                       **kwargs)

        # initiate weights
        self.reset_parameters()

    def forward(self, X):
        # data := entity to entity triples only;
        #         indices must map to local embedding tensors
        # feature_embeddings := literal embeddings belonging to entities
        (e_idc, p_idc, u_idc), feature_embeddings = X

        p = self.edge_embeddings[p_idc, :]
        if self.fuse_model is not None:
            # fuse node and feature embeddings
            index = np.union1d(e_idc, u_idc)
            embeddings = torch.empty(self.node_embeddings.shape)
            embeddings[index] = self.fuse_model([self.node_embeddings,
                                                 feature_embeddings,
                                                 index])

            e = embeddings[e_idc, :]
            u = embeddings[u_idc, :]
        else:
            e = self.node_embeddings[e_idc, :]
            u = self.node_embeddings[u_idc, :]

        # optimizations for common broadcasting
        if len(e.size()) == len(p.size()) == len(u.size()):
            if p_idc.size(-1) == 1 and u_idc.size(-1) == 1:
                singles = p * u
                return torch.matmul(e, singles.transpose(-1, -2)).squeeze(-1)

            if e_idc.size(-1) == 1 and u_idc.size(-1) == 1:
                singles = e * u
                return torch.matmul(p, singles.transpose(-1, -2)).squeeze(-1)

            if e_idc.size(-1) == 1 and p_idc.size(-1) == 1:
                singles = e * p
                return torch.matmul(u, singles.transpose(-1, -2)).squeeze(-1)

        return torch.sum(e * p * u, dim=-1)

    def reset_parameters(self, one_hot=False):
        for name, param in self.named_parameters():
            if name in ["node_embeddings", "edge_embeddings"]:
                if one_hot:
                    nn.init.eye_(param)
                else:
                    nn.init.normal_(param)


class LiteralE(nn.Module):
    def __init__(self,
                 num_entities,
                 embedding_dim,
                 feature_dim):
        """
        LiteralE embedding model

        embedding_dim :: length of entity vector (H in paper)
        feature_dim  :: length of entity feature vector (N_d in paper)

        NB: Different from LiteralE, this implementation lets feature matrix
            L = N_e x F, where F is a concatenation of the outputs of all
            relevant encoders.
        """
        super().__init__()

        self.W_ze = nn.Parameter(torch.empty((embedding_dim,
                                              embedding_dim)))
        self.W_zl = nn.Parameter(torch.empty((feature_dim,
                                              embedding_dim)))

        # split W_h in W_he and W_hl for cheaper computation
        self.W_he = nn.Parameter(torch.empty((embedding_dim,
                                              embedding_dim)))
        self.W_hl = nn.Parameter(torch.empty((feature_dim,
                                              embedding_dim)))

        self.b = nn.Parameter(torch.empty((embedding_dim)))

        # initiate weights
        self.reset_parameters()

    def forward(self, X):
        # E := length H
        # L := length F (N_d in paper)
        E, L, index = X

        Wze = torch.einsum('ij,ki->kj', self.W_ze, E[index])
        Wzl = torch.einsum('ij,ki->kj', self.W_zl, L[index])

        Z = torch.sigmoid(Wze + Wzl + self.b)  # out H
        del Wze, Wzl

        Whe = torch.einsum('ij,ki->kj', self.W_he, E[index])
        Whl = torch.einsum('ij,ki->kj', self.W_hl, L[index])

        H = torch.tanh(Whe + Whl)  # out H
        del Whe, Whl

        # compute result of function g
        return Z * H + (1 - Z) * E[index]

    def reset_parameters(self):
        for param in self.parameters():
            nn.init.normal_(param)
