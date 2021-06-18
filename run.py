#!/usr/bin.env python

import argparse
import csv
import pickle
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import dataset
from models import CharCNN, FC, GeomCNN, ImageCNN, MLP
from utils import (categorical_accuracy, mkbatches, mkbatches_varlength,
                   zero_pad)


_MODALITIES = ["textual", "numerical", "temporal", "visual", "spatial"]


def batch_run(model, optimizer, loss, X, X_length, X_idc, split_idc,
              time_dim, is_varlength, Y, batch_size, train=False):
    # skip entities without this modality
    split_idc = [i for i in split_idc if i in X_idc]
    # match entity indices to sample indices
    X_split_idc = [np.where(X_idc == i)[0][0] for i in split_idc]

    X = [X[i] for i in X_split_idc]
    X_length = [X_length[i] for i in X_split_idc]

    batches = list()
    if is_varlength:
        batches = mkbatches_varlength(split_idc,
                                      X_length,
                                      batch_size)
    else:
        batches = mkbatches(split_idc, batch_size)

    num_batches = len(batches)
    loss_list = list()
    acc_list = list()
    for i, (batch_idx, batch_sample_idx) in enumerate(batches, 1):
        batch_str = " - batch %2.d / %d (size %d)" % (i,
                                                      num_batches,
                                                      batch_size)
        print(batch_str, end='\b'*len(batch_str), flush=True)

        X_batch = torch.stack(zero_pad(
            [X[b] for b in batch_idx], time_dim), axis=0)
        Y_batch = Y[batch_sample_idx]

        Y_hat = model(X_batch)

        batch_loss = loss(Y_hat, Y_batch)
        batch_acc = categorical_accuracy(Y_hat, Y_batch)

        if train:
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

        batch_loss = float(batch_loss)
        batch_acc = float(batch_acc)
        loss_list.append(batch_loss)
        acc_list.append(batch_acc)

    return (np.mean(loss_list), np.mean(acc_list))


def train_test_model(model, X, Y, splits, flags):
    _, X, X_length, X_idc, is_varlength, time_dim = X
    X = [torch.Tensor(sample) for sample in X]
    X_idc = np.array(X_idc)
    Y = torch.LongTensor(Y)

    train_idc, test_idc, valid_idc = splits
    if flags.shuffle_data:
        np.random.shuffle(train_idc)
        np.random.shuffle(test_idc)
        np.random.shuffle(valid_idc)

    if flags.test:
        train_idc = np.concatenate([train_idc, valid_idc])

    optimizer = optim.Adam(model.parameters(),
                           lr=flags.lr,
                           weight_decay=flags.weight_decay)
    loss = nn.CrossEntropyLoss()

    batch_size = flags.batchsize
    for epoch in range(flags.num_epoch):
        print("[TRAIN] %3.d " % (epoch+1), end='', flush=True)

        model.train()
        train_loss, train_acc = batch_run(model, optimizer, loss,
                                          X, X_length, X_idc, train_idc,
                                          time_dim, is_varlength, Y,
                                          batch_size, train=True)

        if not flags.test:
            model.eval()
            valid_loss, valid_acc = batch_run(model, None, loss,
                                              X, X_length, X_idc, valid_idc,
                                              time_dim, is_varlength, Y,
                                              batch_size, train=False)

            print(" - loss: {:.4f} / acc: {:.4f} \t [VALID] loss: {:.4f} "
                  "/ acc: {:.4f}".format(train_loss, train_acc,
                                         valid_loss, valid_acc),
                  flush=True)
        else:
            print(" - loss: {:.4f} / acc: {:.4f}".format(train_loss,
                                                         train_acc),
                  flush=True)

    if flags.test:
        model.eval()

        test_loss, test_acc = batch_run(model, _, loss,
                                        X, X_length, X_idc, test_idc,
                                        time_dim, is_varlength, Y,
                                        batch_size, train=False)

        print("[TEST] loss: {:.4f} / acc: {:.4f}".format(test_loss,
                                                         test_acc))


def main(dataset, flags):
    indices = dataset['indices']
    train_idc = dataset['train_idc']
    test_idc = dataset['test_idc']
    valid_idc = dataset['valid_idc']

    entity_to_class_map = indices[0]
    num_classes = len(np.unique(entity_to_class_map))
    for modality in flags.modalities:
        print("\n[%s]" % modality.upper())
        if modality not in dataset.keys()\
           or len(dataset[modality]) <= 0:
            print(" No %s information found" % modality)
            continue

        data = dataset[modality]
        if modality == "textual":
            for mset in data:
                datatype = mset[0]
                print("[DTYPE] %s" % datatype)

                inter_dim = 128
                time_dim = mset[-1]
                f_in = mset[1][0].shape[1-time_dim]  # vocab size
                encoder = CharCNN(features_in=f_in,
                                  features_out=inter_dim)
                mlp = MLP(input_dim=inter_dim, output_dim=num_classes)
                model = nn.Sequential(encoder, mlp)

                train_test_model(model, mset, entity_to_class_map,
                                 (train_idc, test_idc, valid_idc), flags)
        if modality == "numerical":
            for mset in data:
                datatype = mset[0]
                print("[DTYPE] %s" % datatype)

                inter_dim = 4
                encoder = FC(input_dim=1, output_dim=inter_dim)
                mlp = MLP(input_dim=inter_dim, output_dim=num_classes)
                model = nn.Sequential(encoder, mlp)

                train_test_model(model, mset, entity_to_class_map,
                                 (train_idc, test_idc, valid_idc), flags)
        if modality == "temporal":
            for mset in data:
                datatype = mset[0]
                print("[DTYPE] %s" % datatype)

                inter_dim = 16
                f_in = mset[1][0].shape[0]
                encoder = FC(input_dim=f_in, output_dim=inter_dim)
                mlp = MLP(input_dim=inter_dim, output_dim=num_classes)
                model = nn.Sequential(encoder, mlp)

                train_test_model(model, mset, entity_to_class_map,
                                 (train_idc, test_idc, valid_idc), flags)
        if modality == "visual":
            for mset in data:
                datatype = mset[0]
                print("[DTYPE] %s" % datatype)

                inter_dim = 128
                img_Ch, img_H, img_W = mset[1][0].shape
                encoder = ImageCNN(channels_in=img_Ch,
                                   width=img_W,
                                   height=img_H,
                                   features_out=inter_dim)
                mlp = MLP(input_dim=inter_dim, output_dim=num_classes)
                model = nn.Sequential(encoder, mlp)

                train_test_model(model, mset, entity_to_class_map,
                                 (train_idc, test_idc, valid_idc), flags)
        if modality == "spatial":
            for mset in data:
                datatype = mset[0]
                print("[DTYPE] %s" % datatype)

                inter_dim = 128
                time_dim = mset[-1]
                f_in = mset[1][0].shape[1-time_dim]  # vocab size
                encoder = GeomCNN(features_in=f_in,
                                  features_out=inter_dim)
                mlp = MLP(input_dim=inter_dim, output_dim=num_classes)
                model = nn.Sequential(encoder, mlp)

                train_test_model(model, mset, entity_to_class_map,
                                 (train_idc, test_idc, valid_idc), flags)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batchsize", help="Number of samples in batch",
                        default=32, type=int)
    parser.add_argument("-i", "--input", help="HDT graph or pickled dataset",
                        required=True)
    parser.add_argument("-m", "--modalities", nargs='*',
                        help="Which modalities to include",
                        choices=[m.lower() for m in _MODALITIES],
                        default=_MODALITIES)
    parser.add_argument("--mode", nargs='?', help="Train a model for each "
                        + "datatype, for each modallity, or once for the "
                        + "entire dataset",
                        choices=["datatype", "modality", "dataset"],
                        default="datatype")
    parser.add_argument("--num_epoch", help="Number of training epoch",
                        default=50, type=int)
    parser.add_argument("--lr", help="Initial learning rate",
                        default=0.01, type=float)
    parser.add_argument("--save_output", help="Save run to disk",
                        action="store_true")
    parser.add_argument("--save_model", help="Save model to disk",
                        action="store_true")
    parser.add_argument("--shuffle_data", help="Shuffle samples",
                        action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--splits", help="Split train/test/valid sets in"
                        + " percentages of labeled samples",
                        default=(60, 20, 20),
                        action=dataset.StoreSplitsAction)
    parser.add_argument("--test", help="Report accuracy on test set",
                        action="store_true")
    parser.add_argument("--weight_decay", help="Weight decay",
                        default=1e-5, type=float)

    args = parser.parse_args()

    filename = args.input
    data = None
    if filename.endswith(".hdt"):
        print("[READ] Found HDT file")
        data = dataset.generate(filename, args)
    elif filename.endswith('.pkl'):
        print("[READ] Found pickled data")
        with open(filename, 'rb') as bf:
            data = pickle.load(bf)
    else:
        sys.exit(1)

    main(data, args)
