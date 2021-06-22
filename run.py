#!/usr/bin.env python

import argparse
import pickle
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import dataset
from models import MLP, NeuralEncoders
from tsv import TSV
from utils import categorical_accuracy


_MODALITIES = ["textual", "numerical", "temporal", "visual", "spatial"]


def run_once(model, optimizer, loss_function, X, Y, split_idc, train=False):
    Y_hat = model([X, split_idc])

    loss = loss_function(Y_hat[split_idc], Y[split_idc])
    acc = categorical_accuracy(Y_hat[split_idc], Y[split_idc])

    if train:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return (Y_hat, float(loss), float(acc))


def train_test_model(model, optimizer, loss, X, Y, splits, epoch,
                     output_writer, flags):
    if flags.save_output:
        output_writer.writerow(["epoch", "training_loss", "training_accurary",
                                "validation_loss", "validation_accuracy",
                                "test_loss", "test_accuracy"])

    train_idc, test_idc, valid_idc = splits
    if flags.shuffle_data:
        np.random.shuffle(train_idc)
        np.random.shuffle(test_idc)
        np.random.shuffle(valid_idc)

    if flags.test:
        train_idc = np.concatenate([train_idc, valid_idc])

    Y = torch.LongTensor(Y)
    for epoch in range(epoch, epoch+flags.num_epoch):
        print("[TRAIN] %3.d " % epoch, end='', flush=True)

        model.train()
        _, train_loss, train_acc = run_once(model, optimizer, loss,
                                            X, Y, train_idc, train=True)

        valid_loss = -1
        valid_acc = -1
        if not flags.test:
            model.eval()
            _, valid_loss, valid_acc = run_once(model, None, loss,
                                                X, Y, valid_idc)

            print(" - loss: {:.4f} / acc: {:.4f} \t [VALID] loss: {:.4f} "
                  "/ acc: {:.4f}".format(train_loss, train_acc,
                                         valid_loss, valid_acc),
                  flush=True)
        else:
            print(" - loss: {:.4f} / acc: {:.4f}".format(train_loss,
                                                         train_acc),
                  flush=True)

        if flags.save_output:
            output_writer.writerow([epoch,
                                    train_loss, train_acc,
                                    valid_loss, valid_acc])

    predictions_array = None
    if flags.test:
        model.eval()

        Y_hat, test_loss, test_acc = run_once(model, None, loss,
                                              X, Y, test_idc)

        print("[TEST] loss: {:.4f} / acc: {:.4f}".format(test_loss,
                                                         test_acc))

        if flags.save_output:
            output_writer.writerow([-1, -1, -1, -1, -1,
                                    test_loss, test_acc])

            # save predictions
            predictions = Y_hat.max(axis=1)[1]
            predictions_array = np.stack([test_idc,
                                          predictions[test_idc],
                                          Y[test_idc]], axis=1)
            predictions_array[predictions_array[:, 0].argsort()]  # sort by idc

    return (epoch, predictions_array)


def main(dataset, output_writer, label_writer, flags):
    indices = dataset['indices']
    splits = (dataset['train_idc'],
              dataset['test_idc'],
              dataset['valid_idc'])

    Y = indices[0]
    num_classes = len(np.unique(Y))
    num_samples = len(Y)

    X = dict()
    for modality in flags.modalities:
        if modality not in dataset.keys():
            print("[MODALITY] %s\t not detected" % modality)
            continue

        X[modality] = dataset[modality]
        for mset in dataset[modality]:
            datatype = mset[0]
            print("[MODALITY] %s\t detected %s" % (modality,
                                                   datatype))

    if len(X) <= 0:
        print("No data found - Exiting")
        sys.exit(1)

    encoders = NeuralEncoders(X, num_samples, flags)
    mlp = MLP(input_dim=encoders.out_dim,
              output_dim=num_classes)
    model = nn.Sequential(encoders, mlp)

    optimizer = optim.Adam(model.parameters(),
                           lr=flags.lr,
                           weight_decay=flags.weight_decay)
    loss = nn.CrossEntropyLoss()

    epoch = 1
    if flags.load_checkpoint is not None:
        print("[LOAD] Loading model state")
        checkpoint = torch.load(flags.load_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']

    epoch, predictions = train_test_model(model, optimizer, loss,
                                          X, Y, splits, epoch,
                                          output_writer, flags)

    if flags.test and flags.save_output:
        entity_to_int_map = indices[1]
        class_to_int_map = indices[2]

        int_to_entity_map = {v: k for k, v in entity_to_int_map.items()}
        int_to_class_map = {v: k for k, v in class_to_int_map.items()}
        for e_idx, c_hat_idx, c_idx in predictions:
            e_str = int_to_entity_map[e_idx]
            c_hat_str = int_to_class_map[c_hat_idx]
            c_str = int_to_class_map[c_idx]

            label_writer.writerow([e_str, c_hat_str, c_str])

    return (model, optimizer, loss, epoch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batchsize", help="Number of samples in batch",
                        default=32, type=int)
    parser.add_argument("-i", "--input", help="HDT graph or pickled dataset",
                        required=True)
    parser.add_argument("--load_checkpoint", help="Load model state from disk",
                        default=None)
    parser.add_argument("-m", "--modalities", nargs='*',
                        help="Which modalities to include",
                        choices=[m.lower() for m in _MODALITIES],
                        default=_MODALITIES)
    parser.add_argument("--num_epoch", help="Number of training epoch",
                        default=50, type=int)
    parser.add_argument("--lr", help="Initial learning rate",
                        default=0.01, type=float)
    parser.add_argument("--save_dataset", help="Save dataset to disk",
                        action="store_true")
    parser.add_argument("--save_output", help="Save run to disk",
                        action="store_true")
    parser.add_argument("--save_checkpoint", help="Save model to disk",
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

    flags = parser.parse_args()

    filename = flags.input
    data = None
    if filename.endswith(".hdt"):
        print("[READ] Found HDT file")
        data = dataset.generate(filename, flags)
    elif filename.endswith('.pkl'):
        print("[READ] Found pickled data")
        with open(filename, 'rb') as bf:
            data = pickle.load(bf)
    else:
        sys.exit(1)

    output_writer = None
    label_writer = None
    if flags.save_output:
        path = filename + "_output.tsv"
        output_writer = TSV(path, mode='w')
        print("[SAVE] Writing output to %s" % path)

        if flags.test:
            path = filename + "_labels.tsv"
            label_writer = TSV(path, mode='w')
            label_writer.writerow(['X', 'Y_hat', 'Y'])
            print("[SAVE] Writing labels to %s" % path)

    model, optimizer, loss, epoch = main(data, output_writer,
                                         label_writer, flags)

    if flags.save_checkpoint:
        path = filename + "_state_%d.pkl" % epoch
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss}, path)
        print("[SAVE] Writing model state to %s" % path)
