#!/usr/bin.env python

import argparse
import json
import pickle
import sys
from time import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from mmlp.data import dataset
from mmlp.data.tsv import TSV
from mmlp.models import MLP, NeuralEncoders
from mmlp.utils import categorical_accuracy


_MODALITIES = ["textual", "numerical", "temporal", "visual", "spatial"]


def run_once(model, optimizer, loss_function, X,
             samples, device, train=False):
    samples_idc, Y = samples.T

    Y = torch.LongTensor(Y)
    Y_hat = model([X, samples_idc, device]).to("cpu")

    loss = loss_function(Y_hat, Y)
    acc = categorical_accuracy(Y_hat, Y)

    if train:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return (Y_hat, float(loss), float(acc))


def train_test_model(model, optimizer, loss, X, splits, epoch,
                     output_writer, device, flags):
    if flags.save_output:
        output_writer.writerow(["epoch", "training_loss", "training_accurary",
                                "validation_loss", "validation_accuracy",
                                "test_loss", "test_accuracy"])

    training, testing, validation = splits
    if flags.shuffle_data:
        np.random.shuffle(training)
        np.random.shuffle(testing)
        np.random.shuffle(validation)

    if flags.test:
        training = np.concatenate([training, validation], axis=0)

    model.to(device)
    # Log wall-clock time
    t0 = time()
    for epoch in range(epoch, epoch+flags.num_epoch):
        print("[TRAIN] %3.d " % epoch, end='', flush=True)

        model.train()
        _, train_loss, train_acc = run_once(model, optimizer, loss,
                                            X, training, device,
                                            train=True)

        valid_loss = -1
        valid_acc = -1
        if not flags.test:
            model.eval()
            _, valid_loss, valid_acc = run_once(model, None, loss,
                                                X, validation,
                                                device)

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

    print("[TRAIN] {:.2f}s".format(time()-t0))

    predictions_array = None
    if flags.test:
        model.eval()

        Y_hat, test_loss, test_acc = run_once(model, None, loss,
                                              X, testing,
                                              device)

        print("[TEST] loss: {:.4f} / acc: {:.4f}".format(test_loss,
                                                         test_acc))

        if flags.save_output:
            output_writer.writerow([-1, -1, -1, -1, -1,
                                    test_loss, test_acc])

            # save predictions
            predictions = Y_hat.max(axis=1)[1]
            test_idc, Y = testing.T
            predictions_array = np.stack([test_idc,
                                          predictions,
                                          Y], axis=1)
            predictions_array[predictions_array[:, 0].argsort()]  # sort by idc

    return (epoch, predictions_array)


def main(dataset, output_writer, label_writer, device, config, flags):
    splits = (dataset['training'],
              dataset['testing'],
              dataset['validation'])
    num_classes = dataset['num_classes']

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

    encoders = NeuralEncoders(X, config['encoders'], flags)
    mlp = MLP(input_dim=encoders.out_dim,
              output_dim=num_classes)
    model = nn.Sequential(encoders, mlp)

    if "optim" not in config.keys()\
       or sum([len(c) for c in config["optim"].values()]) <= 0:
        optimizer = optim.Adam(model.parameters(),
                               lr=flags.lr,
                               weight_decay=flags.weight_decay)
    else:
        params = [{"params": model[1].parameters()}]  # MLP
        for modality in flags.modalities:
            if modality not in config["optim"].keys():
                continue

            conf = config["optim"][modality]
            # use hyperparameters specified in config.json
            param = [{"params": enc.parameters()} | conf
                     for enc in model[0].modalities[modality]]

            params.extend(param)

        optimizer = optim.Adam(params,
                               lr=flags.lr,
                               weight_decay=flags.weight_decay)

    loss = nn.CrossEntropyLoss()

    epoch = 1
    if flags.load_checkpoint is not None:
        model.to("cpu")

        print("[LOAD] Loading model state")
        checkpoint = torch.load(flags.load_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']

    epoch, predictions = train_test_model(model, optimizer, loss,
                                          X, splits, epoch,
                                          output_writer, device,
                                          flags)

    if flags.test and flags.save_output:
        for e_idx, c_hat_idx, c_idx in predictions:
            label_writer.writerow([e_idx, c_hat_idx, c_idx])

    return (model, optimizer, loss, epoch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batchsize", help="Number of samples in batch",
                        default=32, type=int)
    parser.add_argument("-c", "--config",
                        help="JSON file with hyperparameters",
                        default=None)
    parser.add_argument("--device", help="Device to run on (e.g., 'cuda:0')",
                        default="cpu", type=str)
    parser.add_argument("-i", "--input", help="Pickled dataset or directory"
                        + " with CSV files (generated by `generateInput.py`)",
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
    parser.add_argument("--shuffle_data", help="Shuffle samples (True)",
                        action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--test", help="Report accuracy on test set",
                        action="store_true")
    parser.add_argument("--weight_decay", help="Weight decay",
                        default=1e-5, type=float)

    flags = parser.parse_args()

    path = flags.input
    data = None
    if path.endswith('.pkl'):
        print("[READ] Found pickled data")
        with open(path, 'rb') as bf:
            data = pickle.load(bf)
    else:
        data = dataset.generate_pickled(flags)

    output_writer = None
    label_writer = None
    if flags.save_output:
        output_writer = TSV(path + "output.tsv", mode='w')
        print("[SAVE] Writing output to %s" % path)

        if flags.test:
            label_writer = TSV(path + "labels.tsv", mode='w')
            label_writer.writerow(['X', 'Y_hat', 'Y'])
            print("[SAVE] Writing labels to %s" % path)

    config = {"encoders": dict(), "optim": dict()}
    if flags.config is not None:
        with open(flags.config, 'r') as f:
            config = json.load(f)

    device = torch.device(flags.device)
    if device.type.startswith("cuda") and not torch.cuda.is_available():
        device = torch.device("cpu")
        print("[DEVICE] device %s not available - falling back to 'cpu'" %
              flags.device)

    model, optimizer, loss, epoch = main(data, output_writer,
                                         label_writer, device,
                                         config, flags)

    if flags.save_checkpoint:
        path = path + "_state_%d.pkl" % epoch
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss}, path)
        print("[SAVE] Writing model state to %s" % path)
