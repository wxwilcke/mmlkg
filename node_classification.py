#!/usr/bin.env python

import argparse
import json
import sys
from time import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from mmlkg.data import dataset
from mmlkg.data.hdf5 import HDF5
from mmlkg.data.tsv import TSV
from mmlkg.models import MLP, NeuralEncoders
from mmlkg.utils import add_noise_, categorical_accuracy, zero_pad


_MODALITIES = ["textual", "numerical", "temporal", "visual", "spatial"]


def run_once(model, optimizer, loss_function, data,
             samples, device, flags, train=False):
    encoders, discriminator = model

    samples_idx, Y = samples.T
    num_samples = len(samples_idx)

    Y = torch.LongTensor(Y)
    Y_hat = torch.zeros((num_samples, discriminator.output_dim),
                        dtype=torch.float32)

    loss_lst = list()
    acc_lst = list()

    batches = [slice(begin, min(begin+flags.batchsize, num_samples))
               for begin in range(0, num_samples, flags.batchsize)]
    num_batches = len(batches)
    for batch_id, batch in enumerate(batches, 1):
        batch_str = " - batch %2.d / %d" % (batch_id, num_batches)
        print(batch_str, end='\b'*len(batch_str), flush=True)

        batch_idx = samples_idx[batch]

        # encoders
        batch_out_dev = encoders([data, batch_idx, device])

        # descriminator
        Y_hat_batch = discriminator(batch_out_dev).to('cpu')

        # evaluate
        loss_batch = loss_function(Y_hat_batch, Y[batch])
        acc_batch = categorical_accuracy(Y_hat_batch, Y[batch])

        loss_lst.append(float(loss_batch))
        acc_lst.append(float(acc_batch))

        if train:
            optimizer.zero_grad()
            loss_batch.backward()
            optimizer.step()

        Y_hat[batch] = Y_hat_batch.detach()

    return (Y_hat, np.mean(loss_lst), np.mean(acc_lst))


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
                                            flags, train=True)

        if flags.L1lambda > 0:
            l1_regularization = torch.tensor(0.)
            for name, param in model.named_parameters():
                if 'weight' not in name or not name.startswith('W_'):
                    continue
                l1_regularization += torch.sum(param.abs())

            train_loss += flags.L1lambda * l1_regularization

        if flags.L2lambda > 0:
            l2_regularization = torch.tensor(0.)
            for name, param in model.named_parameters():
                if 'weight' not in name or not name.startswith('W_'):
                    continue
                l2_regularization += torch.sum(param ** 2)

            train_loss += flags.L2lambda * l2_regularization

        valid_loss = -1
        valid_acc = -1
        if not flags.test:
            model.eval()
            with torch.no_grad():
                _, valid_loss, valid_acc = run_once(model, None, loss,
                                                    X, validation,
                                                    device, flags)

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
                                    valid_loss, valid_acc,
                                    -1, -1])

    print("[TRAIN] {:.2f}s".format(time()-t0))

    predictions_array = None
    if flags.test:
        model.eval()

        t0 = time()
        with torch.no_grad():
            Y_hat, test_loss, test_acc = run_once(model, None, loss,
                                                  X, testing, device,
                                                  flags)

        print("[TEST] loss: {:.4f} / acc: {:.4f}".format(test_loss,
                                                         test_acc))
        print("[TEST] {:.2f}s".format(time()-t0))

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

        # add noise to input data
        if "encoders" in config.keys()\
           and modality in config["encoders"].keys():
            modconf = config["encoders"][modality]
            m_noise = 0.01 if "m_noise" not in modconf.keys()\
                else modconf["m_noise"]

            if "p_noise" in modconf.keys() and modconf["p_noise"] > 0:
                add_noise_(X[modality], modconf["p_noise"], m_noise)

        # TODO: add structure as modality, via RDF2Vec

    if len(X) <= 0:
        print("No data found - Exiting")
        sys.exit(1)

    encoders = NeuralEncoders(X, config['encoders'])
    mlp = MLP(input_dim=encoders.out_dim,
              output_dim=num_classes)
    model = nn.ModuleList([encoders, mlp])

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

        print("[LOAD] Loading model state", end='')
        checkpoint = torch.load(flags.load_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']

        print(f" - {epoch} epoch")
        epoch += 1

    epoch, predictions = train_test_model(model, optimizer, loss,
                                          X, splits, epoch,
                                          output_writer, device,
                                          flags)

    if flags.test and flags.save_output:
        for e_idx, c_hat_idx, c_idx in predictions:
            label_writer.writerow([e_idx, c_hat_idx, c_idx])

    return (model, optimizer, loss, epoch)


if __name__ == "__main__":
    t_init = "%d" % (time() * 1e7)

    parser = argparse.ArgumentParser()
    parser.add_argument("--batchsize", help="Number of samples in batch",
                        default=32, type=int)
    parser.add_argument("-c", "--config",
                        help="JSON file with hyperparameters",
                        default=None)
    parser.add_argument("--device", help="Device to run on (e.g., 'cuda:0')",
                        default="cpu", type=str)
    parser.add_argument("-i", "--input", help="HDF5 dataset or directory"
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
                        default=0.001, type=float)
    parser.add_argument("--L1lambda", help="L1 normalization lambda",
                        default=0.0, type=float)
    parser.add_argument("--L2lambda", help="L2 normalization lambda",
                        default=0.0, type=float)
    parser.add_argument("-o", "--output", help="Output directory",
                        default=None)
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

    out_dir = flags.input if flags.output is None else flags.output
    out_dir = out_dir + '/' if not out_dir.endswith('/') else out_dir

    data = dict()
    if flags.input.endswith('.h5'):
        print("[READ] Found HDF5 data")
        hf = HDF5(flags.input, 'r')

        data = hf.read_dataset(task=HDF5.NODE_CLASSIFICATION,
                               modalities=flags.modalities)
    else:
        data = dict()
        for name, item in dataset.generate_dataset(flags):
            data[name] = item

        if flags.save_dataset:
            path = out_dir + 'dataset.h5'
            hf = HDF5(path, mode='w')

            print('[SAVE] Saving HDF5 dataset to %s...' % path)
            hf.write_dataset(data, task=HDF5.NODE_CLASSIFICATION)

    output_writer = None
    label_writer = None
    if flags.save_output:
        f_out = out_dir + "output_%s.tsv" % t_init
        output_writer = TSV(f_out, mode='w')
        print("[SAVE] Writing output to %s" % f_out)

        f_json = out_dir + "flags_%s.json" % t_init
        with open(f_json, 'w') as jf:
            json.dump(vars(flags), jf, indent=4)
        print("[SAVE] Writing flags to %s" % f_json)

        if flags.test:
            f_lbl = out_dir + "labels_%s.tsv" % t_init
            label_writer = TSV(f_lbl, mode='w')
            label_writer.writerow(['X', 'Y_hat', 'Y'])
            print("[SAVE] Writing labels to %s" % f_lbl)

    config = {"encoders": dict(), "optim": dict()}
    if flags.config is not None:
        print("[CONF] Using configuration from %s" % flags.config)
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
        f_state = out_dir + "model_state_%s_%d.pkl" % (t_init, epoch)
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss}, f_state)
        print("[SAVE] Writing model state to %s" % f_state)
