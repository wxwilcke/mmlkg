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
from mmlkg.models import DistMult, NeuralEncoders
from mmlkg.utils import (add_noise_, binary_crossentropy,
                         entity_to_entity_triples, global_to_local)


_MODALITIES = ["textual", "numerical", "temporal", "visual", "spatial"]


def filter_scores_(scores, batch_data, heads_and_tails, head=True):
    heads, tails = heads_and_tails

    # set scores of existing facts to -inf
    indices = list()
    for i, (s, p, o) in enumerate(batch_data):
        s, p, o = (s.item(), p.item(), o.item())
        if head:
            indices.extend([(i, si) for si in heads[p, o] if si != s])
        else:
            indices.extend([(i, oi) for oi in tails[s, p] if oi != o])
        # we add the indices of all know triples except the one corresponding
        # to the target triples.

    if len(indices) <= 0:
        return

    indices = torch.tensor(indices)
    scores[indices[:, 0], indices[:, 1]] = float('-inf')


def truedicts(data):
    heads = dict()
    tails = dict()
    for split in data:
        for i in range(split.shape[0]):
            fact = split[i]
            s, p, o = fact[0].item(), fact[1].item(), fact[2].item()

            if (p, o) not in heads.keys():
                heads[(p, o)] = list()
            if (s, p) not in tails.keys():
                tails[(s, p)] = list()

            heads[(p, o)].append(s)
            tails[(s, p)].append(o)

    return heads, tails


def compute_ranks_fast(model, node_features, data, heads_and_tails, filtered,
                       devices, flags):
    encoders, distmult = model
    encoder_device, distmult_device = devices

    # compute feature embeddings
    feature_embeddings = None
    if not flags.featureless:
        X, X_idc = node_features
        features = [X, X_idc, encoder_device]
        feature_embeddings = encoders(features).to(distmult_device)

    batch_size = flags.batchsize_mrr
    num_facts = data.shape[0]
    num_nodes = distmult.node_embeddings.shape[0]
    num_batches = int((num_facts + batch_size-1)//batch_size)
    ranks = torch.empty((num_facts*2), dtype=torch.int64)
    for head in [False, True]:  # head or tail prediction
        offset = int(head) * num_facts
        for batch_id in range(num_batches):
            batch_begin = batch_id * batch_size
            batch_end = min(num_facts, (batch_id+1) * batch_size)

            batch_data = data[batch_begin:batch_end]
            batch_num_facts = batch_data.shape[0]

            # compute the full score matrix (filter later)
            bases = batch_data[:, 1:] if head else batch_data[:, :2]
            targets = batch_data[:, 0] if head else batch_data[:, 2]

            # collect the triples for which to compute scores
            bexp = bases.view(batch_num_facts, 1, 2).expand(batch_num_facts,
                                                            num_nodes, 2)
            ar = torch.arange(num_nodes).view(1, num_nodes, 1)
            ar = ar.expand(batch_num_facts, num_nodes, 1)
            candidates = torch.cat([ar, bexp] if head else [bexp, ar], dim=2)

            candidates_dev = candidates.to(distmult_device)
            scores = distmult([(candidates_dev[:, :, 0],
                                candidates_dev[:, :, 1],
                                candidates_dev[:, :, 2]),
                               feature_embeddings]).to('cpu')

            if distmult_device != torch.device('cpu'):
                del candidates_dev
                torch.cuda.empty_cache()

            # filter out the true triples that aren't the target
            if filtered:
                filter_scores_(scores, batch_data, heads_and_tails, head=head)

            # Select the true scores, and count the number of values larger
            true_scores = scores[torch.arange(batch_num_facts), targets]
            true_scores_view = true_scores.view(batch_num_facts, 1)
            batch_ranks = torch.sum(scores > true_scores_view, dim=1,
                                    dtype=torch.int64)
            # -- This is the "optimistic" rank (assuming it's sorted
            #    to the front of the ties)
            num_ties = torch.sum(scores == true_scores_view, dim=1,
                                 dtype=torch.int64)

            # Account for ties (put the true example halfway down the ties)
            batch_ranks = batch_ranks + torch.round((num_ties - 1) / 2).long()

            ranks[offset+batch_begin:offset+batch_end] = batch_ranks

    return ranks + 1


def train_once(model, optimizer, loss_function, X, X_idc,
               data, devices, flags):
    encoders, distmult = model
    encoder_device, distmult_device = devices

    distmult.train()
    if not flags.featureless:
        encoders.train()

    # sample negative triples by copying and corrupting positive triples
    num_samples = data.shape[0]
    num_corrupt = num_samples//5
    num_corrupt_head = num_corrupt//2
    num_corrupt_tail = num_corrupt - num_corrupt_head

    neg_samples_idx = torch.from_numpy(np.random.choice(np.arange(num_samples),
                                                        num_corrupt,
                                                        replace=False))
    corrupted_data = torch.from_numpy(np.copy(data[neg_samples_idx]))

    # corrupt elements by replacing them with random elements
    # note that a small amount may still exist
    num_nodes = distmult.node_embeddings.shape[0]
    corrupt_heads = np.random.choice(num_nodes,
                                     num_corrupt_head,
                                     replace=True)
    corrupt_tails = np.random.choice(num_nodes,
                                     num_corrupt_tail,
                                     replace=True)

    corrupted_data[:num_corrupt_head, 0] = torch.from_numpy(corrupt_heads)
    corrupted_data[-num_corrupt_tail:, 2] = torch.from_numpy(corrupt_tails)

    # create labels; positive samples are 1, negative 0
    Y = torch.ones((num_samples+num_corrupt), dtype=torch.float32)
    Y[-num_corrupt:] = 0.

    # compute feature embeddings
    feature_embeddings = None
    if not flags.featureless:
        features = [X, X_idc, encoder_device]
        feature_embeddings = encoders(features).to(distmult_device)

    # compute scores
    Y_hat = torch.empty((num_samples+num_corrupt), dtype=torch.float32)

    data_dev = data.to(distmult_device)
    Y_hat[:num_samples] = distmult([(data_dev[:, 0],
                                     data_dev[:, 1],
                                     data_dev[:, 2]),
                                    feature_embeddings]).to('cpu')

    if distmult_device != torch.device('cpu'):
        # clear GPU memory
        del data_dev
        torch.cuda.empty_cache()

    corrupted_data_dev = corrupted_data.to(distmult_device)
    Y_hat[-num_corrupt:] = distmult([(corrupted_data_dev[:, 0],
                                      corrupted_data_dev[:, 1],
                                      corrupted_data_dev[:, 2]),
                                     feature_embeddings]).to('cpu')

    # compute loss
    loss = binary_crossentropy(Y_hat, Y, loss_function)

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()  # training loss
    optimizer.step()

    return float(loss)


def test_once(model, node_features, data, heads_and_tails, devices, flags):
    encoders, distmult = model

    distmult.eval()
    if not flags.featureless:
        encoders.eval()

    mrr = dict()
    hits_at_k = dict()
    rankings = dict()
    with torch.no_grad():
        for filtered in [False, True]:
            rank_type = "flt" if filtered else "raw"
            if filtered is True and not flags.filter_ranks:
                mrr[rank_type] = -1
                hits_at_k[rank_type] = [-1, -1, -1]
                rankings[rank_type] = [-1]

                continue

            ranks = compute_ranks_fast(model,
                                       node_features,
                                       data,
                                       heads_and_tails,
                                       filtered,
                                       devices,
                                       flags)

            mrr[rank_type] = torch.mean(1.0 / ranks.float()).item()
            hits_at_k[rank_type] = list()
            for k in [1, 3, 10]:
                rank = float(torch.mean((ranks <= k).float()))
                hits_at_k[rank_type].append(rank)

            ranks = ranks.tolist()
            rankings[rank_type] = ranks

    return (mrr, hits_at_k, rankings)


def train_test_model(model, optimizer, loss_function, X, X_idc, splits, epoch,
                     output_writer, devices, flags):
    if flags.save_output:
        header = ["epoch", "loss"]
        for split in ["train", "valid", "test"]:
            header.extend([split+"_mrr_raw", split+"_H@1_raw",
                           split+"_H@3_raw", split+"_H@10_raw",
                           split+"_mrr_flt", split+"_H@1_flt",
                           split+"_H@3_flt", split+"_H@10_flt"])
        output_writer.writerow(header)

    # gather the heads and tails used in positive samples
    heads, tails = truedicts(splits)

    # combine training and validation set when evaluating on test data
    training, testing, validation = splits
    if flags.test:
        training = torch.cat([training, validation], dim=0)
        validation = None

    if flags.shuffle_data:
        training = training[torch.randperm(training.shape[0]), :]
        testing = testing[torch.randperm(testing.shape[0]), :]

        if validation is not None:
            validation = validation[torch.randperm(validation.shape[0]), :]

    nepoch = epoch + flags.num_epoch
    # Log wall-clock time
    t0 = time()
    for epoch in range(epoch, nepoch):
        print("[TRAIN] %3.d " % epoch, end='', flush=True)

        loss = train_once(model, optimizer, loss_function, X, X_idc,
                          training, devices, flags)

        if flags.L1lambda > 0:
            l1_regularization = torch.tensor(0.)
            for name, param in model.named_parameters():
                if 'weight' not in name or not name.startswith('W_'):
                    continue
                l1_regularization += torch.sum(param.abs())

            loss += flags.L1lambda * l1_regularization

        if flags.L2lambda > 0:
            l2_regularization = torch.tensor(0.)
            for name, param in model.named_parameters():
                if 'weight' not in name or not name.startswith('W_'):
                    continue
                l2_regularization += torch.sum(param ** 2)

            loss += flags.L2lambda * l2_regularization

        print_str = f" - loss: {loss:.4f}"
        result_str = [str(epoch), str(loss)]
        if epoch % flags.eval_interval == 0 or epoch == nepoch-1:
            train_mrr, train_hits, _ = test_once(model, (X, X_idc),
                                                 training,
                                                 (heads, tails),
                                                 devices, flags)

            result_str.extend([str(train_mrr['raw']),
                               str(train_hits['raw'][0]),
                               str(train_hits['raw'][1]),
                               str(train_hits['raw'][2]),
                               str(train_mrr['flt']),
                               str(train_hits['flt'][0]),
                               str(train_hits['flt'][1]),
                               str(train_hits['flt'][2])])

            print_str += f" | MMR {train_mrr['raw']:.4f} (raw) /"\
                         f" {train_mrr['flt']:.4f} (filtered)"

            if not flags.test:
                valid_mrr, valid_hits, _ = test_once(model, (X, X_idc),
                                                     validation,
                                                     (heads, tails),
                                                     devices, flags)

                result_str.extend([str(valid_mrr['raw']),
                                   str(valid_hits['raw'][0]),
                                   str(valid_hits['raw'][1]),
                                   str(valid_hits['raw'][2]),
                                   str(valid_mrr['flt']),
                                   str(valid_hits['flt'][0]),
                                   str(valid_hits['flt'][1]),
                                   str(valid_hits['flt'][2])])

                print_str += f" | [VALID] MMR {valid_mrr['raw']:.4f} (raw) /"\
                             f" {valid_mrr['flt']:.4f} (filtered)"
            else:
                # add valid set placeholder
                result_str.extend([-1, -1, -1, -1, -1, -1, -1, -1])
        else:
            # add train and valid set placeholder
            result_str.extend([-1 for _ in range(16)])

        print(print_str, flush=True)

        # add test set placeholder
        result_str.extend([-1, -1, -1, -1, -1, -1, -1, -1])
        if flags.save_output:
            output_writer.writerow(result_str)

    print("[TRAIN] {:.2f}s".format(time()-t0))

    ranks = None
    if flags.test:
        t0 = time()
        test_mrr, test_hits, ranks = test_once(model, (X, X_idc), testing,
                                               (heads, tails), devices,
                                               flags)

        print(f"[TEST] MRR {test_mrr['flt']:.4f} (raw) - "
              f"H@1 {test_hits['raw'][0]:.4f} / "
              f"H@3 {test_hits['raw'][1]:.4f} / "
              f"H@10 {test_hits['raw'][2]:.4f} | "
              f"MRR {test_mrr['flt']:.4f} (filtered) - "
              f"H@1 {test_hits['flt'][0]:.4f} / "
              f"H@3 {test_hits['flt'][1]:.4f} / "
              f"H@10 {test_hits['flt'][2]:.4f}", flush=True)
        print("[TEST] {:.2f}s".format(time()-t0))

        if flags.save_output:
            result_str = [-1 for _ in range(18)]
            result_str.extend([str(test_mrr['raw']),
                               str(test_hits['raw'][0]),
                               str(test_hits['raw'][1]),
                               str(test_hits['raw'][2]),
                               str(test_mrr['flt']),
                               str(test_hits['flt'][0]),
                               str(test_hits['flt'][1]),
                               str(test_hits['flt'][2])])
            output_writer.writerow(result_str)

    return (epoch, ranks)


def main(dataset, output_writer, ranks_writer, devices, config, flags):
    X = dict()
    if not flags.featureless:
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

                if "p_noise" in modconf.keys():
                    add_noise_(X[modality], modconf["p_noise"], m_noise)

        if len(X) <= 0:
            print("No data found - Exiting")
            sys.exit(1)

    data = torch.from_numpy(dataset['triples'])  # N x (s, p, o)
    training = data[dataset['training_lp']]
    testing = data[dataset['testing_lp']]
    validation = data[dataset['validation_lp']]

    entities_idc = dataset['entities']
    # rmv triples with datatype properties
    # also do this for featureless learning to make the results comparable
    training = entity_to_entity_triples(training, entities_idc)
    testing = entity_to_entity_triples(testing, entities_idc)
    validation = entity_to_entity_triples(validation, entities_idc)

    # remap global indices to local indices of embeddings
    training = global_to_local(training, entities_idc)
    testing = global_to_local(testing, entities_idc)
    validation = global_to_local(validation, entities_idc)

    # all objecttype properties
    relations = list(set().union(training[:, 1],
                                 testing[:, 1],
                                 validation[:, 1]))

    encoders = None
    distmult = None
    encoder_device, distmult_device = devices
    if flags.featureless:
        print("[MODE] DistMult (Featureless)")
        distmult = DistMult(num_entities=len(entities_idc),
                            num_relations=len(relations),
                            literalE=False)
    else:
        print("[MODE] DistMult + LiteralE")
        encoders = NeuralEncoders(X, config['encoders'], flags)
        distmult = DistMult(num_entities=len(entities_idc),
                            num_relations=len(relations),
                            literalE=True,
                            feature_dim=encoders.out_dim)

    model = nn.ModuleList([encoders, distmult])

    # set encoder-specific optimizer options if specified
    if "optim" not in config.keys()\
       or sum([len(c) for c in config["optim"].values()]) <= 0:
        optimizer = optim.Adam(model.parameters(),
                               lr=flags.lr,
                               weight_decay=flags.weight_decay)
    else:
        params = [{"params": distmult.parameters()}]
        for modality in flags.modalities:
            if modality not in config["optim"].keys():
                continue

            conf = config["optim"][modality]
            # use hyperparameters specified in config.json
            param = [{"params": enc.parameters()} | conf
                     for enc in encoders.modalities[modality]]

            params.extend(param)

        optimizer = optim.Adam(params,
                               lr=flags.lr,
                               weight_decay=flags.weight_decay)

    loss = nn.BCEWithLogitsLoss()

    # load saved model state
    epoch = 1
    if flags.load_checkpoint is not None:
        print("[LOAD] Loading model state", end='')
        checkpoint = torch.load(flags.load_checkpoint)
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        distmult.load_state_dict(checkpoint['distmult_model_state_dict'])
        if not flags.featureless:
            encoders.load_state_dict(checkpoint['encoders_model_state_dict'])

        print(f" - {epoch} epoch")
        epoch += 1

    X_idc = np.array(entities_idc)
    splits = (training,
              testing,
              validation)

    # move to device and initialize weights
    distmult.to(distmult_device)
    distmult.init()
    if not flags.featureless:
        encoders.to(encoder_device)
        encoders.init()

    epoch, ranks = train_test_model(model, optimizer, loss,
                                    X, X_idc, splits, epoch,
                                    output_writer, devices,
                                    flags)

    if flags.test and flags.save_output:
        if flags.filter_ranks:
            ranks_writer.writerow(['raw', 'filtered'])
            ranks_writer.writerows(zip(ranks['raw'],
                                       ranks['flt']))
        else:
            ranks_writer.writerow(['raw'])
            for row in ranks['raw']:
                ranks_writer.writerow(row)

    return (model, optimizer, loss, epoch)


if __name__ == "__main__":
    t_init = "%d" % (time() * 1e7)

    parser = argparse.ArgumentParser()
    parser.add_argument("--batchsize", help="Number of samples in batch",
                        default=32, type=int)
    parser.add_argument("--batchsize_mrr", help="Number of samples in "
                        + "MRR batch", default=32, type=int)
    parser.add_argument("-c", "--config",
                        help="JSON file with hyperparameters",
                        default=None)
    parser.add_argument("--distmult_device", help="Device to run DistMult on "
                        + "(e.g., 'cuda:0')", default="cpu", type=str)
    parser.add_argument("--eval_interval", help="Number of epoch between "
                        + "MRR evaluations", default=10, type=int)
    parser.add_argument("--feature_device", help="Device to learn feature "
                        + "embeddings on (e.g., 'cuda:0')", default="cpu",
                        type=str)
    parser.add_argument("--featureless", help="Learn without features "
                        + "(structure only)", action="store_true")
    parser.add_argument("--filter_ranks", help="Compute raw AND filtered MRR"
                        + " ranks", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("-i", "--input", help="HDF5 dataset or directory"
                        + " with CSV files (generated by `generateInput.py`)",
                        type=str, required=True)
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
    parser.add_argument("--save_dataset_and_exit", help="Save dataset to disk "
                        + "and exit", action="store_true")
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

        data = hf.read_dataset(task=HDF5.LINK_PREDICTION,
                               modalities=flags.modalities)
    else:
        data = dict()
        for name, item in dataset.generate_dataset(flags):
            data[name] = item

        if flags.save_dataset:
            path = out_dir + 'dataset.h5'
            hf = HDF5(path, mode='w')

            print('[SAVE] Saving HDF5 dataset to %s...' % path)
            hf.write_dataset(data, task=HDF5.LINK_PREDICTION)

    output_writer = None
    ranks_writer = None
    if flags.save_output:
        f_out = out_dir + "output_%s.tsv" % t_init
        output_writer = TSV(f_out, mode='w')
        print("[SAVE] Writing output to %s" % f_out)

        f_json = out_dir + "flags_%s.json" % t_init
        with open(f_json, 'w') as jf:
            json.dump(vars(flags), jf, indent=4)
        print("[SAVE] Writing flags to %s" % f_json)

        if flags.test:
            f_lbl = out_dir + "ranks_%s.tsv" % t_init
            ranks_writer = TSV(f_lbl, mode='w')
            print("[SAVE] Writing ranks to %s" % f_lbl)

    config = {"encoders": dict(), "optim": dict()}
    if flags.config is not None:
        print("[CONF] Using configuration from %s" % flags.config)
        with open(flags.config, 'r') as f:
            config = json.load(f)

    feature_device = torch.device(flags.feature_device)
    distmult_device = torch.device(flags.distmult_device)
    if (feature_device.type.startswith("cuda")
        or distmult_device.type.startswith("cuda"))\
       and not torch.cuda.is_available():
        feature_device = torch.device("cpu")
        distmult_device = torch.device("cpu")
        print("[DEVICE] GPU not available - falling back to CPU")

    devices = (feature_device, distmult_device)

    model, optimizer, loss, epoch = main(data, output_writer,
                                         ranks_writer, devices,
                                         config, flags)

    if flags.save_checkpoint:
        f_state = out_dir + "model_state_%s_%d.pkl" % (t_init, epoch)
        torch.save({'epoch': epoch,
                    'encoders_model_state_dict': model[0].state_dict(),
                    'distmult_model_state_dict': model[1].state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss}, f_state)
        print("[SAVE] Writing model state to %s" % f_state)
