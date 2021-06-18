#!/usr/bin.env python

import numpy as np
import torch.nn.functional as f


def mkbatches(sample_idx, batch_size=1):
    """ split array in batches
    """
    n = len(sample_idx)  # number of samples
    batch_size = min(n, batch_size)
    idc = np.arange(n, dtype=np.int32)

    idc_assignments = np.array_split(idc, n/batch_size)
    sample_assignments = [np.array(sample_idx, dtype=np.int32)[slce]
                          for slce in idc_assignments]

    return list(zip(idc_assignments, sample_assignments))


def mkbatches_varlength(sample_idx, seq_length_map,
                        batch_size=1):
    n = len(sample_idx)
    batch_size = min(n, batch_size)

    # sort on length
    idc = np.arange(n, dtype=np.int32)
    _, sequences_sorted_idc = zip(*sorted(zip(seq_length_map, idc)))

    seq_assignments = np.array_split(sequences_sorted_idc, n/batch_size)
    sample_assignments = [np.array(sample_idx, dtype=np.int32)[slce]
                          for slce in seq_assignments]

    return list(zip(seq_assignments, sample_assignments))


def zero_pad(samples, time_dim):
    if time_dim < 0:
        return samples

    max_height = max([t.shape[0] for t in samples])
    max_width = max([t.shape[1] for t in samples])

    return [f.pad(t, [0, max_width-t.shape[1], 0, max_height-t.shape[0]])
            for t in samples]


def mksplits(entity_to_class_map, splits):
    classes, counts = np.unique(entity_to_class_map, return_counts=True)
    train, test, valid = list(), list(), list()
    for i, c in enumerate(classes):
        num_test = round(counts[i] * (splits[1]/100))
        num_train = counts[i] - num_test
        num_valid = round(num_train * (splits[2]/100))
        num_train -= num_valid

        class_idx = np.where(entity_to_class_map == c)[0]
        np.random.shuffle(class_idx)

        train.extend(class_idx[:num_train])
        valid.extend(class_idx[num_train:(num_train+num_valid)])
        test.extend(class_idx[-num_test:])

    return (train, test, valid)


def categorical_accuracy(Y_hat, Y_ground):
    predictions = Y_hat.max(axis=1)[1]
    num_correct = sum(predictions == Y_ground)

    return num_correct / len(Y_ground)
