#!/usr/bin/python3

from re import sub
from string import punctuation

import numpy as np


# textual
_STR_MAX_CHARS = 512  # one-hot encoded
_VOCAB = [chr(32)] + [chr(i) for i in range(97, 123)]\
        + [chr(i) for i in range(48, 58)]
_VOCAB_MAP = {v: k for k, v in enumerate(_VOCAB)}
_VOCAB_MAX_IDX = len(_VOCAB)


def generate_data(g, datatypes):
    is_varlength = True
    time_dim = 1

    datatypes = list(datatypes)
    data = [list() for dtype in datatypes]
    data_length = [list() for dtype in datatypes]
    data_entity_map = [list() for dtype in datatypes]

    # maps global subject index to global subject index
    num_facts = g.triples.shape[0]
    object_to_subject = np.empty(num_facts, dtype=int)
    object_to_subject[g.triples[:, 2]] = g.triples[:, 0]

    int_to_datatype_map = dict(enumerate(datatypes))
    datatype_to_int_map = {v: k for k, v in int_to_datatype_map.items()}
    seen_datatypes = set()
    for datatype in datatypes:
        datatype_int = datatype_to_int_map[datatype]
        for g_idx in g.datatype_l2g(datatype):
            value, _ = g.i2n[g_idx]

            sequence = None
            seq_length = -1
            try:
                value = str(value)

                sequence = string_preprocess(value)
                sequence = string_encode(sequence)[:_STR_MAX_CHARS]
                seq_length = len(sequence)
            except ValueError:
                continue

            if seq_length <= 0:
                continue

            a = np.zeros(shape=(_VOCAB_MAX_IDX, seq_length), dtype=np.float32)
            a[sequence, range(seq_length)] = 1

            # global idx of entity to which this belongs
            e_int = object_to_subject[g_idx]

            data[datatype_int].append(a)
            data_length[datatype_int].append(seq_length)
            data_entity_map[datatype_int].append(e_int)
            seen_datatypes.add(datatype_int)

    seen_datatypes = list(seen_datatypes)
    data = [data[i] for i in seen_datatypes]
    data_length = [data_length[i] for i in seen_datatypes]
    data_entity_map = [data_entity_map[i] for i in seen_datatypes]

    if len(seen_datatypes) <= 0:
        return list()

    return list(zip([int_to_datatype_map[i] for i in seen_datatypes],
                    data, data_length, data_entity_map,
                    [is_varlength for _ in seen_datatypes],
                    [time_dim for _ in seen_datatypes]))


def string_encode(seq):
    return [_VOCAB_MAP[char] for char in seq if char in _VOCAB]


def string_preprocess(seq):
    seq = seq.lower()
    seq = sub('['+punctuation+']', '', seq).split()

    return " ".join(seq)
