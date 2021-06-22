#!/usr/bin/python3

from re import sub
from string import punctuation

import numpy as np
from rdflib import Literal


# textual
_STR_MAX_CHARS = 512  # one-hot encoded
_VOCAB = [chr(32)] + [chr(i) for i in range(97, 123)]\
        + [chr(i) for i in range(48, 58)]
_VOCAB_MAP = {v: k for k, v in enumerate(_VOCAB)}
_VOCAB_MAX_IDX = len(_VOCAB)


def generate_data(g, indices, datatypes):
    entity_to_class_map, entity_to_int_map, _ = indices
    is_varlength = True
    time_dim = 1

    datatypes = list(datatypes)
    data = [list() for dtype in datatypes]
    data_length = [list() for dtype in datatypes]
    data_entity_map = [list() for dtype in datatypes]

    int_to_datatype_map = dict(enumerate(datatypes))
    datatype_to_int_map = {v: k for k, v in int_to_datatype_map.items()}
    seen_datatypes = set()
    for (s, p, o), _ in g.triples((None, None, None), None):
        if type(o) is not Literal or (str(o.datatype) not in datatypes
                                      and str(o.language) is not None):
            continue

        s = str(s)
        s_int = entity_to_int_map[s]

        if o.datatype is None:
            # if has language tag
            o_dtype = "http://www.w3.org/2001/XMLSchema#string"
        else:
            o_dtype = str(o.datatype)
        o_dtype_int = datatype_to_int_map[o_dtype]

        sequence = None
        seq_length = -1
        try:
            value = str(o)

            sequence = string_preprocess(value)
            sequence = string_encode(sequence)[:_STR_MAX_CHARS]
            seq_length = len(sequence)
        except ValueError:
            continue

        if seq_length <= 0:
            continue

        a = np.zeros(shape=(_VOCAB_MAX_IDX, seq_length), dtype=np.int8)
        a[sequence, range(seq_length)] = 1

        data[o_dtype_int].append(a)
        data_length[o_dtype_int].append(seq_length)
        data_entity_map[o_dtype_int].append(s_int)
        seen_datatypes.add(o_dtype_int)

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
