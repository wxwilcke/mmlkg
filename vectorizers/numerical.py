#!/usr/bin/python3

import numpy as np


def generate_data(g, datatypes):
    is_varlength = False
    time_dim = -1

    datatypes = list(datatypes)
    data = [list() for dtype in datatypes]
    data_length = [list() for dtype in datatypes]
    data_entity_map = [list() for dtype in datatypes]

    int_to_datatype_map = dict(enumerate(datatypes))
    datatype_to_int_map = {v: k for k, v in int_to_datatype_map.items()}
    seen_datatypes = set()
    for datatype in datatypes:
        datatype_int = datatype_to_int_map[datatype]
        for g_idx in g.datatype_l2g(datatype):
            value, _ = g.i2n[g_idx]
            try:
                value = float(value)
            except ValueError:
                continue

            # global idx of entity to which this belongs
            e_int = g.triples[np.where(g.triples[:, 2] == g_idx)][0][0]

            data[datatype_int].append(value)
            data_entity_map[datatype_int].append(e_int)
            data_length[datatype_int].append(1)

            seen_datatypes.add(datatype_int)

    seen_datatypes = list(seen_datatypes)
    data = [data[i] for i in seen_datatypes]
    data_length = [data_length[i] for i in seen_datatypes]
    data_entity_map = [data_entity_map[i] for i in seen_datatypes]

    if len(seen_datatypes) <= 0:
        return list()

    for i in range(len(data)):
        a = np.array(data[i])
        v_min = a.min()
        v_max = a.max()

        a = (2 * (a - v_min) / (v_max - v_min)) - 1.0

        data[i] = [[e] for e in a]

    return list(zip([int_to_datatype_map[i] for i in seen_datatypes],
                    data, data_length, data_entity_map,
                    [is_varlength for _ in seen_datatypes],
                    [time_dim for _ in seen_datatypes]))
