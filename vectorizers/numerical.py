#!/usr/bin/python3

import numpy as np
from rdflib import Literal


def generate_data(g, indices, datatypes):
    entity_to_class_map, entity_to_int_map = indices
    is_varlength = False
    time_dim = -1

    datatypes = list(datatypes)
    data = [list() for dtype in datatypes]
    data_length = [list() for dtype in datatypes]
    data_entity_map = [list() for dtype in datatypes]

    int_to_datatype_map = dict(enumerate(datatypes))
    datatype_to_int_map = {v: k for k, v in int_to_datatype_map.items()}
    seen_datatypes = set()
    for (s, p, o), _ in g.triples((None, None, None), None):
        if type(o) is not Literal or str(o.datatype) not in datatypes:
            continue

        s = str(s)
        s_int = entity_to_int_map[s]

        o_dtype = str(o.datatype)
        o_dtype_int = datatype_to_int_map[o_dtype]

        try:
            o = float(o)
        except ValueError:
            continue

        data[o_dtype_int].append(o)
        data_entity_map[o_dtype_int].append(s_int)
        data_length[o_dtype_int].append(1)

        seen_datatypes.add(o_dtype_int)

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
