#!/usr/bin/python3

import base64
from io import BytesIO
from PIL import Image

import numpy as np
from rdflib.term import Literal


_IMG_SIZE = (64, 64)
_IMG_MODE = "RGB"


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

        value = str(o)
        blob = None
        try:
            blob = b64_to_img(value)
            blob = downsample(blob)
        except ValueError:
            continue

        # add to matrix structures
        a = np.array(blob, dtype=np.float32)
        if _IMG_MODE == "RGB":
            # from WxHxC to CxWxH
            a = a.transpose((0, 2, 1)).transpose((1, 0, 2))

        o_dtype = str(o.datatype)
        o_dtype_int = datatype_to_int_map[o_dtype]
        seen_datatypes.add(o_dtype_int)

        data[o_dtype_int].append(a)
        data_length[o_dtype_int].append(-1)
        data_entity_map[o_dtype_int].append(s_int)

    seen_datatypes = list(seen_datatypes)
    data = [data[i] for i in seen_datatypes]
    data_length = [data_length[i] for i in seen_datatypes]
    data_entity_map = [data_entity_map[i] for i in seen_datatypes]

    if len(seen_datatypes) <= 0:
        return list()

    # normalization over channels
    for i in range(len(data)):
        a = np.array(data[i])

        for ch in range(a.shape[1]):
            value_min = a[:, i, :, :].min()
            value_max = a[:, i, :, :].max()

            a[:, i, :, :] = (2 * (a[:, i, :, :] - value_min) /
                             (value_max - value_min)) - 1.0

        data[i] = [r for r in a]

    return list(zip([int_to_datatype_map[i] for i in seen_datatypes],
                    data, data_length, data_entity_map,
                    [is_varlength for _ in seen_datatypes],
                    [time_dim for _ in seen_datatypes]))


def b64_to_img(b64string):
    im = Image.open(BytesIO(base64.urlsafe_b64decode(b64string.encode())))
    if im.mode != _IMG_MODE:
        im = im.convert(_IMG_MODE)

    return im


def downsample(im):
    if im.size != _IMG_SIZE:
        return im.resize(_IMG_SIZE)

    return im
