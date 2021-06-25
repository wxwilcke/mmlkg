#!/usr/bin/python3

import base64
from io import BytesIO
from PIL import Image

import numpy as np


_IMG_SIZE = (64, 64)
_IMG_MODE = "RGB"


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

            # global idx of entity to which this belongs
            e_int = g.triples[np.where(g.triples[:, 2] == g_idx)][0][0]

            seen_datatypes.add(datatype_int)

            data[datatype_int].append(a)
            data_length[datatype_int].append(-1)
            data_entity_map[datatype_int].append(e_int)

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
