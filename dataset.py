#!/usr/bin.env python

import argparse
import pickle

from rdflib import URIRef
from rdflib_hdt import HDTStore

from xsd_hierarchy import XSDHierarchy
from utils import mksplits
from vectorizers import numerical, temporal, textual, spatial, visual


def generate_indices(g):
    entity_to_class_map = list()
    entity_to_int_map = dict()
    class_to_int_map = dict()

    i = 0
    j = 0
    for (s, p, o), _ in g.triples((None,
                                   URIRef("https://example.org/hasClass"),
                                   None), None):
        s = str(s)
        entity_to_int_map[s] = i
        i += 1

        o = str(o)
        if o not in class_to_int_map.keys():
            class_to_int_map[o] = j
            j += 1
        o_int = class_to_int_map[o]

        entity_to_class_map.append(o_int)

    return (entity_to_class_map, entity_to_int_map, class_to_int_map)


def generate(hdtfile, flags):
    dataset = dict()
    xsd_tree = XSDHierarchy()

    # read HDT file
    g = HDTStore(hdtfile)

    indices = generate_indices(g)
    entity_to_class_map, entity_to_int_map, _ = indices

    train_idc, test_idc, valid_idc = mksplits(entity_to_class_map,
                                              flags.splits)

    dataset['indices'] = indices
    dataset['train_idc'] = train_idc
    dataset['test_idc'] = test_idc
    dataset['valid_idc'] = valid_idc
    for modality in flags.modalities:
        print("[%s] Generating data" % modality.upper())
        datatypes = set()
        if modality == "textual":
            datatypes = set.union(xsd_tree.descendants("string"),
                                  xsd_tree.descendants("anyURI"))
            datatypes = expand_prefix(datatypes)
            data = textual.generate_data(g, indices, datatypes)
            if len(data) <= 0:
                print(" No information found")
                continue

            dataset[modality] = data

        if modality == "numerical":
            datatypes = set.union(xsd_tree.descendants("numeric"),
                                  xsd_tree.descendants("boolean"))
            datatypes = expand_prefix(datatypes)
            data = numerical.generate_data(g, indices, datatypes)
            if len(data) <= 0:
                print(" No information found")
                continue

            dataset[modality] = data

        if modality == "temporal":
            datatypes = set.union(xsd_tree.descendants("date"),
                                  xsd_tree.descendants("dateTime"),
                                  xsd_tree.descendants("gYear"))
            datatypes = expand_prefix(datatypes)
            data = temporal.generate_data(g, indices, datatypes)
            if len(data) <= 0:
                print(" No information found")
                continue

            dataset[modality] = data

        if modality == "visual":
            datatypes = {"http://kgbench.info/dt#base64Image"}
            data = visual.generate_data(g, indices, datatypes)
            if len(data) <= 0:
                print(" No information found")
                continue

            dataset[modality] = data

        if modality == "spatial":
            datatypes = {"http://www.opengis.net/ont/geosparql#wktLiteral"}
            data = spatial.generate_data(g, indices, datatypes)
            if len(data) <= 0:
                print(" No information found")
                continue

            dataset[modality] = data

    if flags.save_dataset:
        print('Saving data to disk...')
        with open('./%s.pkl' % hdtfile, 'wb') as f:
            pickle.dump(dataset, f)

    return dataset


def expand_prefix(datatypes):
    result = set()
    for datatype in datatypes:
        result.add(datatype.replace("xsd.",
                                    "http://www.w3.org/2001/XMLSchema#"))

    return result


class StoreSplitsAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        splits = tuple(int(v) for v in values.split('/'))
        setattr(namespace, self.dest, splits)
