#!/usr/bin/env python

import argparse
import csv
import sys

import pandas as pd
from rdflib.term import BNode, Literal, URIRef
from rdflib_hdt import HDTStore


def _node_type(node):
    if isinstance(node, BNode):
        return "blank_node"
    elif isinstance(node, URIRef):
        return "iri"
    elif isinstance(node, Literal):
        if node.datatype is not None:
            return str(node.datatype)
        elif node.language is not None:
            return '@' + node.language
        else:
            return "none"
    else:
        raise Exception()


def _generate_context(g):
    entities = set()
    relations = set()
    datatypes = set()

    for (s, p, o), _ in g.triples((None, None, None), None):
        s_type = _node_type(s)
        o_type = _node_type(o)

        datatypes.add(s_type)
        datatypes.add(o_type)

        entities.add((str(s), s_type))
        entities.add((str(o), o_type))

        relations.add(str(p))

    i2e = list(entities)
    i2r = list(relations)
    i2d = list(datatypes)

    # ensure deterministic order
    i2e.sort()
    i2r.sort()
    i2d.sort()

    e2i = {e: i for i, e in enumerate(i2e)}
    r2i = {r: i for i, r in enumerate(i2r)}

    triples = pd.DataFrame([[e2i[(str(s), _node_type(s))],
                             r2i[str(p)],
                             e2i[(str(o), _node_type(o))]]
                            for (s, p, o), _ in list(g.triples((None,
                                                                None,
                                                                None),
                                                               None))],
                           columns=["index_lhs_node",
                                    "index_relation",
                                    "index_rhs_node"])

    nodes = [(i, dt, ent) for i, (ent, dt) in enumerate(i2e)]
    nodes = pd.DataFrame(nodes, columns=['index', 'annotation', 'label'])
    relations = pd.DataFrame(enumerate(i2r), columns=['index', 'label'])
    nodetypes = pd.DataFrame(enumerate(i2d), columns=['index', 'annotation'])

    return ((nodes, nodetypes, relations, triples), e2i, r2i)


def _generate_splits(splits, e2i, r2i):
    """ Expect splits as CSV files with two (anonymous) columns: node and class
    """
    classes = pd.unique(pd.concat([df.iloc[:, 1] for df in splits
                                   if df is not None]))
    c2i = {c: i for i, c in enumerate(classes)}

    df_train, df_test, df_valid, df_meta = splits
    df_train = pd.DataFrame(zip([e2i[e, "iri"] for e in df_train.iloc[:, 0]],
                                [c2i[c] for c in df_train.iloc[:, 1]]),
                            columns=["node_index", "class_index"])
    df_test = pd.DataFrame(zip([e2i[e, "iri"] for e in df_test.iloc[:, 0]],
                               [c2i[c] for c in df_test.iloc[:, 1]]),
                           columns=["node_index", "class_index"])

    df_complete = pd.concat([df_train, df_test])
    if df_valid is not None:
        df_valid = pd.DataFrame(zip(
            [e2i[e, "iri"] for e in df_valid.iloc[:, 0]],
            [c2i[c] for c in df_valid.iloc[:, 1]]),
                                columns=["node_index", "class_index"])

        df_complete = pd.concat([df_complete, df_valid])

    if df_meta is not None:
        df_meta = pd.DataFrame(zip(
            [e2i[e, "iri"] for e in df_meta.iloc[:, 0]],
            [c2i[c] for c in df_meta.iloc[:, 1]]),
                                columns=["node_index", "class_index"])

        df_complete = pd.concat([df_complete, df_meta])

    df_complete.columns = ["node_index", "class_index"]

    return ((df_train, df_test, df_valid, df_meta), df_complete)


def generate_mapping(g, splits):
    data, e2i, r2i = _generate_context(g)
    splits, complete = _generate_splits(splits, e2i, r2i)

    return (data, splits, complete)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="HDT graph",
                        required=True)
    parser.add_argument("-d", "--dir", help="Output directory", default="./")
    parser.add_argument("-ts", "--train", help="Training set (CSV) with "
                        + "samples on the left-hand side and their classes "
                        + "on the right",
                        required=True)
    parser.add_argument("-ws", "--test", help="Withheld set (CSV) for testing "
                        + "with samples on the left-hand side and their "
                        + "classes on the right",
                        required=True)
    parser.add_argument("-vs", "--valid", help="Validation set (CSV) with "
                        + "samples on the left-hand side and their classes "
                        + "on the right", default=None)
    parser.add_argument("-ms", "--meta", help="Meta withheld set (CSV) with "
                        + "samples on the left-hand side and their classes "
                        + "on the right", default=None)
    flags = parser.parse_args()

    hdtfile = flags.input
    g = HDTStore(hdtfile)

    train = pd.read_csv(flags.train)
    test = pd.read_csv(flags.test)

    valid, meta = None, None
    if flags.valid is not None:
        valid = pd.read_csv(flags.valid)
    if flags.meta is not None:
        meta = pd.read_csv(flags.meta)

    data, splits, df_complete = generate_mapping(g, (train, test, valid, meta))

    df_nodes, df_nodetypes, df_relations, df_triples = data
    df_train, df_test, df_valid, df_meta = splits

    # Write to CSV
    path = flags.dir if flags.dir.endswith('/') else flags.dir + '/'

    df_nodes.to_csv(path+'nodes.int.csv', index=False, header=True,
                    quoting=csv.QUOTE_NONNUMERIC)
    df_relations.to_csv(path+'relations.int.csv', index=False, header=True,
                        quoting=csv.QUOTE_NONNUMERIC)
    df_nodetypes.to_csv(path+'nodetypes.int.csv', index=False, header=True)
    df_triples.to_csv('triples.int.csv', index=False, header=True)

    df_train.to_csv(path+'training.int.csv', index=False, header=True)
    df_test.to_csv(path+'testing.int.csv', index=False, header=True)
    if df_valid is not None:
        df_valid.to_csv(path+'validation.int.csv', index=False, header=True)
    if df_meta is not None:
        df_meta.to_csv(path+'meta-testing.int.csv', index=False, header=True)
