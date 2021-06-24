#!/usr/bin/env python

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
    args = sys.argv[1:]
    if len(args) < 1 or len(args) > 5:
        print("USAGE: ./generate_input.py <graph_stripped.hdt> "
              + "[<train_set.csv> <test_set.csv> "
              + "[<valid_set.csv>] [<meta_set.csv>]]")

    hdtfile = args[0]
    g = HDTStore(hdtfile)

    splits = args[1:]
    train, test, valid, meta = None, None, None, None
    if len(splits) >= 2:
        train_path, test_path = splits[:2]
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)

        if len(splits) >= 3:
            valid_path = splits[2]
            valid = pd.read_csv(valid_path)

        if len(splits) >= 4:
            meta_path = splits[3]
            meta = pd.read_csv(meta_path)

    data, splits, df_complete = generate_mapping(g, (train, test, valid, meta))

    df_nodes, df_nodetypes, df_relations, df_triples = data
    df_train, df_test, df_valid, df_meta = splits

    # Write to CSV
    df_nodes.to_csv('nodes.int.csv', index=False, header=True,
                    quoting=csv.QUOTE_NONNUMERIC)
    df_relations.to_csv('relations.int.csv', index=False, header=True,
                        quoting=csv.QUOTE_NONNUMERIC)
    df_nodetypes.to_csv('nodetypes.int.csv', index=False, header=True)
    df_triples.to_csv('triples.int.csv', index=False, header=True)

    df_train.to_csv('training.int.csv', index=False, header=True)
    df_test.to_csv('testing.int.csv', index=False, header=True)
    if df_valid is not None:
        df_valid.to_csv('validation.int.csv', index=False, header=True)
    if df_meta is not None:
        df_meta.to_csv('meta-testing.int.csv', index=False, header=True)
