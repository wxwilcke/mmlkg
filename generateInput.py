#!/usr/bin/env python

import argparse
import csv

import pandas as pd
from rdflib.namespace import XSD
from rdflib.term import BNode, Literal, URIRef
from rdflib_hdt import HDTStore


def _node_str(node):
    value = str(node)

    # avoid rdflib bug which serializes years as dates
    if isinstance(node, Literal) and node.datatype is XSD.gYear:
        value = value[:4]

    return value


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


def _generate_context(graphs):
    entities = set()
    relations = set()
    datatypes = set()

    for g in graphs:
        for (s, p, o), _ in g.triples((None, None, None), None):
            s_type = _node_type(s)
            o_type = _node_type(o)

            datatypes.add(s_type)
            datatypes.add(o_type)

            entities.add((_node_str(s), s_type))
            entities.add((_node_str(o), o_type))

            relations.add(_node_str(p))

    i2e = list(entities)
    i2r = list(relations)
    i2d = list(datatypes)

    # ensure deterministic order
    i2e.sort()
    i2r.sort()
    i2d.sort()

    e2i = {e: i for i, e in enumerate(i2e)}
    r2i = {r: i for i, r in enumerate(i2r)}

    triples = list()
    for g in graphs:
        for (s, p, o), _ in g.triples((None, None, None), None):
            triples.append([e2i[(_node_str(s), _node_type(s))],
                            r2i[_node_str(p)],
                            e2i[(_node_str(o), _node_type(o))]])

    triples = pd.DataFrame(triples,
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
    classes = set()
    for df in splits:
        if df is None:
            continue

        classes |= set(df.iloc[:, 1])
    c2i = {c: i for i, c in enumerate(classes)}

    df_train, df_test, df_valid = splits
    if df_train is not None:
        df_train = pd.DataFrame(zip(
            [e2i[e, "iri"] for e in df_train.iloc[:, 0]],
            [c2i[c] for c in df_train.iloc[:, 1]]),
                                columns=["node_index", "class_index"])

    if df_test is not None:
        df_test = pd.DataFrame(zip(
            [e2i[e, "iri"] for e in df_test.iloc[:, 0]],
            [c2i[c] for c in df_test.iloc[:, 1]]),
                               columns=["node_index", "class_index"])

    if df_valid is not None:
        df_valid = pd.DataFrame(zip(
            [e2i[e, "iri"] for e in df_valid.iloc[:, 0]],
            [c2i[c] for c in df_valid.iloc[:, 1]]),
                                columns=["node_index", "class_index"])

    return (df_train, df_test, df_valid)


def generate_node_classification_mapping(flags):
    hdtfile = flags.context
    g = [HDTStore(hdtfile)]

    train, test, valid = None, None, None
    if flags.train is not None:
        train = pd.read_csv(flags.train)
    if flags.test is not None:
        test = pd.read_csv(flags.test)
    if flags.valid is not None:
        valid = pd.read_csv(flags.valid)

    data, e2i, r2i = _generate_context(g)
    df_splits = _generate_splits((train, test, valid), e2i, r2i)

    return (data, df_splits)


def generate_link_prediction_mapping(flags):
    data = None
    if flags.context is None:
        g_list = list()
        for hdtfile in [flags.train, flags.test, flags.valid]:
            if hdtfile is not None:
                g_list.append(HDTStore(hdtfile))

        data, e2i, r2i = _generate_context(g_list)
        df_nodes, _, df_relations, df_triples = data

        df_nodes = df_nodes.set_index('label')
        df_relations = df_relations.set_index('label')
    else:
        # allign indices with established data
        path = flags.context if flags.context.endswith('/')\
            else flags.context + '/'
        df_triples = pd.read_csv(path + "triples.int.csv")
        df_nodes = pd.read_csv(path + "nodes.int.csv", index_col='label')
        df_relations = pd.read_csv(path + "relations.int.csv",
                                   index_col='label')

    # map triples to split index
    fact_idc = list()
    split_idc = list()
    facts2i = {tuple(df_triples.iloc[i]): i for i in range(len(df_triples))}
    for i, hdtfile in enumerate([flags.train, flags.test, flags.valid]):
        if hdtfile is None:
            continue

        g = HDTStore(hdtfile)
        for (s, p, o), _ in g.triples((None, None, None), None):
            s_int = df_nodes.loc[_node_str(s)]['index']
            p_int = df_relations.loc[_node_str(p)]['index']
            o_int = df_nodes.loc[_node_str(o)]['index']

            index = facts2i[(s_int, p_int, o_int)]
            fact_idc.append(index)
            split_idc.append(i)

    df_splits = pd.concat([pd.Series(fact_idc), pd.Series(split_idc)], axis=1)
    df_splits.columns = ["triple_index", "split_index"]

    return (data, df_splits)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--context", help="HDT graph (node "
                        + "classification) or previously generated CSV "
                        + "directory to align to (link prediction)",
                        default=None)
    parser.add_argument("-d", "--dir", help="Output directory", default="./")
    parser.add_argument("-ts", "--train", help="Training set (CSV) with "
                        + "samples on the left-hand side and their classes "
                        + "on the right (node classification), or HDT graph "
                        + "(link prediction)", default=None)
    parser.add_argument("-ws", "--test", help="Withheld set (CSV) for testing "
                        + "with samples on the left-hand side and their "
                        + "classes on the right (node classification), or HDT "
                        + "graph (link prediction)", default=None)
    parser.add_argument("-vs", "--valid", help="Validation set (CSV) with "
                        + "samples on the left-hand side and their classes "
                        + "on the right (node classification), or HDT graph "
                        + "(link prediction)", default=None)
    flags = parser.parse_args()

    path = flags.dir if flags.dir.endswith('/') else flags.dir + '/'
    if flags.context.lower().endswith('.hdt'):
        data, splits = generate_node_classification_mapping(flags)

        df_train, df_test, df_valid = splits
        if df_train is not None:
            df_train.to_csv(path+'training.int.csv',
                            index=False, header=True)
        if df_test is not None:
            df_test.to_csv(path+'testing.int.csv',
                           index=False, header=True)
        if df_valid is not None:
            df_valid.to_csv(path+'validation.int.csv',
                            index=False, header=True)
    else:  # assume link prediction
        data, df_splits = generate_link_prediction_mapping(flags)
        df_splits.to_csv(path+'linkprediction_splits.int.csv',
                         index=False, header=True)

    if data is not None:
        df_nodes, df_nodetypes, df_relations, df_triples = data

        df_nodes.to_csv(path+'nodes.int.csv', index=False, header=True,
                        quoting=csv.QUOTE_NONNUMERIC)
        df_relations.to_csv(path+'relations.int.csv', index=False, header=True,
                            quoting=csv.QUOTE_NONNUMERIC)
        df_nodetypes.to_csv(path+'nodetypes.int.csv', index=False, header=True)
        df_triples.to_csv(path+'triples.int.csv', index=False, header=True)
