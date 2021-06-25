#!/usr/bin.env python

from os.path import join
import pickle

import numpy as np
import pandas as pd

from mmlp.data.xsd_hierarchy import XSDHierarchy
from mmlp.vectorizers import numerical, temporal, textual, spatial, visual


_SPECIAL = {'iri': '0', 'blank_node': '1', 'none': '2'}
_XSD_NS = "http://www.w3.org/2001/XMLSchema#"


class Data:
    """
    Class representing a dataset.

    """

    def __init__(self, path):
        """ The edges of the knowledge graph (the triples), represented by
            their integer indices. A (m, 3) numpy array."""
        self.triples = None

        """ A mapping from an integer index to a relation representation,
        and its reverse."""
        self.i2r, self.r2i = None, None  # list, dict

        """ A mapping from an integer index to a node representation, and
        its reverse. A node is either a simple string indicating the label
        of the node (an IRI, blank node or literal), or it is a pair
        indicating the datatype and the label (in that order)."""
        self.i2n, self.n2i = None, None  # list, dict

        """ Total number of distinct nodes in the graph."""
        self.num_nodes = None

        """ Total number of distinct relation types in the graph."""
        self.num_relations = None

        """ Total number of classes in the classification task."""
        self.num_classes = None

        """ Split data: a matrix with entity indices in column 0 and class
        indices in column 1."""
        self.train = None
        self.test = None
        self.valid = None

        self._dt_l2g = {}
        self._dt_g2l = {}

        self._datatypes = None
        if path is not None:
            self.triples = np.loadtxt(join(path, 'triples.int.csv'),
                                      dtype=np.int,
                                      delimiter=',', skiprows=1)

            self.i2r, self.r2i = load_indices(join(path, 'relations.int.csv'))
            self.i2n, self.n2i = load_entities(join(path, 'nodes.int.csv'))

            self.num_nodes = len(self.i2n)
            self.num_relations = len(self.i2r)

            self.train = np.loadtxt(join(path, 'training.int.csv'),
                                    dtype=np.int,
                                    delimiter=',', skiprows=1)
            self.test = np.loadtxt(join(path, 'testing.int.csv'),
                                   dtype=np.int,
                                   delimiter=',', skiprows=1)
            self.valid = np.loadtxt(join(path, 'validation.int.csv'),
                                    dtype=np.int,
                                    delimiter=',', skiprows=1)

            self.num_classes = len(set(np.concatenate([self.train[:, 1],
                                                       self.test[:, 1],
                                                       self.valid[:, 1]])))

    def datatype_g2l(self, dtype, copy=True):
        """
        Returns a list mapping a global index of an entity (the indexing over
        all nodes) to its _local index_ the indexing over all nodes of the
        given datatype

        :param dtype:
        :param copy:
        :return: A dict d so that `d[global_index] = local_index`
        """
        if dtype not in self._dt_l2g:
            self._dt_l2g[dtype] = [i for i, (label, dt) in enumerate(self.i2n)
                                   if dt == dtype
                                   or (dtype == _XSD_NS+"string"
                                       and dt.startswith('@'))]
            self._dt_g2l[dtype] = {v: k
                                   for k, v in enumerate(self._dt_l2g[dtype])}

        return dict(self._dt_g2l[dtype]) if copy else self._dt_g2l[dtype]

    def datatype_l2g(self, dtype, copy=True):
        """
        Maps local to global indices.

        :param dtype:
        :param copy:
        :return: A list l so that `l[local index] = global_index`
        """
        self.datatype_g2l(dtype, copy=False)  # init dicts

        return list(self._dt_l2g[dtype]) if copy else self._dt_l2g[dtype]

    def get_strings(self, dtype):
        """
        Retrieves a list of all string representations of a given datatype in
        order

        :return:
        """
        return [self.i2n[g][0] for g in self.datatype_l2g(dtype)]

    def datatypes(self, i=None):
        """
        :return: If i is None:a list containing all datatypes present in this
        dataset (including literals without datatype, URIs and blank nodes), in
        canonical order (dee `datatype_key()`).  If `i` is a nonnegative
        integer, the i-th element in this list.  """
        if self._datatypes is None:
            self._datatypes = {dt for _, dt in self.i2n}
            self._datatypes = list(self._datatypes)
            self._datatypes.sort(key=datatype_key)

        if i is None:
            return self._datatypes

        return self._datatypes[i]


def datatype_key(string):
    """ A key that defines the canonical ordering for datatypes. The datatypes
    'iri', 'blank_node' and 'none' are sorted to the front in that order, with
    any further datatypes following in lexicographic order.

    :param string: :return:
    """

    if string in _SPECIAL:
        return _SPECIAL[string] + string

    return '9' + string


def load_indices(file):
    df = pd.read_csv(file, na_values=[], keep_default_na=False)

    assert len(df.columns) == 2, "CSV file should have two columns"
    assert not df.isnull().any().any(), "CSV file %s has missing values" % file

    idxs = df['index'].tolist()
    labels = df['label'].tolist()

    i2l = list(zip(idxs, labels))
    i2l.sort(key=lambda p: p[0])
    for i, (j, _) in enumerate(i2l):
        assert i == j, f'Indices in {file} are not contiguous'

    i2l = [l for i, l in i2l]
    l2i = {l: i for i, l in enumerate(i2l)}

    return i2l, l2i


def load_entities(file):
    df = pd.read_csv(file, na_values=[], keep_default_na=False)

    assert len(df.columns) == 3, 'Entity file should have three columns'
    assert not df.isnull().any().any(), f'CSV file {file} has missing values'

    idxs = df['index'].tolist()
    dtypes = df['annotation'].tolist()
    labels = df['label'].tolist()

    ents = zip(labels, dtypes)

    i2n = list(zip(idxs, ents))
    i2n.sort(key=lambda p: p[0])
    for i, (j, _) in enumerate(i2n):
        assert i == j, 'Indices in entities.int.csv are not contiguous'

    i2n = [l for i, l in i2n]

    n2i = {e: i for e in enumerate(i2n)}

    return i2n, n2i


def generate_pickled(flags):
    dataset = dict()
    xsd_tree = XSDHierarchy()

    # read data dir
    g = Data(flags.input)

    dataset['num_nodes'] = g.num_nodes
    dataset['num_classes'] = g.num_classes
    dataset['training'] = g.train
    dataset['testing'] = g.test
    dataset['validation'] = g.valid
    for modality in flags.modalities:
        print("[%s] Generating data" % modality.upper())
        datatypes = set()
        if modality == "textual":
            datatypes = set.union(xsd_tree.descendants("string"),
                                  xsd_tree.descendants("anyURI"))
            datatypes = expand_prefix(datatypes)
            data = textual.generate_data(g, datatypes)

        if modality == "numerical":
            datatypes = set.union(xsd_tree.descendants("numeric"),
                                  xsd_tree.descendants("boolean"))
            datatypes = expand_prefix(datatypes)
            data = numerical.generate_data(g, datatypes)

        if modality == "temporal":
            datatypes = set.union(xsd_tree.descendants("date"),
                                  xsd_tree.descendants("dateTime"),
                                  xsd_tree.descendants("gYear"))
            datatypes = expand_prefix(datatypes)
            data = temporal.generate_data(g, datatypes)

        if modality == "visual":
            datatypes = {"http://kgbench.info/dt#base64Image"}
            data = visual.generate_data(g, datatypes)

        if modality == "spatial":
            datatypes = {"http://www.opengis.net/ont/geosparql#wktLiteral"}
            data = spatial.generate_data(g, datatypes)

        if len(data) <= 0:
            print(" No information found")
            continue

        dataset[modality] = data

    if flags.save_dataset:
        print('Saving data to disk...')
        with open(flags.input + 'dataset.pkl', 'wb') as f:
            pickle.dump(dataset, f)

    return dataset


def expand_prefix(datatypes):
    result = set()
    for datatype in datatypes:
        result.add(datatype.replace("xsd.", _XSD_NS))

    return result