#! /usr/bin/env python

import h5py
import numpy as np


_MODALITIES = ["textual", "numerical", "temporal", "visual", "spatial"]


class HDF5:
    _hf = None
    _compression = None

    NODE_CLASSIFICATION = "node_classification"
    LINK_PREDICTION = "link_prediction"

    def __init__(self, path, mode='r', compression="gzip"):
        self._hf = h5py.File(path, mode)
        self._compression = compression

    def read_dataset(self, task=None, modalities=None):
        dataset = dict()

        dataset['num_nodes'] = self._read_metadata(self._hf, 'num_nodes')
        if task == self.NODE_CLASSIFICATION or task is None:
            nc_group = self._hf[self.NODE_CLASSIFICATION]
            dataset |= self._read_data_node_classification(nc_group)
        if task == self.LINK_PREDICTION or task is None:
            lp_group = self._hf[self.LINK_PREDICTION]
            dataset |= self._read_data_link_prediction(lp_group)

        modalities = modalities if modalities is not None else _MODALITIES
        for modality in modalities:
            if modality not in self._hf.keys():
                continue

            data = list()
            group = self._hf[modality]
            for datatype in group.keys():
                data.append(self._read_data_from_group(group, datatype))

            dataset[modality] = data

        return dataset

    def write_dataset(self, dataset):
        self.write_metadata('num_nodes', dataset['num_nodes'])
        self.write_task_data(dataset)
        for modality in _MODALITIES:
            if modality in dataset.keys():
                self.write_modality_data(dataset[modality], modality)

    def write_metadata(self, key, value):
        self._write_metadata(self._hf, key, value)

    def write_modality_data(self, data, modality):
        group = self._hf[modality] if modality in self._hf.keys()\
                else self._hf.create_group(modality)
        for datatype_set in data:
            self._write_data_to_group(group, datatype_set,
                                      self._compression)

    def write_task_data(self, dataset):
        if 'num_classes' in dataset.keys():
            nc_group = self._hf.create_group(self.NODE_CLASSIFICATION)
            self._write_data_node_classification(nc_group, dataset,
                                                 compression=self._compression)
        if 'triples' in dataset.keys():
            lp_group = self._hf.create_group(self.LINK_PREDICTION)
            self._write_data_link_prediction(lp_group, dataset,
                                             compression=self._compression)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._hf.close()

    def _write_metadata(self, group, name, data):
        group.attrs[name] = data

    def _write_data_node_classification(self, group, dataset,
                                        compression):
        num_classes = dataset['num_classes']
        train = dataset['training']
        test = dataset['testing']
        valid = dataset['validation']

        group.create_dataset("training", data=train,
                             compression=compression)
        group.create_dataset("testing", data=test,
                             compression=compression)
        group.create_dataset("validation", data=valid,
                             compression=compression)

        group.attrs["num_classes"] = num_classes

    def _write_data_link_prediction(self, group, dataset,
                                    compression):
        entities = dataset['entities']
        triples = dataset['triples']
        train = dataset['training_lp']
        test = dataset['testing_lp']
        valid = dataset['validation_lp']

        group.create_dataset("training_lp", data=train,
                             compression=compression)
        group.create_dataset("testing_lp", data=test,
                             compression=compression)
        group.create_dataset("validation_lp", data=valid,
                             compression=compression)

        group.create_dataset("triples", data=triples,
                             compression=compression)
        group.create_dataset("entities", data=entities,
                             compression=compression)

    def _write_data_to_group(self, group, dtype_data, compression):
        datatype, data, data_length, data_entity_map,\
                is_varlength, time_dim = dtype_data
        dtype = datatype.split('/')[-1]
        subgroup = group.create_group(dtype)

        # stack and write data
        sample = data[0]
        if isinstance(sample, np.ndarray):
            if is_varlength:
                self._write_var_length_2D(subgroup, "data",
                                          data, data_length,
                                          time_dim,
                                          compression=compression)
            else:
                self._write_fixed_length_ND(subgroup, "data",
                                            data, data_length,
                                            compression=compression)
        else:  # nested list or raw values
            self._write_fixed_length_2D(subgroup, "data",
                                        data, data_length,
                                        compression=compression)

        # write lengths and mapping index
        subgroup.create_dataset("data_length",
                                data=np.array(data_length),
                                compression=compression)
        subgroup.create_dataset("data_entity_map",
                                data=np.array(data_entity_map),
                                compression=compression)

        # write metada
        subgroup.attrs['is_varlength'] = is_varlength
        subgroup.attrs['time_dim'] = time_dim

    def _write_fixed_length_2D(self, subgroup, name,
                               data, data_length, compression):
        nrows = len(data)
        ncols = max(data_length)

        f = subgroup.create_dataset(name, shape=(nrows, ncols),
                                    dtype='f4', compression=compression)
        for i in range(nrows):
            # items are lists or raw values
            row = data[i]
            f[i] = np.array(row)

    def _write_fixed_length_ND(self, subgroup, name,
                               data, data_length, compression):
        shapes = set(item.shape for item in data)
        assert len(shapes) == 1, "Shapes of variable size not supported"
        shape = shapes.pop()

        num_items = len(data)
        f = subgroup.create_dataset(name, shape=(num_items, *shape),
                                    dtype='f4', compression=compression)
        for i in range(num_items):
            # items are ndarrays
            f[i] = data[i]

    def _write_var_length_2D(self, subgroup, name,
                             data, data_length, time_dim, compression):
        # stack on time_dim
        alphabet_size = set(item.shape[1-time_dim] for item in data)
        assert len(alphabet_size) == 1, "Alphabets of variable size not "\
                                        "supported"

        alphabet_size = alphabet_size.pop()
        full_length = sum(data_length)

        offset = 0
        num_items = len(data)
        if time_dim == 0:
            shape = (full_length, alphabet_size)
            f = subgroup.create_dataset(name, shape=shape, dtype='f4',
                                        compression=compression)
            for i in range(num_items):
                # items are ndarrays
                item = data[i]
                item_length = item.shape[time_dim]
                f[offset:offset+item_length, :] = item

                offset += item_length
        elif time_dim == 1:
            shape = (alphabet_size, full_length)
            f = subgroup.create_dataset(name, shape=shape, dtype='f',
                                        compression=compression)
            for i in range(num_items):
                item = data[i]
                item_length = item.shape[time_dim]
                f[:, offset:offset+item_length] = item

                offset += item_length
        else:
            raise Exception("Unsupported time dimension %d" % time_dim)

    def _read_metadata(self, group, name):
        return group.attrs[name]

    def _read_data_node_classification(self, group):
        out = dict()

        out['num_classes'] = group.attrs["num_classes"]
        for name in ["training", "testing", "validation"]:
            out[name] = np.array(group.get(name))

        return out

    def _read_data_link_prediction(self, group):
        out = dict()

        for name in ["entities", "triples",
                     "training_lp", "testing_lp", "validation_lp"]:
            out[name] = np.array(group.get(name))

        return out

    def _read_data_from_group(self, group, datatype):
        subgroup = group[datatype]

        # read metadata
        is_varlength = subgroup.attrs['is_varlength']
        time_dim = subgroup.attrs['time_dim']

        # read length and index mapping
        data_length = np.array(subgroup.get("data_length"))
        data_entity_map = np.array(subgroup.get("data_entity_map"))

        # read data
        data = None
        if is_varlength:
            data = self._read_var_length(subgroup, "data",
                                         data_length, time_dim)
        else:
            data = self._read_fixed_length(subgroup, "data")

        return (datatype, data, data_length, data_entity_map,
                is_varlength, time_dim)

    def _read_fixed_length(self, subgroup, name):
        out = list()
        data = np.array(subgroup.get(name))
        for i in range(data.shape[0]):
            item = data[i]

            if item.ndim == 1 and len(item) == 1:
                item = [item[0]]

            out.append(item)

        return out

    def _read_var_length(self, subgroup, name, data_length, time_dim):
        out = list()
        offset = 0
        data = np.array(subgroup.get(name))
        if time_dim == 0:
            for item_length in data_length:
                item = data[offset:offset+item_length, :]

                offset += item_length
                out.append(item)
        elif time_dim == 1:
            for item_length in data_length:
                item = data[:, offset:offset+item_length]

                offset += item_length
                out.append(item)
        else:
            raise Exception("Unsupported time dimension %d" % time_dim)

        return out
