# End-to-End MultiModal Machine Learning on Knowledge Graphs (MMLKG)

This packages provides multimodal node classification and link prediction for RDF knowledge graphs, by feeding literal nodes to modality-specific neural encoders, of which the resulting embeddings are used as input for a neural network (node classification) or translation model (link prediction). By default, the network is a simple two-layer MLP, whereas the translation model consists of DistMult plus LiteralE.

The purpose of this package is to provide baselines for the [MR-GCN](https://gitlab.com/wxwilcke/mrgcn).

## Getting Started

1) To install, clone the repository and run:

```
pip install . 
```

2) Once installed, we must first prepare a dataset by calling `generateInput`, which expects graphs in HDT format. Use [rdf2dt](https://github.com/rdfhdt/hdt-cpp) if your graphs are in another serialization format.

For node classification, we need the context as HDT, and the splits as CSVs with the entity IRIs and corresponding classes in the first and second column, respectively:

``` 
python generateInput.py -d ./myoutput/ -c context.hdt -ts train.csv -ws test.csv -vs valid.csv 
```

For link prediction, we need the three splits as HDT files:

``` 
python generateInput.py -d ./myoutput/ -ts train_lp.hdt -ws test_lp.hdt -vs valid_lp.hdt 
```

Running the above will generate our dataset as easy-to-use CSV files, as proposed by [KGbench](https://github.com/pbloem/kgbench/). See the example dataset in `data/test/`.

3) Run a task on the prepared dataset by running:

```
python node_classification.py -i ./myoutput --num_epoch 50 --lr 0.001
```

or

```
python link_prediction.py -i ./myoutput --num_epoch 50 --lr 0.001
```

If reading the graph from CSV files is slow, consider saving the internal data structure as compressed pickle file using `--save_dataset`, and replace the CSV folder `./myoutput/` with the pickled file in the line above.

Use the help flag `-h` to see all options.

Note that you can set encoder-specific options in `config.json`. To do so, provide the file using the `-c` flag.


## Supported data types

The following data types are supported and automatically encoded if they come with a well-defined data type declaration:

Booleans:

```
- xsd:boolean
```

Numbers:

```
- xsd:decimal
- xsd:double
- xsd:float
- xsd:integer
- xsd:long
- xsd:int
- xsd:short
- xsd:byte

- xsd:nonNegativeInteger
- xsd:nonPositiveInteger
- xsd:negativeInteger
- xsd:positiveInteger

- xsd:unsignedLong
- xsd:unsignedInt
- xsd:unsignedShort
- xsd:unsignedByte
```

Strings:

```
- xsd:string
- xsd:normalizedString
- xsd:token
- xsd:language
- xsd:Name
- xsd:NCName
- xsd:ENTITY
- xsd:ID
- xsd:IDREF
- xsd:NMTOKEN
- xsd:anyURI
```

Time/date:

```
- xsd:date
- xsd:dateTime
- xsd:gYear
```

Spatial:

```
- ogc:wktLiteral
```

Images:

```
- kgbench:base64Image (http://kgbench.info/dt)
```

Note that images are expected to be formatted as binary-encoded strings and included in the graph. 
