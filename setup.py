#!/usr/bin/env python

from setuptools import setup


def readme():
    with open('README.md') as f:
        return f.read()

setup(
    name='mmlp',
    version='0.1',
    author='Xander Wilcke',
    author_email='w.x.wilcke@vu.nl',
    url='https://gitlab.com/wxwilcke/mmlp',
    description='Multimodal Multi-Layer Perceptron (M-MLP) for RDF Knowledge Graphs',
    license='GLP3',
    include_package_data=True,
    zip_safe=True,
    install_requires=[
        'torch',
        'numpy',
        'rdflib',
        'rdflib-hdt',
        'pillow',
        'pandas',
        'deep_geometry'
    ],
    packages=['mmlp'],
)
