#!/usr/bin/env python

from setuptools import setup


def readme():
    with open('README.md') as f:
        return f.read()


setup(
    name='mmlkg',
    version='0.2',
    author='Xander Wilcke',
    author_email='w.x.wilcke@vu.nl',
    url='https://gitlab.com/wxwilcke/mmlkg',
    description='End-to-End Multimodal Machine Learning for RDF Knowledge Graphs',
    license='GLP3',
    include_package_data=True,
    zip_safe=True,
    install_requires=[
        'torch',
        'numpy',
        'pillow',
        'pandas',
        'rdflib',
        'rdflib_hdt',
        'deep_geometry'
    ],
    packages=['mmlkg'],
)
