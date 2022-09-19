"""Builds the firing_graph package from the firing_graph folder folder.

To do so run the command below in the root folder:
pip install -e .
"""
from setuptools import setup, find_packages

setup(
    name="firing_graph",
    version="2.0",
    packages=find_packages(exclude=('tests',)),
    author="Pierre Gouedard",
    author_email="pierre.gouedard@alumni.epfl.ch",
    description="Package implementing firing-graph with basic functionality",
)
