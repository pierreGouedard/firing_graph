"""Builds the firing graph package from the fring graph folder.

To do so run the command below in the root folder:
pip install -e .
"""
from setuptools import setup, find_packages

setup(
    name="firing_graph",
    version="1.0",
    packages=find_packages(),
    author="Pierre Gouedard",
    author_email="pierre.gouedard@alumni.epfl.ch",
    description="Package implementing firing-graph with basic functionnality",
)
