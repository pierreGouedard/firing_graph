# Global imports
import unittest
import numpy as np
from scipy.sparse import csc_matrix

# Local import
from firing_graph.servers import ArrayServer
from firing_graph.graph import FiringGraph, create_empty_matrices
from firing_graph.linalg.forward import fpo


# TODO: obsolete => to refactor
class TestServer(unittest.TestCase):
    pass