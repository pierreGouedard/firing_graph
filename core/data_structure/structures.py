# Global imports
import numpy as np

# Local import
from .graph import FiringGraph
from .utils import create_empty_matrices, add_core_vertices


class StructureIntersection(FiringGraph):
    """
    This class implement the main data structure used for fiting data. It is composed of weighted link in the form of
    scipy.sparse matrices and store complement information on vertices such as levels, mask for draining. It also keep
    track of the firing of vertices.

    """
    n_core = 1

    def __init__(self, n_intersection, n_inputs, n_outputs, d_inputs, weight, index_output, ax_levels,
                 enable_draining=True, **kwargs):

        self.n_inputs, self.n_outputs, self.n_intersection = n_inputs, n_outputs, n_intersection
        self.index_output = index_output

        if self.n_intersection > 1:
            depth = 2
        else:
            depth = 3

        # Initialize Matrices
        d_matrices = create_empty_matrices(self.n_inputs, self.n_intersection + (depth - 2), self.n_outputs)

        # Set Input matrix
        for i, l_inputs in d_inputs.keys():
            d_matrices['Iw'][l_inputs, i] = weight

        # Set Core matrix
        for i in range(self.n_intersection - (3 - depth)):
            d_matrices['Cw'][i, self.n_intersection] = 1

        # Set Output matrix
        d_matrices['Ow'][self.n_intersection - (3 - depth), index_output] = 1

        # Update mask if necessary
        if enable_draining:
            d_matrices.update({'Im': d_matrices['Iw'] > 0})

        # Invoke parent constructor
        super(StructureIntersection, self).__init__(
            'StructureIntersection', ax_levels, d_matrices, depth=depth, **kwargs
        )

    @staticmethod
    def from_partition(partition, weight, firing_graph, index_output=None, enable_draining=True):

        # get core vertices of the partitoin
        l_indices = partition.pop('indices')

        # Set levels
        ax_levels = firing_graph.levels[l_indices]

        # Set inputs
        d_inputs = {i: firing_graph.I[:, j].nonzero()[0] for i, j in enumerate(l_indices)}
        d_inputs = {k: v for k, v in d_inputs.items() if len(v) > 0}

        # Set index_output if necessary
        if index_output is None:
            index_output = firing_graph.O[l_indices, :].nonzero()[1][0]

        return StructureIntersection(
            firing_graph.I.shape[0], firing_graph.O.shape[1], len(d_inputs), d_inputs, weight, index_output,
            ax_levels, enable_draining, **partition
        )

    def augment_structure(self, n_intersection, d_inputs, ax_levels, weight, enable_draining=True, partitions=None):

        # Compute new number of core vertices
        n_core = self.n_intersection + n_intersection + 1
        d_matrices = add_core_vertices(self.matrices, n_intersection + 2, offset=self.n_intersection)

        # Set levels
        self.levels = np.hstack((self.levels[:self.n_intersection], ax_levels, np.ones(1)))

        # Set links for new
        for j, l_indices in d_inputs:
            d_matrices['Iw'][l_indices, self.n_intersection + j] = weight
            d_matrices['Im'][l_indices, self.n_intersection + j] = enable_draining

        d_matrices['Cw'][:n_core - 1, n_core - 1] = 1

        # Set outputs
        d_matrices[n_core - 1, self.index_output] = 1

        self.matrices.update(d_matrices)
        self.partitions = partitions

        return self

    def augment_intersection(self, index_inputs, index_intersection=0):
        raise NotImplementedError


class StructureDoubleIntersection(FiringGraph):
    """
    To do if needed: allow anti symetric non activation consideration

    \/  \/
    O   O       levels = [l, l'] 
    | \/
    O O         levels = [1, 1]
    \/
    O           levels = [2]

    When left vertex of layer 1 activate alone then vertex of the last layer activate, otherwise it doesn't activate.
    """
    raise NotImplementedError





