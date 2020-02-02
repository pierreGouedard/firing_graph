# Global imports
import numpy as np
from scipy.sparse import lil_matrix

# Local import
from .graph import FiringGraph
from .utils import create_empty_matrices, add_core_vertices, reduce_matrices, reduce_backward_firing, augment_matrices


class StructureEmptyIntersection(FiringGraph):

    def __init__(self, n_inputs, n_outputs, index_output, **kwargs):

        self.n_inputs, self.n_outputs, self.n_intersection = n_inputs, n_outputs, 0
        self.index_output, n_core = index_output, 0

        kwargs.update({
            'project': 'StructureEmptyIntersection', 'ax_levels': np.array([]), 'depth': 2,
            'matrices': create_empty_matrices(n_inputs, n_outputs, n_core)
        })

        # Invoke parent constructor
        super(StructureEmptyIntersection, self).__init__(**kwargs)


class StructureIntersection(FiringGraph):
    """
    This class implement the main data structure used for fiting data. It is composed of weighted link in the form of
    scipy.sparse matrices and store complement information on vertices such as levels, mask for draining. It also keep
    track of the firing of vertices.

    """

    def __init__(self, n_intersection, n_inputs, n_outputs, ax_levels, index_output, matrices, **kwargs):

        self.n_inputs, self.n_outputs, self.n_intersection = n_inputs, n_outputs, n_intersection
        self.index_output = index_output

        kwargs.update({
            'project': 'StructureIntersection', 'ax_levels': ax_levels, 'depth': 2 + int(self.n_intersection > 1),
            'matrices': matrices
        })

        # Invoke parent constructor
        super(StructureIntersection, self).__init__(**kwargs)

    @staticmethod
    def from_dict(d_struct, graph_kwargs):

        return StructureIntersection(
            n_intersection=d_struct['n_intersection'], n_inputs=d_struct['n_inputs'],
            n_outputs=d_struct['n_outputs'], ax_levels=d_struct['ax_levels'], index_output=d_struct['index_output'],
            matrices=d_struct['matrices'], **graph_kwargs)

    @staticmethod
    def from_partition(partition, firing_graph, index_output=None, add_backward_firing=False):

        # In case of empty partition, stop
        if len(partition['indices']) == 0:
            return None

        # Set number of intersection (remove top vertex in case of multiple intersection)
        n_inter = len(partition['indices']) - int(len(partition['indices']) > 1)

        # Set levels
        ax_levels = firing_graph.levels[partition['indices']]

        # Set inputs
        d_matrices = reduce_matrices(firing_graph.matrices, partition['indices'])

        # Set index_output if necessary
        index_output = partition.get('index_output', index_output)

        if d_matrices['Ow'][-1, index_output] == 0:
            d_matrices['Ow'] = d_matrices['Ow'].tolil()
            d_matrices['Ow'][-1, index_output] = 1
            d_matrices['Ow'] = d_matrices['Ow'].tocsc()

        # Add kwargs
        kwargs = {'partitions': partition.get('partitions', None), 'precision': partition.get('precision', None)}
        if add_backward_firing:
            kwargs.update(
                {'backward_firing': reduce_backward_firing(firing_graph.backward_firing, partition['indices'])}
            )

        return StructureIntersection(
            n_inter, firing_graph.I.shape[0], firing_graph.O.shape[1], ax_levels, index_output, matrices=d_matrices,
            **kwargs
        )

    @staticmethod
    def from_input_indices(n_inputs, n_outputs, ax_levels, index_output, l_inputs, weight, enable_drain=True, **kwargs):

        # Set number of intersection
        n_intersection = len(l_inputs)

        # Initialize Matrices
        d_matrices = create_empty_matrices(n_inputs, n_outputs, n_intersection + int(n_intersection > 1))

        # Set Input matrix
        for i, l_bits in enumerate(l_inputs):
            d_matrices['Iw'][list(l_bits), i] = weight

        # Set Core matrix
        for i in range(n_intersection - int(n_intersection == 1)):
            d_matrices['Cw'][i, n_intersection] = 1

        # Set Output matrix
        d_matrices['Ow'][n_intersection - int(n_intersection == 1), index_output] = 1

        # Update mask if necessary
        if enable_drain:
            d_matrices.update({'Im': d_matrices['Iw'] > 0})

        return StructureIntersection(
            n_intersection, n_inputs, n_outputs, ax_levels, index_output, matrices=d_matrices, **kwargs
        )

    def augment_structure(self, n_intersection, l_inputs, ax_levels, weight, enable_draining=True, partitions=None):

        # Compute new number of core vertices
        n_core = self.n_intersection + n_intersection + 1
        d_matrices = add_core_vertices(self.matrices, n_intersection + 1, offset=self.n_intersection)

        # Set levels
        self.levels = np.hstack((self.levels[:self.n_intersection], ax_levels, np.ones(1)))

        # Set links for new intersections
        for i, l_bits in enumerate(l_inputs):
            d_matrices['Iw'][list(l_bits), self.n_intersection + i] = weight
            d_matrices['Im'][list(l_bits), self.n_intersection + i] = enable_draining

        d_matrices['Cw'][:n_core - 1, n_core - 1] = 1
        d_matrices['Iw'], d_matrices['Cw'] = d_matrices['Iw'].tocsc(), d_matrices['Cw'].tocsc()

        # Set outputs
        d_matrices['Ow'] = lil_matrix((n_core, self.n_outputs))
        d_matrices['Ow'][n_core - 1, self.index_output] = 1
        d_matrices['Ow'] = d_matrices['Ow'].tocsc()

        self.matrices.update(d_matrices)
        self.partitions = partitions
        self.depth = 3

        return self

    def augment_intersection(self, l_indices, weight, delta_level, index_intersection=0):

        # Add inputs to intersection of interest
        self.matrices['Iw'] = self.matrices['Iw'].tolil()
        self.matrices['Iw'][l_indices, index_intersection] = weight
        self.matrices['Iw'] = self.matrices['Iw'].tocsc()

        # Increase level accordingly
        self.levels[index_intersection] += delta_level

        return self

    def copy(self):

        d_graph = super(StructureIntersection, self).to_dict(deep_copy=True)
        d_struct = {
            'n_inputs': self.n_inputs,
            'n_outputs': self.n_outputs,
            'n_intersection': self.n_intersection,
            'index_output': self.index_output,
            'matrices': d_graph.pop('matrices'),
            'ax_levels': d_graph.pop('ax_levels')
        }

        return self.from_dict(d_struct, {k: v for k, v in d_graph.items()})


class StructureDoubleIntersection(FiringGraph):
    """
    To do if needed: allow anti symetric non activation.

    \/  \/
    O   O       levels = [l, l'] 
    | \/
    O O         levels = [1, 1]
    \/
    O           levels = [2]

    When left vertex of layer 1 activate alone then vertex of the last layer activate, otherwise it doesn't activate.
    """


class StructureYala(FiringGraph):
    """
    This class implement the main data structure used for fiting data. It is composed of weighted link in the form of
    scipy.sparse matrices and store complement information on vertices such as levels, mask for draining. It also keep
    track of the firing of vertices.

    """
    n_core = 1

    def __init__(self, n_inputs, n_outputs, ax_levels, index_output, matrices, depth, **kwargs):

        self.n_inputs, self.n_outputs, self.index_output = index_output, n_inputs, n_outputs
        kwargs.update({'project': 'StructureYala', 'ax_levels': ax_levels, 'matrices': matrices})

        # Invoke parent constructor
        super(StructureYala, self).__init__(**kwargs)

    @staticmethod
    def from_structure(base_structure, transient_structure):

        depth, n_core = 3, base_structure.n_intersection + transient_structure.n_intersection + 1
        matrices = augment_matrices(base_structure.matrices, transient_structure.matrices)
        ax_levels = np.hstack((base_structure.levels, transient_structure.levels))

        # If base structure is not empty add merging layer's vertices
        if base_structure.n_intersection > 0:
            depth, n_core = depth + 1, n_core + 2
            matrices = add_core_vertices(matrices, 2, base_structure.n_intersection + transient_structure.n_intersection)
            ax_levels = np.hstack((ax_levels, np.array([1, 2])))

            # Link core vertices
            matrices['Cw'][base_structure.n_intersection, -2], matrices['Cw'][-2, -1] = 1, 1
            matrices['Cw'][base_structure.n_intersection + transient_structure.n_intersection, -1] = 1

        # Transform matrices to csc sparse format
        matrices['Iw'], matrices['Cw'] = matrices['Iw'].tocsc(), matrices['Cw'].tocsc()

        # Set output connection
        matrices['Ow'] = lil_matrix((n_core, base_structure.n_outputs))
        matrices['Ow'][-1, base_structure.index_output] = 1
        matrices['Ow'] = matrices['Ow'].tocsc()

        # Create partitions
        partitions = [
            {'indices': range(base_structure.n_intersection), 'name': "base", 'precision': base_structure.precision,
             "depth": 2},
            {'indices': [base_structure.n_intersection + j for j in range(transient_structure.n_intersection)],
             "name": "transient", "depth": 3},
        ]

        # Add kwargs
        kwargs = {'partitions': partitions, 'precision': base_structure.precision, 'depth': depth}

        return StructureYala(
            base_structure.n_inputs, base_structure.n_outputs, ax_levels, base_structure.index_output, matrices,
            **kwargs
        )




