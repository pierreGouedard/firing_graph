# Global imports
import numpy as np

# local import
from core.data_structure.utils import mat_from_tuples
from core.data_structure.graph import FiringGraph


class TestPattern(object):
    l_seq_types = ['det', 'rand']

    def __init__(self, name, input_vertices, core_vertices, output_vertices, depth):

        self.name = name
        self.input_vertices = input_vertices
        self.core_vertices = core_vertices
        self.output_vertices = output_vertices
        self.depth = depth

    def build_graph_pattern_final(self):
        """
        Return a firing graph as it should be at the end of the test

        :return:
        """
        raise NotImplementedError

    def build_graph_pattern_init(self):
        """
        Return a firing graph as it should be at the beginning of the test
        :return:
        """
        raise NotImplementedError

    def build_deterministic_io(self):
        """
        Build deterministic activation of input output pair as couple of numpy.array
        :return:
        """
        raise NotImplementedError

    def build_random_io(self):
        """
        Build random activation of input with all zeros output pair as couple of numpy.array
        :return:
        """
        raise NotImplementedError

    def init_io_sequence(self,):

        raise NotImplementedError

    def generate_io_sequence(self, length, p=0.5, seed=1830):
        """
        Return a mix between deterministic io and noisy io
        :return:
        """
        np.random.seed(seed)
        ax_isequence, ax_osequence = self.init_io_sequence()

        for _ in range(length):
            seqtype = np.random.choice(self.l_seq_types, p=[p, 1-p])

            if seqtype == 'det':
                # Add deterministic pattern
                ax_inputs, ax_outputs = self.build_deterministic_io()
                ax_isequence, ax_osequence = np.vstack((ax_isequence, ax_inputs)), np.vstack((ax_osequence, ax_outputs))

            else:
                # Add random pattern
                ax_inputs, ax_outputs = self.build_random_io()
                ax_isequence, ax_osequence = np.vstack((ax_isequence, ax_inputs)), np.vstack((ax_osequence, ax_outputs))

        return ax_isequence, ax_osequence


class AndPattern2(TestPattern):
    """
    The primary test purpose of this pattern is to test for edge removal in a firing graph of depth 2. After a
    significant number of iteration, only correct edges should remain
    """

    def __init__(self, ni, no, w=100, p=0.5, random_target=False, seed=None):

        if seed is not None:
            np.random.seed(seed)

        # Core params of test
        self.ni, self.no, self.w, self.p, self.random_target = ni, no, w, p, random_target,

        # Init
        self.mask_vertice_drain, firing_graph = {'I': np.ones(self.ni * self.no), 'C': np.zeros(self.no)}, None

        # Set targets
        self.target = [
            np.random.choice(range(self.ni * j, self.ni * (j + 1)), np.random.randint(1, int(0.5 * self.ni)), replace=False)
            for j in range(self.no)
        ]

        # Build list of vertices
        input_vertices = [['input_{}'.format((self.ni * i) + j) for j in range(self.ni)] for i in range(self.no)]
        core_vertices = ['core_{}'.format(i) for i in range(self.no)]
        output_vertices = ['output_{}'.format(i) for i in range(self.no)]
        TestPattern.__init__(self, 'and', input_vertices, core_vertices, output_vertices, 2)

    def build_graph_pattern_final(self):
        # Build edges
        l_edges = []
        for i in range(self.no):
            l_edges += [(self.input_vertices[i][j], self.core_vertices[i]) for j in range(self.ni)
                        if j + self.ni * i in self.target[i]]
        l_edges += zip(self.core_vertices, self.output_vertices)

        # Build Firing graph
        sax_I, sax_C, sax_O = mat_from_tuples(self.ni * self.no, self.no, self.no, l_edges, weights=self.w)
        self.firing_graph = FiringGraph.from_matrices(
            'AndPatFinal2', sax_I, sax_C, sax_O, np.ones(self.no), self.mask_vertice_drain, self.depth
        )

        return self.firing_graph

    def build_graph_pattern_init(self):

        # Build edges
        l_edges = []
        for i in range(self.no):
            l_edges += [(self.input_vertices[i][j], self.core_vertices[i]) for j in range(self.ni)]
        l_edges += zip(self.core_vertices, self.output_vertices)

        # Build Firing graph
        sax_I, sax_C, sax_O = mat_from_tuples(self.ni * self.no, self.no, self.no, l_edges, weights=self.w)
        self.firing_graph = FiringGraph.from_matrices(
            'AndPat2', sax_I, sax_C, sax_O, np.ones(self.no), self.mask_vertice_drain, self.depth
        )

        return self.firing_graph

    def build_deterministic_io(self):
        ax_inputs = np.zeros((1, self.ni * self.no))

        for i in range(self.no):
            ax_inputs_ = np.zeros((1, self.ni * self.no))
            ax_inputs_[0, self.target[i]] = 1

            ax_inputs = np.vstack((ax_inputs, ax_inputs_))

        ax_outputs = np.eye(self.no)

        return ax_inputs[1:, :], ax_outputs

    def build_random_io(self):

        # Generate random inputs
        ax_inputs = np.random.binomial(1, self.p, (1, self.ni * self.no))

        if not self.random_target:
            for i in range(self.no):
                ax_inputs[0, self.target[i]] = 0

        # Generate outputs
        ax_outputs = np.zeros((1, self.no))

        return ax_inputs, ax_outputs

    def init_io_sequence(self):
        return np.zeros((1, self.ni * self.no)), np.zeros((1, self.no))

    def layout(self):
        pos = dict()

        pos.update({'inputs': {'pos': {i: (0, i) for i in range(self.ni * self.no)}, 'color': 'r'}})
        n = self.ni * self.no
        pos.update({'cores': {
            'pos': {(n + i): (1, ((self.ni * i) + (self.ni * (i + 1))) / 2) for i in range(self.no)},
            'color': 'k'}
        })
        n = self.ni * self.no + self.no
        pos.update({'outputs': {
            'pos': {(n + i): (2, ((self.ni * i) + (self.ni * (i + 1))) / 2) for i in range(self.no)},
            'color': 'b'}
        })

        return pos


class AndPattern3(TestPattern):
    """
    The primary test purpose of this pattern is to test for edge removal in a firing graph of depth 3. After a
    significant number of iteration, only correct edges should remain
    """
    def __init__(self, ni, no, w=100, p=0.5, n_selected=2, random_target=False, seed=None):

        try:
            assert n_selected < ni
        except AssertionError:
            raise ValueError('Number of input must be larger than the number of selected bits')

        if seed is not None:
            np.random.seed(seed)

        # Core params of test
        self.ni, self.no, self.nc, self.w, self.p, self.random_target = ni, no, 3, w, p, random_target

        # Init
        self.firing_graph, self.mask_vertice_drain = None, {'I': np.ones(self.ni * self.no),
                                                            'C': np.zeros(self.no * self.nc)}

        # Set targets
        self.target = [
            np.random.choice(
                range(self.ni * j, self.ni * (j + 1)), np.random.randint(n_selected + 1, self.ni / 2), replace=False)
            for j in range(self.no)
        ]
        self.target_selected = [self.target[j][:n_selected] for j in range(self.no)]

        # Build list of vertices
        input_vertices = [['input_{}'.format((self.ni * i) + j) for j in range(self.ni)] for i in range(self.no)]
        core_vertices = [
            ['core_{}'.format(self.nc * i), 'core_{}'.format(self.nc * i + 1), 'core_{}'.format(self.nc * i + 2)]
            for i in range(self.no)
        ]
        output_vertices = ['output_{}'.format(i) for i in range(self.no)]
        TestPattern.__init__(self, 'and', input_vertices, core_vertices, output_vertices, 3)

    def build_graph_pattern_final(self):
        # Build edges
        l_edges, ax_levels = [], np.ones(self.nc * self.no)

        for i in range(self.no):
            l_edges += [
                (self.input_vertices[i][j], self.core_vertices[i][0]) for j in range(self.ni)
                if self.ni * i + j in self.target_selected[i]
            ]
            ax_levels[i * self.nc] = len(self.target_selected[i])

        for i in range(self.no):
            l_edges += [
                (self.input_vertices[i][j], self.core_vertices[i][1]) for j in range(self.ni)
                if self.ni * i + j not in self.target_selected[i] and self.ni * i + j in self.target[i]
            ]

        for i in range(self.no):
            l_edges += [
                (self.core_vertices[i][0], self.core_vertices[i][self.nc - 1]),
                (self.core_vertices[i][1], self.core_vertices[i][self.nc - 1]),
                (self.core_vertices[i][self.nc - 1], self.output_vertices[i])
            ]
            ax_levels[i * self.nc + self.nc - 1] = self.nc - 1

        # Build Firing graph
        sax_I, sax_C, sax_O = mat_from_tuples(self.ni * self.no, self.no, self.no * self.nc, l_edges, weights=self.w)
        self.firing_graph = FiringGraph.from_matrices(
            'AndPatFinal3', sax_I, sax_C, sax_O, ax_levels, self.mask_vertice_drain, self.depth
        )

        return self.firing_graph

    def build_graph_pattern_init(self):

        # Build edges
        l_edges, ax_levels = [], np.ones(self.nc * self.no)

        for i in range(self.no):
            l_edges += [
                (self.input_vertices[i][j], self.core_vertices[i][0]) for j in range(self.ni)
                if self.ni * i + j in self.target_selected[i]
            ]
            ax_levels[i * self.nc] = len(self.target_selected[i])

        for i in range(self.no):
            l_edges += [
                (self.input_vertices[i][j], self.core_vertices[i][1]) for j in range(self.ni)
                if self.ni * i + j not in self.target_selected[i]
            ]

        for i in range(self.no):
            l_edges += [
                (self.core_vertices[i][0], self.core_vertices[i][self.nc - 1]),
                (self.core_vertices[i][1], self.core_vertices[i][self.nc - 1]),
                (self.core_vertices[i][self.nc - 1], self.output_vertices[i])
            ]
            ax_levels[i * self.nc + self.nc - 1] = self.nc - 1

        # Build Firing graph
        sax_I, sax_C, sax_O = mat_from_tuples(self.ni * self.no, self.no, self.no * self.nc, l_edges, weights=self.w)
        self.firing_graph = FiringGraph.from_matrices(
            'AndPatInit3', sax_I, sax_C, sax_O, ax_levels, self.mask_vertice_drain, self.depth
        )

        return self.firing_graph

    def build_deterministic_io(self):
        ax_inputs = np.zeros((1, self.ni * self.no))

        for i in range(self.no):
            ax_inputs_ = np.zeros((1, self.ni * self.no))
            ax_inputs_[0, self.target[i]] = 1

            ax_inputs = np.vstack((ax_inputs, ax_inputs_))

        ax_outputs = np.eye(self.no)

        return ax_inputs[1:, :], ax_outputs

    def build_random_io(self):

        # Generate random inputs
        ax_inputs = np.random.binomial(1, self.p, (1, self.ni * self.no))

        if not self.random_target:
            for i in range(self.no):
                ax_inputs[0, self.target[i]] = 0

                if np.random.binomial(1, 0.5) == 1:
                    ax_inputs[0, self.target_selected[i]] = 1

        # Generate outputs
        ax_outputs = np.zeros((1, self.no))

        return ax_inputs, ax_outputs

    def init_io_sequence(self):
        return np.zeros((1, self.ni * self.no)), np.zeros((1, self.no))

    def layout(self):
        pos = dict()

        pos.update({'inputs': {'pos': {i: (0, i) for i in range(self.ni * self.no)}, 'color': 'r'}})
        n = self.ni * self.no

        pos['cores'] = {'pos': {}, 'color': 'k'}
        for i in range(self.no):
            pos['cores']['pos'].update(
                {n: (1, ((self.ni * i) * 3 + (self.ni * (i + 1))) / 4),
                 n + 1:  (1, ((self.ni * i) + (self.ni * (i + 1)) * 3) / 4),
                 n + 2: (2, ((self.ni * i) + (self.ni * (i + 1))) / 2)}
            )
            n += self.nc

        pos.update({'outputs': {
            'pos': {(n + i): (3, ((self.ni * i) + (self.ni * (i + 1))) / 2) for i in range(self.no)},
            'color': 'b'}
        })

        return pos
