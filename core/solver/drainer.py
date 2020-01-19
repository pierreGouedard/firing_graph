# Global import
from scipy.sparse import csc_matrix, csr_matrix
import numpy as np

# Local import
from ..tools.equations.backward import bpo, bpc, btc
from ..tools.equations.forward import fti, ftc, fto, fpi, fpc, fpo
from ..tools.equations.structure import buc, buo, bui


class FiringGraphDrainer(object):
    def __init__(self, t, p, q, batch_size, firing_graph, imputer, verbose=0):

        # Core params
        self.p, self.q = p, q
        self.t = t
        self.bs = batch_size
        self.firing_graph = firing_graph
        self.verbose = verbose

        # stream feed forward and backward
        self.imputer = imputer

        # Init signals
        self.sax_i, self.sax_c, self.sax_o = init_forward_signal(self.firing_graph, batch_size)
        self.sax_im, self.sax_cm = init_forward_memory(self.firing_graph, batch_size)
        self.sax_cb, self.sax_ob = init_backward_signal(self.firing_graph, batch_size)
        self.iter = 0

    def reset_all(self, imputer=False):

        self.reset_forward()
        self.reset_backward()
        self.iter = 0

        if imputer:
            self.reset_imputer()

    def reset_imputer(self):
        self.imputer.stream_features()

    def reset_forward(self):
        self.sax_i, self.sax_c, self.sax_o = init_forward_signal(self.firing_graph, self.bs)
        self.sax_im, self.sax_cm = init_forward_memory(self.firing_graph, self.bs)

    def reset_backward(self):
        self.sax_cb, self.sax_ob = init_backward_signal(self.firing_graph, self.bs)

    def drain_all(self, t_max=10000, adapt_bs=False):

        stop, t, max_batch_size = False, 0, self.bs
        while not stop:
            for _ in range(int(max_batch_size / self.bs)):
                # Drain and reset signals
                self.drain()
                self.reset_all()

            # Stop conditions
            if self.firing_graph.Im.nnz == 0 and self.firing_graph.Cm.nnz == 0 and self.firing_graph.Om.nnz == 0:
                stop = True

            if t > t_max:
                stop = True

            j += self.bs
            print("[Drainer]: {} iterations has been completed".format(j * self.bs))

            # Adapt batch size if specified
            if adapt_bs:
                self.adapt_batch_size(max_batch_size)

    def drain(self, n=1):

        early_stopping, j = False, 0
        while j < n:
            self.run_iteration(True, True)

            # Display
            if j % 100 == 0 and j != 0:
                print("[Drainer]: {} batch has been completed".format(j))

            # Condition of stop that has to be put in the drainer
            if self.firing_graph.Im.nnz == 0 and self.firing_graph.Cm.nnz == 0 and self.firing_graph.Om.nnz == 0:
                early_stopping = True
                break

            # Increment count iteration
            j += 1

        # Flush remaining forward and backward signals
        if not early_stopping:
            self.flush_signals()

        return self

    def flush_signals(self):
        for _ in range(self.firing_graph.depth - 1):
            self.run_iteration(False, True)

        for _ in range(self.firing_graph.depth - 1):
            self.run_iteration(False, False)

    def run_iteration(self, load_input, load_output):
        # Forward pass
        self.forward_transmiting(load_input=load_input)
        self.forward_processing(load_output=load_output)

        # Backward pass
        self.backward_processing()
        self.backward_transmiting()

        # Increment iteration nb
        self.iter += 1

    def adapt_batch_size(self, max_batch_size):

        l_batch_size = []
        # With output matrix
        if self.firing_graph.Om.nnz > 0:
            sax_bfo = self.firing_graph.backward_firing['o'].multiply(self.firing_graph.Om)
            l_batch_size += [max(min(self.t - sax_bfo.tocsc().max(), max_batch_size), 1)]

        # With Core matrix
        if self.firing_graph.Cm.nnz > 0:
            sax_bfc = self.firing_graph.backward_firing['c'].multiply(self.firing_graph.Cm)
            l_batch_size += [max(min(self.t - sax_bfc.tocsc().max(), max_batch_size), 1)]

        # With Input matrix
        if self.firing_graph.Cm.nnz > 0:
            sax_bfi = self.firing_graph.backward_firing['i'].multiply(self.firing_graph.Im)
            l_batch_size += [max(min(self.t - sax_bfi.tocsc().max(), max_batch_size), 1)]

        # Adapt batch size
        batch_size = min(l_batch_size)
        self.bs = int(max_batch_size / np.ceil(max_batch_size / batch_size))

    def forward_transmiting(self, load_input=True):
        # Get new input
        if load_input:
            self.sax_i = fti(self.imputer, self.firing_graph, self.bs)
        else:
            self.sax_i = csr_matrix((self.bs, self.firing_graph.I.shape[0]), dtype=int)

        # Output transmit
        self.sax_o = fto(self.firing_graph.O, self.sax_c)

        # Core transmit
        self.sax_c = ftc(self.firing_graph.C, self.firing_graph.I, self.sax_c, self.sax_i)

    def forward_processing(self, load_output=True):

        # Transform signals and update memory of forward
        self.sax_im = fpi(self.sax_i, self.sax_im)
        self.sax_c, self.sax_cm = fpc(self.sax_c, self.sax_cm, self.firing_graph.levels)

        # If decay reached compute feedback
        if self.iter >= self.firing_graph.depth - 1 and load_output:
            self.sax_ob = fpo(self.sax_o, self.imputer, self.bs, self.p, self.q)

        else:
            self.sax_ob = csc_matrix((self.firing_graph.O.shape[1], self.bs), dtype=int)

    def backward_transmiting(self):

        # Update Output matrix
        if self.firing_graph.Om.nnz > 0:
            sax_track = buo(self.sax_ob, self.sax_cm, self.firing_graph)
            self.firing_graph.update_backward_firing('O', sax_track)

        # Update Core matrix adjacency matrix and drainer mask
        if self.firing_graph.Cm.nnz > 0:
            sax_track = buc(self.sax_cb, self.sax_cm, self.firing_graph)
            self.firing_graph.update_backward_firing('C', sax_track)

        # Update Input matrix
        if self.firing_graph.Im.nnz > 0:
            sax_track = bui(self.sax_cb, self.sax_im, self.firing_graph)
            self.firing_graph.update_backward_firing('I', sax_track)

        self.firing_graph.update_mask(self.t)

        # Backward transmit
        self.sax_cb = btc(self.sax_ob, self.sax_cb, self.sax_cm, self.firing_graph.O, self.firing_graph.C)

    def backward_processing(self):

        # Backward core processing: decay feedback by batch size
        self.sax_ob = bpo(self.sax_ob, get_mem_size(self.bs, self.firing_graph.depth), self.bs)

        # Backward core processing: decay backward signal by 2 * batch size
        self.sax_cb = bpc(self.sax_cb, self.bs)


def get_mem_size(batch_size, depth):
    return batch_size + ((depth - 1) * 2 * batch_size + batch_size)


def init_forward_signal(fg, batch_size):
    sax_i = csc_matrix((batch_size, fg.I.shape[0]), dtype=int)
    sax_c = csc_matrix((batch_size, fg.C.shape[0]), dtype=int)
    sax_o = csc_matrix((batch_size, fg.O.shape[1]), dtype=int)
    return sax_i, sax_c, sax_o


def init_forward_memory(fg, batch_size):
    # Get memory size needed
    mem_size = get_mem_size(batch_size, fg.depth)

    # Init memory signals
    sax_im = csr_matrix((mem_size, fg.I.shape[0]), dtype=int)
    sax_cm = csr_matrix((mem_size, fg.C.shape[0]), dtype=int)

    return sax_im, sax_cm


def init_backward_signal(fg, batch_size):
    # Get memory size needed
    mem_size = get_mem_size(batch_size, fg.depth)

    # Init backward signals
    sax_cb = csc_matrix((fg.C.shape[0], mem_size), dtype=int)
    sax_ob = csc_matrix((fg.O.shape[1], mem_size), dtype=int)

    return sax_cb, sax_ob