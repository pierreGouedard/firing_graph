# Global import
from scipy.sparse import csc_matrix, csr_matrix
import numpy as np

# Local import
from ..tools.equations.backward import bpo, bpc, btc
from ..tools.equations.forward import fti, ftc, fto, fpi, fpc, fpo
from ..tools.equations.graph import buc, buo, bui


class FiringGraphDrainer(object):
    def __init__(self, firing_graph, server, batch_size, penalties=None, rewards=None, t=-1, verbose=0):

        # Core params
        self.ax_p, self.ax_r = penalties, rewards
        self.t = t
        self.bs = batch_size
        self.firing_graph = firing_graph
        self.verbose = verbose

        # stream feed forward and backward
        self.server = server

        # Init signals
        self.sax_i, self.sax_c, self.sax_o = init_forward_signal(self.firing_graph, self.bs)
        self.sax_im, self.sax_cm = init_forward_memory(self.firing_graph, self.bs)
        self.sax_cb, self.sax_ob = init_backward_signal(self.firing_graph, self.bs, p=max(self.ax_p), r=max(self.ax_r))

        # set dtypes of server
        self.server.dtype_forward = self.sax_i.dtype
        self.server.dtype_backward = self.sax_cb.dtype

        # Init iteration count
        self.iter = 0

    def reset_all(self, server=False):

        self.reset_forward()
        self.reset_backward()
        self.iter = 0

        if server:
            self.reset_server()

    def reset_server(self):
        self.server.stream_features()

    def reset_forward(self):
        self.sax_i, self.sax_c, self.sax_o = init_forward_signal(self.firing_graph, self.bs, self.server.dtype_forward)
        self.sax_im, self.sax_cm = init_forward_memory(self.firing_graph, self.bs, self.server.dtype_forward)

    def reset_backward(self):
        self.sax_cb, self.sax_ob = init_backward_signal(self.firing_graph, self.bs, self.server.dtype_backward)

    def drain_all(self, n_max=10000, adapt_bs=False):

        stop, n = False, 0
        while not stop:
            # Drain and reset signals
            self.drain()
            self.reset_all()

            # Stop conditions
            if self.firing_graph.Im.nnz == 0 and self.firing_graph.Cm.nnz == 0 and self.firing_graph.Om.nnz == 0:
                stop = True

            n += self.bs
            if n >= n_max:
                stop = True

            # Adapt batch size if specified
            if adapt_bs and not stop and self.t > 0:
                self.adapt_batch_size(self.bs)
                self.reset_all()

        print("[Drainer]: {} samples has been propagated through firing graph".format(n))

        return self

    def drain(self, n=1):
        early_stopping, j = False, 0
        while j < n:
            self.run_iteration(True, True)

            # Condition of stop that has to be put in the drainer
            if self.firing_graph.Im.nnz == 0 and self.firing_graph.Cm.nnz == 0 and self.firing_graph.Om.nnz == 0:
                early_stopping = True
                self.server.synchonize_steps()
                break

            # Increment count iteration
            j += 1

        # Flush remaining forward and backward signals
        if not early_stopping:
            self.flush_signals()

        # Make sure forward and backward signal are synchronized.
        if not early_stopping:
            self.server.check_synchro()

        return self

    def run_iteration(self, load_input, load_output):
        # Forward pass
        self.forward_transmiting(load_input=load_input)
        self.forward_processing(load_output=load_output)

        # Backward pass
        self.backward_processing()
        self.backward_transmiting()

        # Increment iteration nb
        self.iter += 1

    def flush_signals(self):
        for _ in range(self.firing_graph.depth - 1):
            self.run_iteration(False, True)

        for _ in range(self.firing_graph.depth - 1):
            self.run_iteration(False, False)

    def forward_transmiting(self, load_input=True):
        # Get new input
        if load_input:
            self.sax_i = fti(self.server, self.firing_graph, self.bs)
        else:
            self.sax_i = csr_matrix((self.bs, self.firing_graph.I.shape[0]), dtype=self.sax_i.dtype)

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
            self.sax_ob = fpo(self.sax_o, self.server, self.bs, self.ax_p, self.ax_r)

        else:
            self.sax_ob = csc_matrix((self.firing_graph.O.shape[1], self.bs), dtype=self.sax_ob.dtype)

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

        # Backward firing_graph processing: decay feedback by batch size
        self.sax_ob = bpo(self.sax_ob, get_mem_size(self.bs, self.firing_graph.depth), self.bs)

        # Backward firing_graph processing: decay backward signal by 2 * batch size
        self.sax_cb = bpc(self.sax_cb, self.bs)


def get_mem_size(batch_size, depth):
    return batch_size + ((depth - 1) * 2 * batch_size + batch_size)


def init_forward_signal(fg, batch_size, dtype=None):

    if dtype is None:
        dtype = set_forward_type(fg)

    sax_i = csr_matrix((batch_size, fg.I.shape[0]), dtype=dtype)
    sax_c = csr_matrix((batch_size, fg.C.shape[0]), dtype=dtype)
    sax_o = csr_matrix((batch_size, fg.O.shape[1]), dtype=dtype)

    return sax_i, sax_c, sax_o


def init_forward_memory(fg, batch_size, dtype=None):

    if dtype is None:
        dtype = set_forward_type(fg)

    # Get memory size needed
    mem_size = get_mem_size(batch_size, fg.depth)

    # Init memory signals
    sax_im = csr_matrix((mem_size, fg.I.shape[0]), dtype=dtype)
    sax_cm = csr_matrix((mem_size, fg.C.shape[0]), dtype=dtype)

    return sax_im, sax_cm


def init_backward_signal(fg, batch_size, dtype=None, p=None, r=None):

    if dtype is None:
        dtype = set_backward_type(fg, p, r)

    # Get memory size needed
    mem_size = get_mem_size(batch_size, fg.depth)

    # Init backward signals
    sax_cb = csc_matrix((fg.C.shape[0], mem_size), dtype=dtype)
    sax_ob = csc_matrix((fg.O.shape[1], mem_size), dtype=dtype)

    return sax_cb, sax_ob


def set_forward_type(fg):
    max_value = max([fg.I.sum(axis=0).max(), fg.C.sum(axis=0).max(), fg.O.sum(axis=0).max()])
    if max_value < np.iinfo(np.uint8).max:
        dtype = np.uint8

    elif max_value < np.iinfo(np.uint16).max:
        dtype = np.uint16

    else:
        dtype = np.uint32

    return dtype


def set_backward_type(fg, p, r):

    max_outcoming = max([fg.I.sum(axis=1).max(), fg.C.sum(axis=1).max(), fg.O.sum(axis=1).max()])
    max_value, min_value = max_outcoming * r, max_outcoming * -1 * p

    if max_value < np.iinfo(np.int8).max and min_value > np.iinfo(np.int8).min:
        dtype = np.int8

    elif max_value < np.iinfo(np.int16).max and min_value > np.iinfo(np.int16).min:
        dtype = np.int16

    else:
        dtype = np.int32

    return dtype
