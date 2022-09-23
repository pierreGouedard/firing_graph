# Global import
from numpy import int32, ones


def bui(sax_cb, sax_i, firing_graph):

    # Build mask
    sax_mask = firing_graph.I.multiply(firing_graph.Im)

    # Update inputs
    firing_graph.matrices['Iw'] += sax_i.T.dot(sax_cb).multiply(sax_mask)

    # Return count of updates
    sax_cb.data = ones(len(sax_cb.data), dtype=int32)
    return sax_i.T.dot(sax_cb).multiply(sax_mask)
