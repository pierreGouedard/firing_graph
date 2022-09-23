# Global import
from scipy.sparse import diags
from numpy import zeros, ceil, int32

# Local import


def set_drain_params(ax_precision, margin, min_feedback=5):
    # Get params
    ax_precision = ax_precision.clip(max=1., min=margin + 0.01)

    # Compute penalty and reward values
    ax_p, ax_r = set_feedbacks(ax_precision - margin, ax_precision - (margin / 2))

    # Compute weights
    ax_w = ((ax_p - ((ax_precision - margin) * (ax_p + ax_r))) * min_feedback).astype(int) + 1

    return ax_p, ax_r, ax_w.astype(int32)


def set_feedbacks(ax_phi_old, ax_phi_new, r_max=1000):
    ax_p, ax_r = zeros(ax_phi_new.shape), zeros(ax_phi_new.shape)
    for i, (phi_old, phi_new) in enumerate(zip(*[ax_phi_old, ax_phi_new])):
        p, r = set_feedback(phi_old, phi_new, r_max)
        ax_p[i], ax_r[i] = p, r

    return ax_p, ax_r


def set_feedback(phi_old, phi_new, r_max=1000):
    for r in range(r_max):
        p = ceil(r * phi_old / (1 - phi_old))
        score = (phi_new * (p + r)) - p
        if score > 0.:
            return p, r


def clip_inputs(sax_w, sax_c, ax_init_w, ax_p, ax_r, ax_thr):

    # Get input mask from weights and count
    sax_mask = (sax_w > 0).multiply(sax_c > 0)

    # build nom and denominator that enable to retrieve precision
    sax_nom = sax_w.multiply(sax_mask) - sax_mask.dot(diags(ax_init_w, format='csc'))
    sax_denom = sax_mask.multiply(sax_c.dot(diags(ax_p + ax_r, format='csc')))

    # Compute precision of each entries
    sax_precision = sax_nom.multiply(sax_denom.astype(float).power(-1))
    sax_precision += (sax_precision != 0).dot(diags(ax_p / (ax_p + ax_r), format='csc'))

    # Set to 0 precision entries where precision is below specified threshold
    sax_precision = sax_precision > (sax_precision > 0).dot(diags(ax_thr, format='csc'))

    return sax_precision
