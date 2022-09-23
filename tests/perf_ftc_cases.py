import time
from random import choices
import threading
from scipy.sparse import csr_matrix, lil_matrix, csc_matrix, hstack, diags
import numpy as np
from sparse_dot_mkl import dot_product_mkl


def seq_test(sax_i, sax_I, l_input_partitions):
    for k, l_parts in enumerate(l_input_partitions):
        s, e = l_parts[0]
        ax_mask = sax_i[:, s:e].dot(sax_I[s:e, k]).A[:, 0]
        if not ax_mask.any():
            yield csc_matrix(ax_mask[:, np.newaxis])
        for s, e in l_parts[1:]:
            ax_mask[ax_mask] = sax_i[ax_mask, s:e].dot(sax_I[s:e, k]).A[:, 0]
            if not ax_mask.any():
                break

        yield csc_matrix(ax_mask[:, np.newaxis])


def pure_scipy(sax_i, sax_I, ax_levels):
    sax_prod = sax_i.astype(np.int32).dot(sax_I)
    return sax_prod > (sax_prod > 0).multiply(csr_matrix((ax_levels - 1).clip(min=0), dtype=np.int32))


def pure_scipy_mkl(sax_i, sax_I, ax_levels):
    sax_prod = dot_product_mkl(sax_i.astype(np.float32), sax_I.astype(np.float32))
    return sax_prod > (sax_prod > 0).multiply(csr_matrix((ax_levels - 1).clip(min=0), dtype=np.int32))


if __name__ == '__main__':
    # TODO: multiprocessing as also been tested outside of this scope, but doesn't at all meet perf of pure scipy
    #   The only increase in performance might be to implement routine in pure c and call it from python or using a
    #   cuda python binding that enable sparse operation:
    #    https://stackoverflow.com/questions/49019189/how-can-i-accelerate-a-sparse-matrix-by-dense-vector-product-currently-implemen
    #   It can be through driver pycuda & Cie or directly using tensorflow operation.

    # Get fake signals
    l_all_partitions = [(i * 50, (i + 1) * 50) for i in range(20)]
    sax_i = csr_matrix(np.random.binomial(1, 0.001, 500000).astype(bool))
    sax_i = sax_i.T[:, [0] * 1000]
    sax_I = csr_matrix(np.random.binomial(1, 0.01, (1000, 600)).astype(bool))
    ax_levels = np.random.randint(0, 5, (600,))
    l_input_partitions = [
        [(s, e) for s, e in choices(l_all_partitions, k=np.random.randint(1, 4))] for _ in range(600)
    ]

    # Sequential
    import time
    print('seq partitioned')
    t0 = time.time()
    sax_res = hstack([x for x in seq_test(sax_i, sax_I, l_input_partitions)], format='csr')
    diff = time.time() - t0
    print(f'sequ: {diff}')

    # Pure Scipy
    print('pure scipy')
    t0 = time.time()
    sax_res = pure_scipy(sax_i, sax_I, ax_levels)
    diff = time.time() - t0
    print(f'sequ: {diff}')

    # Pure Scipy on MKL
    print('pure scipy + mkl')
    t0 = time.time()
    sax_res = pure_scipy_mkl(sax_i, sax_I, ax_levels)
    diff = time.time() - t0
    print(f'sequ: {diff}')

    # Multithread + scipy
    print('Mt + scipy')
    t0 = time.time()

    class MyThread(threading.Thread):

        def __init__(self, k, l_parts):
            threading.Thread.__init__(self)
            self.k = k
            self.i_parts = l_parts
            self.res = None

        def run(self):
            s, e = self.i_parts[0]
            ax_mask = sax_i[:, s:e].dot(sax_I[s:e, self.k]).A[:, 0]
            if not ax_mask.any():
                self.res = csc_matrix(ax_mask[:, np.newaxis])
                return
            for s, e in self.i_parts[1:]:
                ax_mask[ax_mask] = sax_i[ax_mask, s:e].dot(sax_I[s:e, self.k]).A[:, 0]
                if not ax_mask.any():
                    break

            self.res = csc_matrix(ax_mask[:, np.newaxis])


    # Create new threads
    threads = [MyThread(i, l_sub_parts) for i, l_sub_parts in enumerate(l_input_partitions)]

    # Start new Threads
    for t in threads:
        t.start()

    for t in threads:
        t.join()
    sax_res = hstack([t.res for t in threads], format='csr')
    diff = time.time() - t0
    print(f'sequ: {diff}')

    import IPython
    IPython.embed()




