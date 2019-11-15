# Global imports
import unittest
from scipy.sparse import csc_matrix
import numpy as np

# Local import
from utils.nmp import NumpyDriver

__maintainer__ = 'Pierre Gouedard'


class TestDriverNumpy(unittest.TestCase):
    # TODO: update (Obsolete test)
    def setUp(self):
        np.random.seed(1393)

        self.driver = NumpyDriver()

        # Init different kind of numpy array
        self.s = np.random.randn(10)
        self.m = np.random.randn(10, 10)
        self.d_s = {'part_{}'.format(i): np.random.randn(10) for i in range(10)}
        self.d_m = {'part_{}'.format(i): np.random.randn(10, 10) for i in range(10)}

    def test_load_and_save(self):
        """

        python -m unittest tests.unit.driver.test_numpy.TestDriverNumpy.test_load_and_save

        """

        # Create tmp dir
        tmp_dir = self.driver.TempDir(prefix='test_driver_', create=True)

        # Save numpy 1d array
        self.driver.write_file(self.s, self.driver.join(tmp_dir.path, 'ax.npy'))

        # Load numpy array
        ax = self.driver.read_file(self.driver.join(tmp_dir.path, 'ax.npy'))

        assert(all(self.s == ax))

        # Do the same for sparse array
        self.driver.write_file(csc_matrix(self.s), self.driver.join(tmp_dir.path, 'ax.npz'), **{'is_sparse': True})

        # Load numpy array
        ax = self.driver.read_file(self.driver.join(tmp_dir.path, 'ax.npz'),  **{'is_sparse': True})

        assert (all(self.s == ax.toarray()[0]))

        # Remove tmp dir
        tmp_dir.remove()

    def test_load_save_partition(self):
        """

        python -m unittest tests.unit.driver.test_numpy.TestDriverNumpy.test_load_save_partition

        """

        # Create tmp dir
        tmp_dir = self.driver.TempDir(prefix='test_driver_', create=True)

        # Save numpy 1d array
        self.driver.write_partioned_file(self.d_s, tmp_dir.path)

        # Load numpy array
        d_ax = self.driver.read_partitioned_file(tmp_dir.path)

        for k in d_ax.keys():
            assert all(self.d_s[k] == d_ax[k])

        # Do the same for sparse array
        self.driver.write_partioned_file({csc_matrix(self.d_s[k]) for k in self.d_s.keys()}, tmp_dir.path,
                                         **{'is_sparse': True})

        # Load numpy array
        d_ax = self.driver.read_partitioned_file(self.driver.join(tmp_dir.path, 'ax.npz'),  **{'is_sparse': True})

        for k in d_ax.keys():
            assert all(self.d_s[k] == d_ax[k].toarray()[0])

        # Remove tmp dir
        tmp_dir.remove()

        raise NotImplementedError

    def test_stream(self):
        """

        python -m unittest tests.unit.driver.test_numpy.TestDriverNumpy.test_stream

        """

        # Create tmp dir
        tmp_dir = self.driver.TempDir(prefix='test_driver_', create=True)

        # Save numpy 1d array
        self.driver.write_partioned_file(self.d_m, tmp_dir.path)

        # Stream numpy array case 1: non cyclic, n_cahce divide number of partittion
        key = lambda x: int(x.split('_')[1].split('.')[0])
        stream = self.driver.init_stream_partition(tmp_dir.path, key_partition=key, n_cache=2, orient='columns')

        # Make sure the stream is ok
        for k in range(4):
            for j in range(2):
                for i in range(10):
                    ax = stream.stream_next()
                    assert all(ax == self.d_m['part_{}'.format((k * 2) + j)][:, i])

            assert stream.d_stream['offset'] == (k + 2) * 2 and stream.d_stream['step'] == 0

        for j in range(2):
            for i in range(10):
                ax = stream.stream_next()
                assert all(ax == self.d_m['part_{}'.format(((k + 1) * 2) + j)][:, i])

        assert stream.stream_next() is None

        # Stream numpy array case 2: non cyclic, n_cache does not divide number of partittion
        stream = self.driver.init_stream_partition(tmp_dir.path, key_partition=key, n_cache=4, orient='columns')

        # Make sure the stream is ok
        for k in range(2):
            for j in range(4):
                for i in range(10):
                    ax = stream.stream_next()
                    assert all(ax == self.d_m['part_{}'.format((k * 4) + j)][:, i])

            assert stream.d_stream['offset'] == (k + 4) * 2 and stream.d_stream['step'] == 0

        for j in range(2):
            for i in range(10):
                ax = stream.stream_next()
                assert all(ax == self.d_m['part_{}'.format(((k + 1) * 4) + j)][:, i])

        assert stream.stream_next() is None

        # Stream numpy array case 3: cyclic, n_cache does not divide number of partittion
        stream = self.driver.init_stream_partition(tmp_dir.path, key_partition=key, n_cache=4, orient='columns',
                                                         is_cyclic=True)

        # Make sure the stream is ok
        for k in range(2):
            for j in range(4):
                for i in range(10):
                    ax = stream.stream_next()
                    assert all(ax == self.d_m['part_{}'.format((k * 4) + j)][:, i])

            assert stream.d_stream['offset'] == (k + 4) * 2 and stream.d_stream['step'] == 0

        for j in range(2):
            for i in range(10):
                ax = stream.stream_next()
                assert all(ax == self.d_m['part_{}'.format(((k + 1) * 4) + j)][:, i])

        # Make sure the stream is again ok
        for k in range(2):
            for j in range(4):
                for i in range(10):
                    ax = stream.stream_next()
                    assert all(ax == self.d_m['part_{}'.format((k * 4) + j)][:, i])

        tmp_dir.remove()
