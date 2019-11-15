# global import
from utils.nmp import NumpyDriver

# Local import
from core.imputer import ImputerDoubleSource


class DoubleArrayImputer(ImputerDoubleSource):

    def __init__(self, project, dirin, dirout, is_sparse=True):

        ImputerDoubleSource.__init__(self, project, dirin, dirout)
        self.driver = NumpyDriver()
        self.is_sparse = is_sparse
        self.name_forward, self.name_backward = None, None

    def copy(self):
        return DoubleArrayImputer(self.project, self.dirin, self.dirout, self.is_sparse)

    def read_raw_data(self, name_forward, name_backward):

        # Read raw data
        self.raw_data_forward = self.driver.read_file(self.driver.join(self.dirin, name_forward),
                                                      is_sparse=self.is_sparse)
        self.raw_data_backward = self.driver.read_file(self.driver.join(self.dirin, name_backward),
                                                       is_sparse=self.is_sparse)

    def read_features(self):

        # Read data
        self.features_forward = self.driver.read_file(self.driver.join(self.dirout, self.name_forward),
                                                      is_sparse=self.is_sparse)
        self.features_backward = self.driver.read_file(self.driver.join(self.dirout, self.name_backward),
                                                       is_sparse=self.is_sparse)

        return self

    def write_features(self, name_forward, name_backward):

        # Set urls
        self.name_forward, self.name_backward = name_forward, name_backward

        # Write data
        self.driver.write_file(self.features_forward, self.driver.join(self.dirout, name_forward), is_sparse=True)
        self.driver.write_file(self.features_backward, self.driver.join(self.dirout, name_backward), is_sparse=True)

    def stream_features(self, partition=None, is_cyclic=True):

        # Set urls
        urlf, urlb = self.driver.join(self.dirout, self.name_forward), self.driver.join(self.dirout, self.name_backward)
        driverf, driverb = NumpyDriver(), NumpyDriver()

        # Stream I/O
        if partition is not None:
            self.stream_forward = driverf.init_stream_partition(urlf, n_cache=1, orient='row',
                                                                is_sparse=self.is_sparse, is_cyclic=True)
            self.stream_backward = driverb.init_stream_partition(urlb, n_cache=1, orient='row',
                                                                 is_sparse=self.is_sparse, is_cyclic=True)
        else:
            self.stream_forward = driverf.init_stream(urlf, is_sparse=self.is_sparse, is_cyclic=is_cyclic, orient='row')
            self.stream_backward = driverb.init_stream(urlb, is_sparse=self.is_sparse, is_cyclic=is_cyclic,
                                                       orient='row')

    def run_preprocessing(self):
        self.features_forward = self.raw_data_forward.copy()
        self.features_backward = self.raw_data_backward.copy()

    def run_postprocessing(self, d_features):
        raise NotImplementedError


