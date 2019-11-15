# Global imports
import pickle

# Local imports
from utils.driver import FileDriver
base_driver = FileDriver('imputer file driver', '')


class Imputer(object):

    def __init__(self, project, dirin, dirout):

        self.project = project

        self.dirin = dirin
        self.dirout = dirout

        self.features_forward = None
        self.features_backward = None
        self.stream_forward = None
        self.stream_backward = None

    def read_raw_data(self, **kwargs):
        raise NotImplementedError

    def read_features(self, **kwargs):
        raise NotImplementedError

    def stream_features(self, **kwargs):
        raise NotImplementedError

    def stream_next_forward(self):
        return self.stream_forward.stream_next()

    def stream_next_backward(self):
        return self.stream_backward.stream_next()

    def write_features(self, name_forward, name_backward):
        raise NotImplementedError

    def run_preprocessing(self):
        raise NotImplementedError

    def run_postprocessing(self, d_features):
        raise NotImplementedError

    def save_as_pickle(self, path):
        with open(path, 'wb') as handle:
            pickle.dump(self, handle)

    @staticmethod
    def load(project):
        pth = settings.deyep_imputer_path.format(project)

        with open(base_driver.join(pth, 'imputer.pickle'), 'rb') as handle:
            imputer = pickle.load(handle)

        return imputer


class ImputerSingleSource(Imputer):

    def __init__(self, project, dirin, dirout):
        Imputer.__init__(self, project, dirin, dirout)

        self.raw_data = None
        self.name_forward = None
        self.name_backward = None

    def read_raw_data(self, name):
        raise NotImplementedError


class ImputerDoubleSource(Imputer):

    def __init__(self, project, dirin, dirout):
        Imputer.__init__(self, project, dirin, dirout)

        self.raw_data_forward = None
        self.raw_data_backward = None
        self.name_forward = None
        self.name_backward = None

    def read_raw_data(self, name_forward, name_backward):
        raise NotImplementedError





