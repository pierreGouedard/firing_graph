# Global imports
import pickle

# Local imports


class FileImputer(object):

    def __init__(self, path_forward, path_backward, streamer, is_cyclic=True):

        # Define core attribute of the file imputer
        self.path_forward = path_forward
        self.path_backward = path_backward
        self.streamer = streamer
        self.is_cyclic = is_cyclic

        # Initialize stream to None
        self.stream_forward, self.stream_backward = None, None

    def stream_data(self):

        # Stream I/O
        self.stream_forward = self.streamer.create_stream(
            self.path_forward, is_cyclic=self.is_cyclic, orient='row'
        )

        self.stream_backward = self.streamer.create_stream(
            self.path_backward, is_cyclic=self.is_cyclic, orient='row'
        )

    def next_forward(self):
        return self.stream_forward.next()

    def next_backward(self):
        return self.stream_backward.next()

    def save_as_pickle(self, path):
        with open(path, 'wb') as handle:
            pickle.dump(self, handle)

    @staticmethod
    def load_pickle(path):

        with open(path, 'rb') as handle:
            imputer = pickle.load(handle)

        return imputer


class ArrayImputer(object):

    def __init__(self, sax_forward, sax_backward, is_cyclic=True, orient='row'):

        # Define core attribute of the file imputer
        self.sax_forward = sax_forward
        self.sax_backward = sax_backward
        self.is_cyclic = is_cyclic

        # Define streaming features
        self.step, self.orient, self.step_forward, self.step_backward = None, orient, None, None

    def stream_features(self):
        self.step_forward, self.step_backward = 0, 0

    def next_forward(self):
        if self.orient == 'row':

            if self.step_forward < self.sax_forward.shape[0]:
                sax_next = self.sax_forward[self.step_forward, :]
            else:
                return None

            self.step_forward += 1

            if self.is_cyclic:
                self.step_forward = self.step_forward % self.sax_forward.shape[0]

        else:
            if self.step_forward < self.sax_forward.shape[-1]:
                sax_next = self.sax_forward[:, self.step_forward]

            else:
                return None

            self.step_forward += 1

            if self.is_cyclic:
                self.step_forward = self.step_forward % self.sax_forward.shape[-1]

        return sax_next

    def next_backward(self):
        if self.orient == 'row':

            if self.step_backward < self.sax_backward.shape[0]:
                sax_next = self.sax_backward[self.step_backward, :]
            else:
                return None

            self.step_backward += 1

            if self.is_cyclic:
                self.step_backward = self.step_backward % self.sax_backward.shape[0]

        else:
            if self.step_backward < self.sax_backward.shape[-1]:
                sax_next = self.sax_backward[:, self.step_backward]

            else:
                return None

            self.step_backward += 1

            if self.is_cyclic:
                self.step_backward = self.step_backward % self.sax_backward.shape[-1]

        return sax_next

    def save_as_pickle(self, path):
        with open(path, 'wb') as handle:
            pickle.dump(self, handle)

    @staticmethod
    def load_pickle(path):

        with open(path, 'rb') as handle:
            imputer = pickle.load(handle)

        return imputer






