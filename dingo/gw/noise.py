from typing import Dict


# TODO:
#  * Support transformations
#    1. add noise to detector_projected wf sample
#    2. whiten
#  * Caveat: Don't use PSDs with float32!
#  * simplest case: fixed designed sensitivity PSD
#  * noise needs to know the domain
#  * more complex: database of PSDs for each detector
#    - randomly select a PSD for each detector
#  * Maybe create a PSD_DataSet class (open / non-open data), and transform
#    - at each call randomly draw a psd
#  * window_function
# bilby's create_white_noise

# TODO: Noise class needs to provide:
# 1. sample_noise()
# 2. provide_context() -- will not modify a tensor, but spit out a tensor that has the shape of expected noise summary
#     particular function of the PSD
class PSD:
    def sample_noise(self):
        pass

    def provide_context(self):
        pass

class PSDDataSet:
    """draw PSD object()"""
    # Where does the random choice of index come in?
    # transform class would have to call sample_index() and then sample_...
    def sample_index(self):
        index = .... # numpy generator -- be careful about setting seed

    def sample_noise(self):
        self._sample_noise(self, self.index)

    def _sample_noise(self, index):
        pass

    def provide_context(self, index):
        pass


class AddNoiseAndWhiten:

    def __init(self):
        pass

    def __call__(self, strain_dict: Dict[str, np.ndarray],
                 psd: PSD) -> Dict[str, np.ndarray]:
        """
        Transform detector strain data transformation
        #    1. add noise to detector_projected wf sample
        #    2. whiten
        """

        pass
