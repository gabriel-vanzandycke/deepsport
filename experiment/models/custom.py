from experimentator import ChunkProcessor
from tf_layers import GammaColorAugmentation


class GammaAugmentation(ChunkProcessor):
    def __init__(self, tensor_name):
        """ Performs Gama Data-Augmentation on `tensor_name`.
            One single name is allowed because the same random augmentation should be applied to dependant items.
        """
        # create random gamma for RGB with R in [0.96, 1.04], G in [0.98, 1.02] and B in [0.96, 1.04]
        self.layer = GammaColorAugmentation([0.02, 0.01, 0.02])
        self.tensor_name = tensor_name
    def color_augmentation(self, chunk):
        assert chunk[self.tensor_name].shape[3] == 3, "GamaAugmentation must be done on RGB images"
        chunk[self.tensor_name] = self.layer(chunk[self.tensor_name])
    def __call__(self, chunk):
        self.color_augmentation(chunk)