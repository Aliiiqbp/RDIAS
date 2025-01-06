import imagehash
from PIL import Image


class PerceptualHash:
    def compute_hash(self, image):
        raise NotImplementedError


class PerceptualHashAlgorithm1(PerceptualHash):
    def __init__(self, hash_size):
        self.hash_size = hash_size

    def compute_hash(self, image):
        # Implement hash computation logic for Algorithm 1
        hash_value = imagehash.phash(image, hash_size=self.hash_size)
        return str(hash_value)


class PerceptualHashAlgorithm2(PerceptualHash):
    def __init__(self, hash_size):
        self.hash_size = hash_size

    def compute_hash(self, image):
        # Implement hash computation logic for Algorithm 2
        pass


# Add more algorithms as needed

class PerceptualHashFactory:
    @staticmethod
    def create(algorithm, hash_size):
        if algorithm == "phash":
            return PerceptualHashAlgorithm1(hash_size)
        elif algorithm == "algorithm2":
            return PerceptualHashAlgorithm2(hash_size)
        # Add more algorithms as needed
        else:
            raise ValueError(f"Unknown perceptual hash algorithm: {algorithm}")
