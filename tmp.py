import pdqhash
import cv2
from PIL import Image
import numpy as np

def pdq_string_hash(image):
    tmp_image = Image.open(image)
    tmp_image = np.array(tmp_image)
    # tmp_image = cv2.cvtColor(tmp_image, cv2.COLOR_BGR2RGB)
    hash_vector, _ = pdqhash.compute(tmp_image)
    string_representation = ''.join(map(str, hash_vector.tolist()))
    return string_representation


print(pdq_string_hash('tmp.jpeg'))

