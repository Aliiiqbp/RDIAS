from trustmark import TrustMark
from PIL import Image
import imagehash


h1 = imagehash.average_hash(Image.open('0801.png'), hash_size=16)
h2 = imagehash.average_hash(Image.open('0801-w.png'), hash_size=16)

print(h1, h2)
print(type(h1))
print(h1 - h2)


