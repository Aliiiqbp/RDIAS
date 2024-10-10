import math
from scipy.special import erfc

def P_c(T, mu, delta):
    return 0.5 * erfc((mu - T) / (math.sqrt(2) * delta))


k = [0, 1, 2, 4, 8]
mu = 0.4976  # Example value for mu
delta = 0.0319  # Example value for delta


for k in k:
    result = P_c(k * delta, mu, delta)
    print(f"P_c({k*delta}) = {result}")


'''
a-64:
mu >>> 0.1198 - delta >>> 0.0303
a-256:
mu >>> 0.4826 - delta >>> 0.0985

d-64:
mu >>> 0.1232 - delta >>> 0.0189
d-256:
mu >>> 0.4946 - delta >>> 0.0398

p-64:
mu >>> 0.1226 - delta >>> 0.0160
p-256:
mu >>> 0.4976 - delta >>> 0.0319

w-64:
mu >>> 0.1199 - delta >>> 0.0303
w-256:
mu >>> 0.4821 - delta >>> 0.0993
'''