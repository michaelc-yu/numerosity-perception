
from scipy.io import loadmat

data = loadmat('DeepNetwork-1500-1000-Mature.mat')

print(data.keys())
print(data['DN'])

DN = data['DN']
print(type(DN), DN.shape)

dn = DN[0, 0]   # extract the struct
print(dn.dtype)

L = dn['L']
print(type(L), L.shape)

layer0 = L[0, 0]
layer1 = L[0, 1]

print(layer0.dtype.names)
print(layer1.dtype.names)

