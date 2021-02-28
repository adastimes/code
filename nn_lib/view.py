import numpy as np
from PIL import Image


vec = np.loadtxt("/mnt/nvme/out.txt")
out = np.reshape(vec,[2,256,512])
label = out.argmax(axis=0)
vec = np.loadtxt("/mnt/nvme/image.txt")
vec2 = np.reshape(vec,[3,256,512])
vec2[1,:,:] += label * 100
inp = np.moveaxis(vec2, 0, 2)
img = Image.fromarray(inp.astype(np.uint8), 'RGB')
img.show()