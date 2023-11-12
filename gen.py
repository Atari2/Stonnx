import random
from array import array
import numpy
REQUIRED_SIZE = 3*224*224

with open('test.bin', 'wb') as f:
    values = [numpy.float32(random.random()) for _ in range(REQUIRED_SIZE)]
    float_array = array('f', values)
    float_array.tofile(f)

