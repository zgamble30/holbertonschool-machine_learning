#!/usr/bin/env python3
import numpy as np
from scipy import linalg

arr = np.arange(10)**2 % 7
k = 4
result = linalg.hankel(arr)[:arr.size-k+1, :k]
