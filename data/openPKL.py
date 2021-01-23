import pickle
import numpy as np

with open('pdp/pdp20_TEST.pkl', 'rb') as f:
    data = pickle.load(f)
    print(data[0])
    dd = np.array(data)
    print(dd.shape)
    print(dd[0])
    print()
    print(dd)