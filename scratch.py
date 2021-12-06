import numpy as np

import pickle

# Call input file, read input
input_file = open('input.pkl', 'rb')
x = pickle.load(input_file)
input_file.close()

def f(x):
    res = np.sum(x ** 2, axis=1)[:, None]
    return res

# Prepare output file, write output
output_file = open('output.pkl', 'wb')
pickle.dump(f(x), output_file)
output_file.close()