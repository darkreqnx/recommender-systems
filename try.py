import numpy as np

np.random.seed(0)
arr = np.zeros((3,3))
s_arr = np.random.random_integers(0, 10, (5,5))

for i in range(3):
    arr[i] = s_arr[2, [0,2,3]]
print(arr)
print(s_arr)