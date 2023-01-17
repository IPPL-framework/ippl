import numpy as np
import random


with open("cholesky.csv", "w") as f:

    random.seed(0)

    for i in range(-10, 10, 1):
        for j in range(10):

            A = np.random.rand(3, 3).astype(np.float64)
            B = np.dot(A, A.transpose()) #semi positive definite
            B = pow(10, i)*B
            L = np.linalg.cholesky(B)

            np.savetxt(f, np.reshape(B, (1,9)), fmt='%e', delimiter=' ')
            np.savetxt(f, np.reshape(L, (1,9)), fmt='%e', delimiter=' ')