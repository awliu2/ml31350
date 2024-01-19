import numpy as np
from numpy.linalg import inv

def calc_beta(X, Z, Y, omega):
    return inv(Z.T @ omega @ Z) @ Z.T @ omega @ Y

def one_sim(size, beta1):
    epsilon = np.random.normal(0, np.sqrt(np.arange(1, size + 1) ** 2), size=(size, size))
    X = np.random.normal(3, 1, size) + epsilon[0, :]
    Z = np.random.normal(3, 1, size * 2).reshape((size, 2))
    Y = beta1 * X + epsilon[:, 0]

    omega1 = np.eye(Z.shape[1])
    b_iv1 = calc_beta(X, Z, Y, omega1)

    omega2 = np.zeros((Z.shape[1], Z.shape[1]))
    for i in range(len(Y)):
        omega2 += np.outer(Z[i, :], Z[i, :])
    omega2 /= len(Y)
    b_iv2 = calc_beta(X, Z, Y, omega2)

    residual = Y - X @ b_iv2

    omega3 = np.zeros((Z.shape[1], Z.shape[1]))
    for i in range(len(Y)):
        omega3 += np.outer(Z[i, :], Z[i, :]) * (residual[i] ** 2)
    omega3 /= len(Y)
    b_iv3 = calc_beta(X, Z, Y, omega3)

    return np.concatenate([b_iv1, b_iv2, b_iv3])

# Example usage
size = 100
beta1 = 2
result = one_sim(size, beta1)
print(result)
