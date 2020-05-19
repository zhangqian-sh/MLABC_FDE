import numpy as np
from scipy.special import gamma
import math
import configparser
import os

# Parameters

config = configparser.ConfigParser()
config.read("config.ini")

alpha = float(config.get("Constant", "alpha"))
sigma = float(config.get("Constant", "sigma"))

L = int(config.get("Scheme Settings", "L"))
N = int(config.get("Scheme Settings", "N"))
tau = float(config.get("Scheme Settings", "tau"))
tau_s = tau * tau
h = L / N

BC_L = float(config.get("Boundary Conditions", "BC_L"))
BC_R = float(config.get("Boundary Conditions", "BC_R"))

print(BC_L, BC_R)

BC = np.zeros(2 * N + 1)

for i in range(0, N):
    BC[i] = BC_L
for i in range(-N, 0):
    BC[i] = BC_R

p = 2

# Generate Discrete Matrix A & Right Hand Item b


def Gamma_Multiply(x, y):
    a = max([x, y])
    b = min([x, y])
    Nmax = int(min([abs(a), abs(b)]))

    z = 1
    for p in range(1, Nmax + 1):
        z *= (a - 1) / b
        a -= 1
        b += 1

    return z * gamma(a) * gamma(b)


# Generate C, c, B, b, q

B, b = 1, 1

C_1, c_1 = np.zeros(2 * N + 1), np.zeros(2 * N + 1)
beta = float(1.0)

for k in range(-N, N + 1):
    C_1[k] = -(
        gamma(beta + 1) * math.sin(beta * math.pi / 2) * math.sin(k * math.pi / 2)
    ) / (2 ** beta * Gamma_Multiply(k / 2 + beta / 2 + 1, -k / 2 + beta / 2 + 1))
    c_1[k] = (
        (-1) ** k * math.cos(beta * math.pi / 2) * gamma(beta + 1)
    ) / Gamma_Multiply(beta / 2 - k + 1, beta / 2 + k + 1)

q_1 = np.zeros(2 * N + 1)

for m in range(-N, N + 1):
    q_1[m] = b * 2 * c_1[m] + B * 2 * C_1[m]

C_alpha, c_alpha = np.zeros(2 * N + 1), np.zeros(2 * N + 1)
beta = 1.0 + alpha

for k in range(-N, N + 1):
    C_alpha[k] = -(
        gamma(beta + 1) * math.sin(beta * math.pi / 2) * math.sin(k * math.pi / 2)
    ) / (2 ** beta * Gamma_Multiply(k / 2 + beta / 2 + 1, -k / 2 + beta / 2 + 1))
    c_alpha[k] = (
        (-1) ** k * math.cos(beta * math.pi / 2) * gamma(beta + 1)
    ) / Gamma_Multiply(beta / 2 - k + 1, beta / 2 + k + 1)

q_alpha = np.zeros(2 * N + 1)

for m in range(-N, N + 1):
    q_alpha[m] = b * 2 * c_alpha[m] + B * 2 * C_alpha[m]

C_3, c_3 = np.zeros(2 * N + 1), np.zeros(2 * N + 1)
beta = 3

for k in range(-N, N + 1):
    C_3[k] = -(
        gamma(beta + 1) * math.sin(beta * math.pi / 2) * math.sin(k * math.pi / 2)
    ) / (2 ** beta * Gamma_Multiply(k / 2 + beta / 2 + 1, -k / 2 + beta / 2 + 1))
    c_3[k] = (
        (-1) ** k * math.cos(beta * math.pi / 2) * gamma(beta + 1)
    ) / Gamma_Multiply(beta / 2 - k + 1, beta / 2 + k + 1)

q_3 = np.zeros(2 * N + 1)

for m in range(-N, N + 1):
    q_3[m] = b * 2 * c_3[m] + B * 2 * C_3[m]

# Generate Operator Matrices and Error Vectors

D_3 = np.zeros((N + 1, N + 1))

for i in range(0, N + 1):
    for j in range(0, N + 1):
        D_3[i][j] = q_3[i - j]

D_3 /= 2 * h ** 3

D_alpha = np.zeros((N + 1, N + 1))

for i in range(0, N + 1):
    for j in range(0, N + 1):
        D_alpha[i][j] = q_alpha[i - j]

D_alpha /= 2 * h ** (1 + alpha)

D_1 = np.zeros((N + 1, N + 1))

for i in range(0, N + 1):
    for j in range(0, N + 1):
        D_1[i][j] = q_1[i - j]

D_1 /= 2 * h

E_3 = np.zeros(N + 1)

for n in range(0, N + 1):
    for m in range(-N, n - N):
        E_3[n] += q_3[m] * BC_R
    for m in range(n + 1, N + 1):
        E_3[n] += q_3[m] * BC_L

E_3 /= 2 * h ** 3

E_alpha = np.zeros(N + 1)

for n in range(0, N + 1):
    for m in range(-N, n - N):
        E_alpha[n] += q_alpha[m] * BC_R
    for m in range(n + 1, N + 1):
        E_alpha[n] += q_alpha[m] * BC_L

E_alpha /= 2 * h ** (1 + alpha)

E_1 = np.zeros(N + 1)

for n in range(0, N + 1):
    for m in range(-N, n - N):
        E_1[n] += q_1[m] * BC_R
    for m in range(n + 1, N + 1):
        E_1[n] += q_1[m] * BC_L

E_1 /= 2 * h

# Compose Matrices and Vectors for Iteration Scheme

output_path = "../data/Operators/Global/"
if not os.path.exists(output_path):
    os.makedirs(output_path)

I = np.eye(N + 1)

A = I - (tau * sigma) * D_3
A_Prime = 1.5 * np.eye(N + 1) - (tau * sigma) * D_3

A_inv = np.linalg.inv(A)
A_Prime_inv = np.linalg.inv(A_Prime)

B_1 = I + tau * D_alpha
B_2 = -tau * D_1
B_3 = (tau * sigma) * E_3 + tau * E_alpha - tau * E_1

P_1 = np.dot(A_inv, B_1)
P_2 = np.dot(A_inv, B_2)
P_3 = np.dot(A_inv, B_3)

np.savetxt(output_path + "P_1.csv", P_1, delimiter=",")
np.savetxt(output_path + "P_2.csv", P_2, delimiter=",")
np.savetxt(output_path + "P_3.csv", P_3, delimiter=",")

B_1_Prime = -0.5 * I - tau * D_alpha
B_2_Prime = tau * D_1
B_3_Prime = 2 * I + 2 * tau * D_alpha
B_4_Prime = -2 * tau * D_1
B_5_Prime = (tau * sigma) * E_3 + tau * E_alpha - tau * E_1

P_1_Prime = np.dot(A_Prime_inv, B_1_Prime)
P_2_Prime = np.dot(A_Prime_inv, B_2_Prime)
P_3_Prime = np.dot(A_Prime_inv, B_3_Prime)
P_4_Prime = np.dot(A_Prime_inv, B_4_Prime)
P_5_Prime = np.dot(A_Prime_inv, B_5_Prime)

np.savetxt(output_path + "P_1_Prime.csv", P_1_Prime, delimiter=",")
np.savetxt(output_path + "P_2_Prime.csv", P_2_Prime, delimiter=",")
np.savetxt(output_path + "P_3_Prime.csv", P_3_Prime, delimiter=",")
np.savetxt(output_path + "P_4_Prime.csv", P_4_Prime, delimiter=",")
np.savetxt(output_path + "P_5_Prime.csv", P_5_Prime, delimiter=",")
