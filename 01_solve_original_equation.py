import numpy as np
import math
import time


# Determine Parameters
L = 1000
N = 1000
T = 450
tau = 0.01
h = L / N
lt = int(T / tau)

lp = int(10)

phi = np.zeros((lt + 1, N + 1))


# Initial Conditions
d = int(N / 10)

phi_0 = np.zeros(N + 1)

for i in range(0, N + 1):
    phi_0[i] = 0.5 * (1 - math.tanh(i - d))

phi[0] = phi_0


# Compute phi^1

path_operator = "data/Operators/Global/"

P_1 = np.loadtxt(path_operator + "P_1.csv", delimiter=",")
P_2 = np.loadtxt(path_operator + "P_2.csv", delimiter=",")
P_3 = np.loadtxt(path_operator + "P_3.csv", delimiter=",")

phi_0_s = phi_0 ** 2

phi_1 = np.dot(P_1, phi_0) + np.dot(P_2, phi_0_s) + P_3
phi[1] = phi_1


# Compute phi^n

time_start = time.time()
print(time_start)

P_1_Prime = np.loadtxt(path_operator + "P_1_Prime.csv", delimiter=",")
P_2_Prime = np.loadtxt(path_operator + "P_2_Prime.csv", delimiter=",")
P_3_Prime = np.loadtxt(path_operator + "P_3_Prime.csv", delimiter=",")
P_4_Prime = np.loadtxt(path_operator + "P_4_Prime.csv", delimiter=",")
P_5_Prime = np.loadtxt(path_operator + "P_5_Prime.csv", delimiter=",")

period = int(lt / lp)

for i in range(2, lt + 1):

    phi_0_s = phi_0 ** 2
    phi_1_s = phi_1 ** 2

    phi_2 = (
        np.dot(P_1_Prime, phi_0)
        + np.dot(P_2_Prime, phi_0_s)
        + np.dot(P_3_Prime, phi_1)
        + np.dot(P_4_Prime, phi_1_s)
        + P_5_Prime
    )

    phi_0 = phi_1
    phi_1 = phi_2

    if i % period == 0:
        print(i)

    phi[i] = phi_2

time_end = time.time()

print(time_end)
print("time = %f" % (time_end - time_start))


# Save train data
path_solution = "data/Solution/Global/"
np.savetxt(path_solution + "phi.csv", phi, delimiter=",")
