import numpy as np
import os


# Load original data
path_train = "data/train_data/"
path_data = "data/Solution/Global/"

phi = np.loadtxt(path_data + "phi.csv", delimiter=",")


# Save phi_in
NL, NR = 0, 400
L = NR - NL + 1
start = 2

phi_in = phi[start - 1 : phi.shape[0] - 1, NL : NR + 1]
print(phi_in.shape)
np.savetxt(path_train + "phi_in.csv", phi_in, delimiter=",")


# Get and Save P5Prime (E)
for i in range(NL, NR):
    phi_temp = phi.T[NL : NR + 1, :]

phi_p = phi_temp.T  # input for the predictor

N = phi_p.shape[0]

path_operator = "data/Operators/Local/"

P_1_Prime = np.loadtxt(path_operator + "P_1_Prime.csv", delimiter=",")
P_2_Prime = np.loadtxt(path_operator + "P_2_Prime.csv", delimiter=",")
P_3_Prime = np.loadtxt(path_operator + "P_3_Prime.csv", delimiter=",")
P_4_Prime = np.loadtxt(path_operator + "P_4_Prime.csv", delimiter=",")

print(phi_p.shape)

E = np.zeros((N - (start + 1), L))

for i in range(start + 1, N):

    E[i - (start + 1)] = phi_p[i] - (
        np.dot(P_1_Prime, phi_p[i - 2])
        + np.dot(P_2_Prime, phi_p[i - 2] ** 2)
        + np.dot(P_3_Prime, phi_p[i - 1])
        + np.dot(P_4_Prime, phi_p[i - 1] ** 2)
    )


np.savetxt(path_train + "E.csv", E, delimiter=",")

print(E.shape)


# get segment info
L = phi_in.shape[0] - 1
r = L % 5
l_total = L - r
num_piece = int(l_total / 5)

l_test = num_piece
l_train = int(4 * num_piece + r)


# save train data
output_path_train = "data/train_data/train/"
output_path_test = "data/train_data/test/"
if not os.path.exists(output_path_train):
    os.makedirs(output_path_train)
if not os.path.exists(output_path_test):
    os.makedirs(output_path_test)

phi_train, E_train = (
    np.zeros((l_train, phi_in.shape[1])),
    np.zeros((l_train, E.shape[1])),
)
phi_test, E_test = np.zeros((l_test, phi_in.shape[1])), np.zeros((l_test, E.shape[1]))

k_train, k_test = 0, 0

for i in range(L):
    if i % 5 < 4:  # train data
        phi_train[k_train] = phi_in[i]
        E_train[k_train] = E[i]
        k_train += 1
    else:
        phi_test[k_test] = phi_in[i]
        E_test[k_test] = E[i]
        k_test += 1

print(l_train, k_train)
print(l_test, k_test)
np.savetxt(output_path_train + "phi_train.csv", phi_train, delimiter=",")
np.savetxt(output_path_train + "E_train.csv", E_train, delimiter=",")
np.savetxt(output_path_test + "phi_test.csv", phi_test, delimiter=",")
np.savetxt(output_path_test + "E_test.csv", E_test, delimiter=",")
