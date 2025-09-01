import pennylane as qml
from pennylane.ops import CNOT, RX, RY, RZ, CZ
from pennylane import numpy as np
import random
import numpy.linalg as la
import math
from math import pi
from datetime import datetime
import os
import test_optimizer as optimizer
import time

# this code is referred from https://openreview.net/forum?id=jXgbJdQ2YIy
# Parameters
n_qubits = 20
n_layer = 8
n_time = 1
n_check = 20
iteration = 300
model = 'All-X'

print('model:', model)

if model == 'chemistry':
    mole, symbols = "LiH", ["Li", "H"]
    geometry = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 2.969280527])
    n_e, n_elec, n_orbitals = 0, 2, 5
    H, n_qubits = qml.qchem.molecular_hamiltonian(symbols, geometry, active_electrons=n_elec, active_orbitals=n_orbitals, charge=n_e)

print("Number of qubits = ", n_qubits)

dev = qml.device("default.qubit", wires=n_qubits)
@qml.qnode(dev, interface="autograd", diff_method="backprop")

def cl_circuit(params):
    k = 0
    for i in range(n_layer):
        for j in range(n_qubits):
            qml.CZ(wires=[j,(j+1)%n_qubits])
        for j in range(n_qubits):
            qml.RX(params[k], wires=j)
            k = k + 1
        for j in range(n_qubits):
            qml.RY(params[k], wires=j)
            k = k + 1
    if model == 'Heisenberg':
        #print([qml.expval(qml.PauliX(n) @ qml.PauliX((n+1)%n_qubits) + qml.PauliY(n) @ qml.PauliY((n+1)%n_qubits) + qml.PauliZ(n) @ qml.PauliZ((n+1)%n_qubits)) for n in range(n_qubits-1)])
        return [qml.expval(qml.PauliX(n) @ qml.PauliX((n+1)%n_qubits) + qml.PauliY(n) @ qml.PauliY((n+1)%n_qubits) + qml.PauliZ(n) @ qml.PauliZ((n+1)%n_qubits)) for n in range(n_qubits-1)]
    
    elif model == 'chemistry':
        return qml.expval(H)
    
    elif model == 'Ising':
        hamiltonian_terms_1 = [qml.PauliZ(n) @ qml.PauliZ((n+1)%n_qubits) for n in range(n_qubits-1)]
        hamiltonian_terms_2 = [qml.PauliX(n) for n in range(n_qubits)]
        hamiltonian_terms = hamiltonian_terms_1 + hamiltonian_terms_2

        coeff = [1] * len(hamiltonian_terms_1) + [-1] * len(hamiltonian_terms_2)
        H = qml.Hamiltonian(coeff, hamiltonian_terms)

        return qml.expval(H)

    elif model == 'All-X':
        pro = qml.PauliX(0)
        for i in range(1,n_qubits):
            pro = pro @ qml.PauliX(i)
        return [qml.expval(pro)]

    else:
        raise Exception('Error!')

t1 = datetime.now()
circuit_name = "xxx"
gamma = 1/(8*n_layer)**0.5
gamma_ours = 1/(2*n_layer*n_layer)**0.5
n_params = 2*n_layer*n_qubits


#noiseless
optimizer.gd_ours(n_layer, cl_circuit, circuit_name, n_qubits, n_params, gamma, 0.01, 0, 0, 0.01, iteration, n_time, n_check, model)
optimizer.gd_ours(n_layer, cl_circuit, circuit_name, n_qubits, n_params, gamma, 0.1, 0, 0, 0.01, iteration, n_time, n_check)
optimizer.gd_ours(n_layer, cl_circuit, circuit_name, n_qubits, n_params, gamma, 1, 0, 0, 0.01, iteration, n_time, n_check)
optimizer.gd_ours(n_layer, cl_circuit, circuit_name, n_qubits, n_params, gamma, 10, 0, 0, 0.01, iteration, n_time, n_check)
optimizer.gd_ours(n_layer, cl_circuit, circuit_name, n_qubits, n_params, gamma, 100, 0, 0, 0.01, iteration, n_time, n_check)


optimizer.gd_gaussian(cl_circuit, circuit_name, n_qubits, n_params, gamma, 0.01, 0, 0, 0.01, iteration, n_time, n_check)
optimizer.gd_gaussian(cl_circuit, circuit_name, n_qubits, n_params, gamma, 0.1, 0, 0, 0.01, iteration, n_time, n_check)
optimizer.gd_gaussian(cl_circuit, circuit_name, n_qubits, n_params, gamma, 1, 0, 0, 0.01, iteration, n_time, n_check)
optimizer.gd_gaussian(cl_circuit, circuit_name, n_qubits, n_params, gamma, 10, 0, 0, 0.01, iteration, n_time, n_check)
optimizer.gd_gaussian(cl_circuit, circuit_name, n_qubits, n_params, gamma, 100, 0, 0, 0.01, iteration, n_time, n_check)
optimizer.gd_zero(cl_circuit, circuit_name, n_qubits, n_params, gamma, 1, 0, 0, 0.01, iteration, n_time, n_check)
optimizer.gd_uniform(cl_circuit, circuit_name, n_qubits, n_params, gamma, 1, 0, 0, 0.01, iteration, n_time, n_check)
optimizer.gd_reduced(cl_circuit, circuit_name, n_qubits, n_params, gamma, 1, 0, 0, 0.01, iteration, n_time, n_check)
'''
#noisy
optimizer.gd_ours(n_layer,cl_circuit, circuit_name, n_qubits, n_params, gamma, 0.01, 1/(100)**0.5, 1, 0.01, iteration, n_time, n_check)
optimizer.gd_ours(n_layer,cl_circuit, circuit_name, n_qubits, n_params, gamma, 0.1, 1/(100)**0.5, 1, 0.01, iteration, n_time, n_check)
optimizer.gd_ours(n_layer,cl_circuit, circuit_name, n_qubits, n_params, gamma, 1, 1/(100)**0.5, 1, 0.01, iteration, n_time, n_check)
optimizer.gd_ours(n_layer,cl_circuit, circuit_name, n_qubits, n_params, gamma, 10, 1/(100)**0.5, 1, 0.01, iteration, n_time, n_check)
optimizer.gd_ours(n_layer,cl_circuit, circuit_name, n_qubits, n_params, gamma, 100, 1/(100)**0.5, 1, 0.01, iteration, n_time, n_check)

optimizer.gd_gaussian(cl_circuit, circuit_name, n_qubits, n_params, gamma, 0.01, 1/(100)**0.5, 1, 0.01, iteration, n_time, n_check)
optimizer.gd_gaussian(cl_circuit, circuit_name, n_qubits, n_params, gamma, 0.1, 1/(100)**0.5, 1, 0.01, iteration, n_time, n_check)
optimizer.gd_gaussian(cl_circuit, circuit_name, n_qubits, n_params, gamma, 1, 1/(100)**0.5, 1, 0.01, iteration, n_time, n_check)
optimizer.gd_gaussian(cl_circuit, circuit_name, n_qubits, n_params, gamma, 10, 1/(100)**0.5, 1, 0.01, iteration, n_time, n_check)
optimizer.gd_gaussian(cl_circuit, circuit_name, n_qubits, n_params, gamma, 100, 1/(100)**0.5, 1, 0.01, iteration, n_time, n_check)
optimizer.gd_zero(cl_circuit, circuit_name, n_qubits, n_params, gamma, 1, 1/(100)**0.5, 1, 0.01, iteration, n_time, n_check)
optimizer.gd_uniform(cl_circuit, circuit_name, n_qubits, n_params, gamma, 1, 1/(100)**0.5, 1, 0.01, iteration, n_time, n_check)
'''


t2 = datetime.now()
print("qubits: ", n_qubits, "layers: ", n_layer, 'seconds for ', circuit_name,  (t2-t1).seconds)

