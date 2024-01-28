import pennylane as qml
from pennylane.ops import CNOT, RX, RY, RZ, CZ
from pennylane import numpy as np
import random
import numpy.linalg as la
import math
from math import pi
from datetime import datetime
import os

seed = 0
np.random.seed(seed)

def f_n(weights, ansatz=None):

    return np.sum(ansatz(weights))

def gd_optimizer(ansatz, weights, noise_gamma, lr, iteration, n_check):
    grad_func = qml.grad(f_n)
    n_params = len(weights)
    grad_norm = np.zeros(iteration)
    loss = np.zeros(iteration)
    gradient_list = np.zeros((iteration, n_params))
    t1 = datetime.now()
    for j in range(iteration):
        weights.requires_grad = True
        noise = np.random.normal(0,noise_gamma, n_params)
        #print(qml.draw(ansatz)(weights), weights[-10:])
        loss[j] = f_n(weights, ansatz=ansatz)
        gradient_list[j] = grad_func(weights, ansatz=ansatz)
        grad_norm[j] = la.norm(gradient_list[j])
        gradient = gradient_list[j] + noise
        weights = weights - lr * gradient
        if(j%n_check==0):
            t2=datetime.now()
            print( j, " loss : %.6f" % loss[j], "gradnorm: %.6f" % grad_norm[j], "noise: %.6f" % la.norm(noise), "time: ", (t2-t1).seconds)
            t1=t2
    return loss, grad_norm, weights, gradient_list

def gd_gaussian(ansatz, circuit_name, qubits, n_params, gamma, gau_rate, noise_gamma, noise_rate, lr, iteration, n_time, n_check):
    

    folder = "./"+circuit_name +"_"+ str(qubits) +"_"+ str(n_params)+"_gd_gaussian"+"_"+str(gau_rate)+"_"+str(noise_rate)
    a = np.random.normal(0, gamma*gau_rate, n_params*n_time).reshape(n_time,n_params)
    print("Quantum Circuit:")
    #print(qml.draw(ansatz)(a[0]))
    if not os.path.exists(folder):
        os.makedirs(folder)
    for time in range(n_time):
        np.save(folder+"/weights_init_"+str(time)+".npy", a[time])
        loss, grad_norm, weights, gradient = gd_optimizer(ansatz, a[time], noise_gamma*noise_rate, lr, iteration, n_check)
        np.save(folder+"/loss_"+str(time)+".npy", loss)
        np.save(folder+"/grad_norm_"+str(time)+".npy",  grad_norm)
        np.save(folder+"/weights_final_"+str(time)+".npy", weights)
        np.save(folder+"/gradient_"+str(time)+".npy", gradient)
        print("the training with gd gaussian ends", gau_rate, noise_rate)
    return 0

def gd_zero(ansatz, circuit_name, qubits, n_params, gamma, gau_rate, noise_gamma, noise_rate, lr, iteration, n_time, n_check):
    

    folder = "./"+circuit_name +"_"+ str(qubits) +"_"+ str(n_params)+"_gd_zero"+"_"+str(noise_rate)
    a = np.random.normal(0, 0, n_params*n_time).reshape(n_time,n_params)
    if not os.path.exists(folder):
        os.makedirs(folder)
    for time in range(n_time):
        np.save(folder+"/weights_init_"+str(time)+".npy", a[time])
        loss, grad_norm, weights, gradient = gd_optimizer(ansatz, a[time], noise_gamma*noise_rate, lr, iteration, n_check)
        np.save(folder+"/loss_"+str(time)+".npy", loss)
        np.save(folder+"/grad_norm_"+str(time)+".npy",  grad_norm)
        np.save(folder+"/weights_final_"+str(time)+".npy", weights)
        np.save(folder+"/gradient_"+str(time)+".npy", gradient)
        print("the training with gd zero ends", noise_rate)
    return 0

def gd_uniform(ansatz, circuit_name, qubits, n_params, gamma, gau_rate, noise_gamma, noise_rate, lr, iteration, n_time, n_check):

    folder = "./"+circuit_name +"_"+ str(qubits) +"_"+ str(n_params)+"_gd_uniform"+"_"+str(noise_rate)

    a = np.random.uniform(-pi, pi, size=n_time*n_params).reshape(n_time,n_params)
    if not os.path.exists(folder):
        os.makedirs(folder)
    for time in range(n_time):
        np.save(folder+"/weights_init_"+str(time)+".npy", a[time])
        loss, grad_norm, weights, gradient = gd_optimizer(ansatz, a[time], noise_gamma*noise_rate, lr, iteration, n_check)
        np.save(folder+"/loss_"+str(time)+".npy", loss)
        np.save(folder+"/grad_norm_"+str(time)+".npy",  grad_norm)
        np.save(folder+"/weights_final_"+str(time)+".npy", weights)
        np.save(folder+"/gradient_"+str(time)+".npy", gradient)
        print("the training with gd uniform ends", noise_rate)
    return 0

def gd_reduced(ansatz, circuit_name, qubits, n_params, gamma, gau_rate, noise_gamma, noise_rate, lr, iteration, n_time, n_check):

    folder = "./"+circuit_name +"_"+ str(qubits) +"_"+ str(n_params)+"_gd_uniform"+"_"+str(noise_rate)

    a = np.random.uniform(-0.07*pi, 0.07*pi, size=n_time*n_params).reshape(n_time,n_params)
    if not os.path.exists(folder):
        os.makedirs(folder)
    for time in range(n_time):
        np.save(folder+"/weights_init_"+str(time)+".npy", a[time])
        loss, grad_norm, weights, gradient = gd_optimizer(ansatz, a[time], noise_gamma*noise_rate, lr, iteration, n_check)
        np.save(folder+"/loss_"+str(time)+".npy", loss)
        np.save(folder+"/grad_norm_"+str(time)+".npy",  grad_norm)
        np.save(folder+"/weights_final_"+str(time)+".npy", weights)
        np.save(folder+"/gradient_"+str(time)+".npy", gradient)
        print("the training with gd uniform ends", noise_rate)
    return 0

def initialize_parameters(num_blocks, arrays, s):
    n = len(arrays)
    parameters = []
    std = 1/np.sqrt(2*num_blocks*s)

    for i in range(n):
        if arrays[i] == 0:
            # If the i-th position of the array is 0, initialize with N(0, 1)
            parameter = np.random.normal(0, std)

        elif arrays[i] == 1:

            # Parameters for the two normal distributions
            mean1 = -np.pi/2
            weight1 = 0.5

            mean2 = np.pi/2
            weight2 = 0.5  

            # If it is 1, initialize with the Gaussian mixture distribution
            component_selection = np.random.choice([0, 1], p=[weight1, weight2])
            if component_selection == 0:
                parameter = np.random.normal(mean1, std)
            else:
                parameter = np.random.normal(mean2, std)
        
        else:
            # Parameters for the three normal distributions
            mean1 = -np.pi
            weight1 = 0.25

            mean2 = np.pi
            weight2 = 0.25  

            mean3 = 0
            weight3 = 0.5  
            
            # If it is 1, initialize with the Gaussian mixture distribution
            component_selection = np.random.choice([0, 1, 2], p=[weight1, weight2, weight3])
            if component_selection == 0:
                parameter = np.random.normal(mean1, std)
            elif component_selection == 1:
                parameter = np.random.normal(mean2, std)
            else:
                parameter = np.random.normal(mean3, std)

        parameters.append(parameter)
    # Convert the list to a NumPy array for convenient element-wise operations
    my_array = np.array(parameters)

    # Apply the transformation to keep values within the range (-pi, pi]
    my_array = (my_array + np.pi) % (2 * np.pi) - np.pi

    # Convert the array back to a list if needed
    parameters = my_array.tolist()

    return parameters

def create_new_array1(nqubits, random_array):
        # Map the elements according to the specified rules
        mapped_array = np.zeros(2 * nqubits, dtype=int)

        for i in range(nqubits):
            if random_array[i] == 0 or random_array[i] == 3:
                mapped_array[i] = 0
                mapped_array[i + nqubits] = 0
            elif random_array[i] == 1:
                mapped_array[i] = 0
                mapped_array[i + nqubits] = 1
            elif random_array[i] == 2:
                mapped_array[i] = 1
                mapped_array[i + nqubits] = 0

        #mapped_array = np.ones(2 * nqubits, dtype=int)

        return mapped_array

def gd_ours(num_blocks, ansatz, circuit_name, qubits, n_params, gamma, gau_rate, noise_gamma, noise_rate, lr, iteration, n_time, n_check, model):
    
    folder = "./" + circuit_name + "_" + str(qubits) + "_" + str(n_params) + "_gd_ours" + "_" + str(
        gau_rate) + "_" + str(noise_rate)
    if model == 'chemistry':
        random_array = np.ones(qubits) * 3
        random_array[0] = 2
        random_array[8] = 2
    elif model == 'Heisenberg':
        random_array = np.zeros(qubits)
        random_array[0] = 2
        random_array[1] = 2
    elif model == 'All-X':
        random_array = np.ones(qubits)
    elif model == 'Ising':
        random_array = np.zeros(qubits)
        random_array[0] = 1

    # Calculate the number of non-zero elements
    non_zero_count = np.count_nonzero(random_array)
    # Example: Generate n arrays of 0s and 1s
    arrays1 = create_new_array1(qubits, random_array)
    #print(arrays1)
    if not os.path.exists(folder):
        os.makedirs(folder)

    for time in range(n_time):
        a = np.random.normal(0, gamma*gau_rate, n_params - 2*qubits)

        # Example parameters initialization
        params_last_layer = initialize_parameters(num_blocks, arrays1, non_zero_count)
        
        # Concatenate the two sets of parameters
        a = np.concatenate([a, params_last_layer])

        np.save(folder+"/weights_init_"+str(time)+".npy", a)
        loss, grad_norm, weights, gradient = gd_optimizer(ansatz, a, noise_gamma*noise_rate, lr, iteration, n_check)
        np.save(folder+"/loss_"+str(time)+".npy", loss)
        np.save(folder+"/grad_norm_"+str(time)+".npy",  grad_norm)
        np.save(folder+"/weights_final_"+str(time)+".npy", weights)
        np.save(folder+"/gradient_"+str(time)+".npy", gradient)
        print("the training with gd ours ends", noise_rate)
    return 0


