import sys
sys.path.append('/fslhome/reecejr/software/QuantumComputingEmulator') # Change this to match your installation location.
sys.path.append('/fslhome/reecejr/software/intel-qs/build/lib') # Change this to match your installation location.
import intelqs_py as simulator
import emulator
import numpy as np
import time
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.sparse import dok_matrix

# Define the number of times to repeat each following experiment.
M = 10

# Define the number of qubits to simulate for each experiment.
min_qubit = 1
max_qubit = 10

# ---------------------------------------------------------------------------- #
#                                Multiplication                                #
# ---------------------------------------------------------------------------- #

# Create a list of various numbers of qubits <= N to simulate.
num_qubits = np.arange(min_qubit, max_qubit+1, 1)

# Define a list to hold the times of each operation.
add_times = []
mult_times = []
add_sparse_times = []
mult_sparse_times = []

# For each iteration of the experiment:
for m in range(1, M+1):
    
    print('m = ' + str(m) + '...', end='')
    
    # Define arrays to hold the results of this iteration (batch).
    add_batch = []
    mult_batch = []
    add_sparse_batch = []
    mult_sparse_batch = []

    # For each number of qubits:
    for n in num_qubits:

        # Define a random initial state.
        alpha = np.random.uniform(0, 1, 2**n).astype(complex)
        beta = np.random.uniform(0, 1, 2**n).astype(complex)
        
        a = dok_matrix((len(alpha), 1))
        for i, x in enumerate(alpha):
            a[i] = x

        b = dok_matrix((len(beta), 1))
        for i, x in enumerate(beta):
            b[i] = x
            
        # Perform the addition with the emulator and time how long it takes.
        start_time = time.time()
        emulator.add(alpha, beta)
        add_batch.append(time.time() - start_time)

        # Perform the multiplication with the emulator and time how long it takes.
        start_time = time.time()
        emulator.multiply(alpha, beta)
        mult_batch.append(time.time() - start_time)

        # Perform the addition with the emulator and time how long it takes.
        start_time = time.time()
        emulator.add_sparse(a, b)
        add_sparse_batch.append(time.time() - start_time)

        # Perform the multiplication with the emulator and time how long it takes.
        start_time = time.time()
        emulator.multiply_sparse(a, b)
        mult_sparse_batch.append(time.time() - start_time)
        
    # Append the batch results to the main array.
    add_times.append(add_batch)
    mult_times.append(mult_batch)
    add_sparse_times.append(add_sparse_batch)
    mult_sparse_times.append(mult_sparse_batch)
    
    print('Done')

print('----------')

# Average the times over each batch to get the average time for each operation.
add_array = np.array(add_times)
mult_array = np.array(mult_times)
add_sparse_array = np.array(add_sparse_times)
mult_sparse_array = np.array(mult_sparse_times)
add_array = np.sum(add_times, axis=0)/m
mult_array = np.sum(mult_times, axis=0)/m
add_sparse_array = np.sum(add_sparse_times, axis=0)/m
mult_sparse_array = np.sum(mult_sparse_times, axis=0)/m

# Record the raw data.
print("Addition data:", add_array)
print("Multiplication data:", mult_array)
print("Addition sparse data:", add_sparse_array)
print("Multiplication sparse data:", mult_sparse_array)
print('----------')

# Plot the times for each operation.
fig = plt.figure()
plt.plot(num_qubits, add_array, 'o-b', label='Standard Addition')
plt.plot(num_qubits, mult_array, 'o-g', label='Standard Multiplication')
plt.plot(num_qubits, add_sparse_array, 'o-r', label='Sparse Addition')
plt.plot(num_qubits, mult_sparse_array, 'o-', color='orange', label='Sparse Multiplication')
plt.xlabel('Number of Qubits per Operand')
plt.ylabel('Time (seconds)')
plt.legend(loc='best')
plt.savefig('Plots/add_mult_sparse.png', dpi=600)

# Plot the times for each operation.
fig = plt.figure()
plt.semilogy(num_qubits, add_array, 'o-b', label='Standard Addition')
plt.semilogy(num_qubits, mult_array, 'o-g', label='Standard Multiplication')
plt.semilogy(num_qubits, add_sparse_array, 'o-r', label='Sparse Addition')
plt.semilogy(num_qubits, mult_sparse_array, 'o-', color='orange', label='Sparse Multiplication')
plt.xlabel('Number of Qubits per Operand')
plt.ylabel('Time (seconds)')
plt.legend(loc='best')
plt.savefig('Plots/add_mult_log_sparse.png', dpi=600)