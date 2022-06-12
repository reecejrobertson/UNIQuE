import sys
sys.path.append('/fslhome/reecejr/software/QuantumComputingEmulator') # Change this to match your installation location.
sys.path.append('/fslhome/reecejr/software/intel-qs/build/lib') # Change this to match your installation location.
import intelqs_py as simulator
import emulator
import numpy as np
import scipy.sparse as sp
import time
from matplotlib import pyplot as plt

# Define the number of times to repeat each following experiment.
M = 10

# ---------------------------------------------------------------------------- #
#                            Arithmetic Sparse                                 #
# ---------------------------------------------------------------------------- #

# Set the maximum number of states to simulate.
N = 100

# Set the number of qubits and an array containing each possible state.
q = 10
numbers = np.arange(0, 2**q, 1)

# Create a list of various numbers of states <= N to simulate.
num_states = np.arange(5, N+1, 5)

# Define a list to hold the times of each operation.
add_times = []
mult_times = []
exp_times = []

# For each iteration of the experiment:
for m in range(M):
    
    print('m = ' + str(m) + '...', end='')
    
    # Define arrays to hold the results of this iteration (batch).
    add_batch = []
    mult_batch = []
    exp_batch = []

    # For each number of states:
    for n in num_states:

        # Define a random initial state.
        a = sp.dok_matrix((2**10, 1), dtype=complex)
        b = sp.dok_matrix((2**10, 1), dtype=complex)
        a_val = np.random.uniform(0, 1, n).astype(complex)
        b_val = np.random.uniform(0, 1, n).astype(complex)
        a_ind = np.random.choice(numbers, n, False)
        b_ind = np.random.choice(numbers, n, False)
        a[a_ind] = a_val
        b[b_ind] = b_val

        # Perform the addition with the emulator and time how long it takes.
        start_time = time.time()
        emulator.add_sparse(a, b)
        add_batch.append(time.time() - start_time)

        # Perform the multiplication with the emulator and time how long it takes.
        start_time = time.time()
        emulator.multiply_sparse(a, b)
        mult_batch.append(time.time() - start_time)

        # Perform the exponentiation with the emulator and time how long it takes.
        start_time = time.time()
        emulator.exponentiate_sparse(a, b)
        exp_batch.append(time.time() - start_time)

    # Append the batch results to the main array.
    add_times.append(add_batch)
    mult_times.append(mult_batch)
    exp_times.append(exp_batch)
    
    print('Done')

    # Average the times over each batch to get the average time for each operation.
    add_array = np.array(add_times)
    mult_array = np.array(mult_times)
    exp_array = np.array(exp_times)
    add_array = np.sum(add_times, axis=0)/m
    mult_array = np.sum(mult_times, axis=0)/m
    exp_array = np.sum(exp_times, axis=0)/m

    # Plot the times for each operation.
    plt.plot(num_states, add_times, 'o-b', label='Addition')
    plt.plot(num_states, mult_times, 'o-g', label='Multiplication')
    plt.plot(num_states, exp_times, 'o-c', label='Exponentiation')
    plt.xticks(np.arange(1, N+1, 1))
    plt.title('Emulator Speed for Sparse Arithmetic Operations')
    plt.xlabel('Number of Nonzero States')
    plt.ylabel('Time (seconds)')
    plt.legend(loc='best')
    plt.savefig('arithmetic.png', dpi=600)
    plt.show()

    # Plot the times for each operation on a log plot.
    plt.semilogy(num_states, add_times, 'o-b', label='Addition')
    plt.semilogy(num_states, mult_times, 'o-g', label='Multiplication')
    plt.semilogy(num_states, exp_times, 'o-c', label='Exponentiation')
    plt.xticks(np.arange(1, N+1, 1))
    plt.title('Emulator Speed for Sparse Arithmetic Operations on Log Plot')
    plt.xlabel('Number of Nonzero States')
    plt.ylabel('Time (seconds)')
    plt.legend(loc='best')
    plt.savefig('arithmetic_log.png', dpi=600)
    plt.show()
