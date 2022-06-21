import sys
sys.path.append('/fslhome/reecejr/software/QuantumComputingEmulator') # Change this to match your installation location.
sys.path.append('/fslhome/reecejr/software/intel-qs/build/lib') # Change this to match your installation location.
import intelqs_py as simulator
import emulator
import numpy as np
import time
from matplotlib import pyplot as plt

# Define the number of times to repeat each following experiment.
M = 10

# ---------------------------------------------------------------------------- #
#                                 Emulator QPE                                 #
# ---------------------------------------------------------------------------- #

# Set the number of qubits to use for each estimation.
S = 15

# Set the maximum size of matrix to simulate (ie the number of qubits for the matrix).
N = 20

# Create a list of various numbers of qubits <= N to simulate.
num_qubits = np.arange(2, N+1, 2)

# Define a list to hold the emulator and simulator times respectively.
em_times = []

# For each iteration of the experiment:
for m in range(1, M+1):
    
    print('m = ' + str(m) + '...', end='')
    
    # Define arrays to hold the results of this iteration (batch).
    em_batch = []

    # For each number of qubits:
    for n in num_qubits:
        
        # Define a matrix and an eigenvector.
        z = np.random.rand()
        U = np.eye(2**n)
        U[-1, -1] = z
        phi = np.zeros(2**n)
        phi[-1] = 1
        
        # Perform the QFT with the emulator and time how long it takes.
        start_time = time.time()
        emulator.qpe(U, phi, S)
        em_batch.append(time.time() - start_time)

    # Append the batch results to the main array.
    em_times.append(em_batch)
    
    print('Done')
      
    # Average the times over each batch to get the average time for each operation.
    em_array = np.array(em_times)
    em_array = np.sum(em_times, axis=0)/m

    # Plot the times for each QPE operation.
    fig = plt.figure()
    plt.plot(num_qubits, em_array, 'o-k')   
    plt.title('Emulator Speed for QPE')
    plt.xlabel('Number of Qubits in U Matrix')
    plt.ylabel('Time (seconds)')
    plt.savefig('Plots/em_qpe.png', dpi=600)

    # Plot the times for each QPE operation on a log plot.
    fig = plt.figure()
    plt.semilogy(num_qubits, em_array, 'o-k')   
    plt.title('Emulator Speed for QPE on Log Plot')
    plt.xlabel('Number of Qubits in U Matrix')
    plt.ylabel('Time (seconds)')
    plt.savefig('Plots/em_qpe_log.png', dpi=600)