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

# Define the number of qubits to simulate for each experiment.
N_EM_QPE = 40

# ---------------------------------------------------------------------------- #
#                                 Emulator QPE                                 #
# ---------------------------------------------------------------------------- #

# Perform this experiment 10 times the amount of the others.
M_hat = 10*M

# Set the maximum number of qubits to simulate.
N = N_EM_QPE

# Define a matrix (the T gate) and an eigenvector.
T = np.array([[1, 0], [0, np.exp(1j*np.pi/4)]])
phi = np.array([0, 1])

# Create a list of various numbers of qubits <= N to simulate.
num_qubits = np.arange(2, N+1, 2)

# Define a list to hold the emulator and simulator times respectively.
em_times = []

# For each iteration of the experiment:
for m in range(1, M_hat+1):
    
    print('m = ' + str(m) + '...', end='')
    
    # Define arrays to hold the results of this iteration (batch).
    em_batch = []

    # For each number of qubits:
    for n in num_qubits:
        
        # Perform the QFT with the emulator and time how long it takes.
        start_time = time.time()
        emulator.qpe(T, phi, n)
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
    plt.xlabel('Number of Qubits')
    plt.ylabel('Time (seconds)')
    plt.savefig('Plots/em_qpe.png', dpi=600)

    # Plot the times for each QPE operation on a log plot.
    fig = plt.figure()
    plt.semilogy(num_qubits, em_array, 'o-k')   
    plt.title('Emulator Speed for QPE on Log Plot')
    plt.xlabel('Number of Qubits')
    plt.ylabel('Time (seconds)')
    plt.savefig('Plots/em_qpe_log.png', dpi=600)