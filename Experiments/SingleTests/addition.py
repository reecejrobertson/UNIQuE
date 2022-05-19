import sys
# sys.path.append('/fslhome/reecejr/software/QuantumComputingEmulator') # Change this to match your installation location.
# sys.path.append('/fslhome/reecejr/software/intel-qs/build/lib') # Change this to match your installation location.
import intelqs_py as simulator
import emulator
import numpy as np
import time
from matplotlib import pyplot as plt

# Define the number of times to repeat each following experiment.
M = 10

# Define the number of qubits to simulate for each experiment.
N_ADDITION = 14

# ---------------------------------------------------------------------------- #
#                                   Addition                                   #
# ---------------------------------------------------------------------------- #

# Set the maximum number of qubits to simulate.
N = N_ADDITION

# Create a list of various numbers of qubits <= N to simulate.
num_qubits = np.arange(2, N+1, 2)

# Define a list to hold the emulator and simulator times respectively.
em_times = []
sim_times = []

# For each iteration of the experiment:
for m in range(1, M+1):

    print('m = ' + str(m) + '...')
        
    # Define arrays to hold the results of this iteration (batch).
    em_batch = []
    sim_batch = []

    # For each n, we will add two numbers of size n, as follows:
    for n in num_qubits:
        
        # Initialize the state of the simulator.
        # Note that to add two numbers of size n, we need 2n + 2 qubits.
        psi = simulator.QubitRegister(2*n+2, 'base', 0, 0)
        for i in range(1, 2*n+1):
            psi.ApplyPauliX(i)
        
        # Initialize the state of the emulator.
        # Note that every qubit is set to 1 for both the emulator and simulator.
        a = np.zeros(2**n, dtype=complex)
        a[-1] = 1
        
        # Perform addition with the emulator and time how long it takes.
        start_time = time.time()
        em_state = emulator.add(a, a)
        em_batch.append(time.time() - start_time)

        # Perform the QFT with the simulator and time how long it takes.
        start_time = time.time()
        for i in range(0, 2*n, 2):
            psi.ApplyCPauliX(i+2, i+1)
            psi.ApplyCPauliX(i+2, i)
            psi.ApplyToffoli(i, i+1, i+2)
        psi.ApplyCPauliX(2*n, 2*n+1)
        for i in range(2*n-2, -2, -2):
            psi.ApplyToffoli(i, i+1, i+2)
            psi.ApplyCPauliX(i+2, i)
            psi.ApplyCPauliX(i, i+1)
        sim_batch.append(time.time() - start_time)
        
        # Extract the sum from the simulator state.
        sim_state = []
        for i in range(2**(2*n+2)):
            sim_state.append(psi[i])
        sim_state = np.array(sim_state, dtype=complex)
        sim_indx = bin(sim_state.nonzero()[0][0])
        sim_sum = sim_indx[2::2]

        # Extract the sum from the emulator state.
        em_indx = bin(em_state.nonzero()[0][0])
        em_sum = em_indx[2::1]

        # Ensure that the final states of the emulator and simulator agree.
        if not em_sum == sim_sum:
            print('Output differed for ' + str(n) + ' qubits.')
            print(sim_sum)
            print(em_sum)
            print()

    # Append the batch results to the main array.     
    em_times.append(em_batch)
    sim_times.append(sim_batch)
    
    print('Done')

    # Average the times over each batch.
    em_array = np.array(em_times)
    sim_array = np.array(sim_times)
    em_array = np.sum(em_times, axis=0)/m
    sim_array = np.sum(sim_times, axis=0)/m

    # Plot the times for each addition operation.
    fig = plt.figure()
    plt.plot(num_qubits, em_array, 'o-k', label='Emulator')   
    plt.plot(num_qubits, sim_array, 'o-r', label='Simulator')
    plt.title('Speed Comparison for Addition')
    plt.xlabel('Number of Qubits')
    plt.ylabel('Time (seconds)')
    plt.legend(loc='best')
    plt.savefig('Plots/addition.png', dpi=600)

    # Plot the times for each addition operation on a log plot.
    fig = plt.figure()
    plt.semilogy(num_qubits, em_array, 'o-k', label='Emulator')   
    plt.semilogy(num_qubits, sim_array, 'o-r', label='Simulator')
    plt.title('Speed Comparison for Addition on Log Plot')
    plt.xlabel('Number of Qubits')
    plt.ylabel('Time (seconds)')
    plt.legend(loc='best')
    plt.savefig('Plots/addition_log.png', dpi=600)