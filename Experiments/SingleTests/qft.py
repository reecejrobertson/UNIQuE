import sys
sys.path.append('/fslhome/reecejr/software/QuantumComputingEmulator')
sys.path.append('/fslhome/reecejr/software/intel-qs/build/lib')
import intelqs_py as simulator
import emulator
import numpy as np
import time
from matplotlib import pyplot as plt

# Define the number of times to repeat each following experiment.
M = 10

# Define the number of qubits to simulate for each experiment.
N_QFT = 35

# ---------------------------------------------------------------------------- #
#                                      QFT                                     #
# ---------------------------------------------------------------------------- #

# Set the maximum number of qubits to simulate.
N = N_QFT

# Define the z-rotations needed for the simulation method.
Z = []
for n in range(2, N+1):
   Z.append(np.array([[1, 0], [0, np.exp(1j*2*np.pi/2**n)]], dtype=complex)) 

# Create a list of various numbers of qubits <= N to simulate.
num_qubits = np.arange(2, N+1, 2)

# Define a list to hold the emulator and simulator times respectively.
em_times = []
sim_times = []

# For each iteration of the experiment:
for m in range(M):

    # Define arrays to hold the results of this iteration (batch).
    em_batch = []
    sim_batch = []
        
    # For each number of qubits:
    for n in num_qubits:
        
        # Initialize the simulator to a random state.
        psi = simulator.QubitRegister(n, 'base', 0, 0)
        rng = simulator.RandomNumberGenerator()
        seed = np.random.randint(0, 10000, 1)
        rng.SetSeedStreamPtrs(seed)
        psi.SetRngPtr(rng)
        psi.Initialize('rand', 0)
        
        # Extract the simulator state to pass it into the emulator.
        state = []
        for i in range(2**n):
            state.append(psi[i])
        state = np.array(state, dtype=complex)
        
        # Perform the QFT with the emulator and time how long it takes.
        start_time = time.time()
        em_state = emulator.qft(state)
        em_batch.append(time.time() - start_time)

        # Perform the QFT with the simulator and time how long it takes.
        start_time = time.time()
        for i in range(n-1, -1, -1):
            psi.ApplyHadamard(i)
            for j in range(0, i, 1):
                psi.ApplyControlled1QubitGate(j, i, Z[i-j-1])
        for i in range(n//2):
            psi.ApplySwap(i, n-i-1)
        sim_batch.append(time.time() - start_time)
        
        # Extract the resultant state from the simulator.
        sim_state = []
        for i in range(2**n):
            sim_state.append(psi[i])
        sim_state = np.array(sim_state, dtype=complex)
        
        # Ensure that the final states of the emulator and simulator agree.
        if not np.allclose(sim_state, em_state):
            print('Output differed for ' + str(n) + ' qubits.')
            print(sim_state)
            print(em_state)
            print()

    # Append the batch results to the main array.     
    em_times.append(em_batch)
    sim_times.append(sim_batch)
        
# Average the times over each batch.
em_times = np.array(em_times)
sim_times = np.array(sim_times)
em_times = np.sum(em_times, axis=0)/M
sim_times = np.sum(sim_times, axis=0)/M

# Plot the times for each QFT operation.
fig = plt.figure()
plt.plot(num_qubits, em_times, 'o-k', label='Emulator')   
plt.plot(num_qubits, sim_times, 'o-r', label='Simulator')
plt.title('Speed Comparison for QFT')
plt.xlabel('Number of Qubits')
plt.ylabel('Time (seconds)')
plt.legend(loc='best')
plt.savefig('Plots/qft.png', dpi=600)

# Plot the times for each QFT operation on a log plot.
fig = plt.figure()
plt.semilogy(num_qubits, em_times, 'o-k', label='Emulator')   
plt.semilogy(num_qubits, sim_times, 'o-r', label='Simulator')
plt.title('Speed Comparison for QFT on Log Plot')
plt.xlabel('Number of Qubits')
plt.ylabel('Time (seconds)')
plt.legend(loc='best')
plt.savefig('Plots/qft_log.png', dpi=600)