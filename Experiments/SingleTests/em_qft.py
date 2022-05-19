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
N_EM_QFT = 40

# ---------------------------------------------------------------------------- #
#                                 Emulator QFT                                 #
# ---------------------------------------------------------------------------- #

# Set the maximum number of qubits to simulate.
N = N_EM_QFT

# Create a list of various numbers of qubits <= N to simulate.
num_qubits = np.arange(2, N+1, 2)

# Define a list to hold the emulator and simulator times respectively.
em_times = []

# For each iteration of the experiment:
for m in range(M):
    
    # Define arrays to hold the results of this iteration (batch).
    em_batch = []

    # For each number of qubits:
    for n in num_qubits:
        
        # Define a random initial state.
        state = np.random.uniform(0, 1, 2**n).astype(complex)
        
        # Perform the QFT with the emulator and time how long it takes.
        start_time = time.time()
        emulator.qft(state)
        em_batch.append(time.time() - start_time)

    # Append the batch results to the main array.
    em_times.append(em_batch)
      
# Average the times over each batch to get the average time for each operation.
em_times = np.array(em_times)
em_times = np.sum(em_times, axis=0)/M

# Plot the times for each QFT operation.
fig = plt.figure()
plt.plot(num_qubits, em_times, 'o-k')   
plt.title('Emulator Speed for QFT')
plt.xlabel('Number of Qubits')
plt.ylabel('Time (seconds)')
plt.savefig('Plots/em_qft.png', dpi=600)

# Plot the times for each QFT operation on a log plot.
fig = plt.figure()
plt.semilogy(num_qubits, em_times, 'o-k')   
plt.title('Emulator Speed for QFT on Log Plot')
plt.xlabel('Number of Qubits')
plt.ylabel('Time (seconds)')
plt.savefig('Plots/em_qft_log.png', dpi=600)
