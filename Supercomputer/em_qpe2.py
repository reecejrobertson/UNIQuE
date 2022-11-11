import sys
sys.path.append('/fslhome/reecejr/software/QuantumComputingEmulator') # Change this to match your installation location.
sys.path.append('/fslhome/reecejr/software/intel-qs/build/lib') # Change this to match your installation location.
import intelqs_py as simulator
import emulator
import numpy as np
import time
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

# Define the number of times to repeat each following experiment.
M = 10

# Define the number of qubits to simulate for each experiment.
min_qubit = 2
max_qubit = 14

# Define the function that we will use to fit the curves.
def curve(x, a, b):
    return a * (2 ** (b * x))

# ---------------------------------------------------------------------------- #
#                                 Emulator QPE                                 #
# ---------------------------------------------------------------------------- #

# Set the number of qubits to use for each estimation.
S = 15

# Create a list of various numbers of qubits <= max_qubit to simulate.
num_qubits = np.arange(min_qubit, max_qubit+1, 2)

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
    
print('----------')
      
# Average the times over each batch to get the average time for each operation.
em_array = np.array(em_times)
em_array = np.sum(em_times, axis=0)/m

# Record the raw data.
print("Emulator data:", em_array)
print('----------')

# Plot the times for each QPE operation.
fig = plt.figure()
plt.plot(num_qubits, em_array, 'o-k')   
plt.xlabel('Number of Qubits in U Matrix')
plt.ylabel('Time (seconds)')
plt.savefig('Plots/em_qpe2.png', dpi=600)

# Fit a curve to the data.
em_params = curve_fit(f=curve, xdata=num_qubits, ydata=em_array, p0=[1, 1], bounds=(-np.inf, np.inf))[0]

# Record the parameters of the fit curve.
print('Parameters for emulator curve:', em_params)

# Plot the raw data points and the fit curve.
domain = np.linspace(min_qubit, max_qubit, 1000)
fig = plt.figure()
plt.plot(num_qubits, em_array, 'ok', label='Emulator Data')
plt.plot(domain, curve(domain, em_params[0], em_params[1]), 'k', label='Emulator Fit Curve')
plt.xlabel('Number of Qubits in U Matrix')
plt.ylabel('Time (seconds)')
plt.legend(loc='best')
plt.savefig('Plots/em_qpe2_fit.png', dpi=600)