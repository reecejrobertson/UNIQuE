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
max_qubit = 14

# Define the function that we will use to fit the curves.
def curve(x, a, b):
    return a * (2 ** (b * x))

# ---------------------------------------------------------------------------- #
#                                Multiplication                                #
# ---------------------------------------------------------------------------- #

# Create a list of various numbers of qubits <= N to simulate.
num_qubits = np.arange(min_qubit, max_qubit+1, 1)

# Define a list to hold the times of each operation.
add_times = []
mult_times = []

# For each iteration of the experiment:
for m in range(1, M+1):
    
    print('m = ' + str(m) + '...', end='')
    
    # Define arrays to hold the results of this iteration (batch).
    add_batch = []
    mult_batch = []

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
        emulator.add_sparse(a, b)
        add_batch.append(time.time() - start_time)

        # Perform the multiplication with the emulator and time how long it takes.
        start_time = time.time()
        emulator.multiply_sparse(a, b)
        mult_batch.append(time.time() - start_time)
        
    # Append the batch results to the main array.
    add_times.append(add_batch)
    mult_times.append(mult_batch)
    
    print('Done')

print('----------')

# Average the times over each batch to get the average time for each operation.
add_array = np.array(add_times)
mult_array = np.array(mult_times)
add_array = np.sum(add_times, axis=0)/m
mult_array = np.sum(mult_times, axis=0)/m

# Record the raw data.
print("Addition data:", add_array)
print("Multiplication data:", mult_array)
print('----------')

# Plot the times for each operation.
fig = plt.figure()
plt.plot(num_qubits, add_array, 'o-b', label='Addition')
plt.plot(num_qubits, mult_array, 'o-g', label='Multiplication')
plt.xlabel('Number of Qubits per Operand')
plt.ylabel('Time (seconds)')
plt.legend(loc='best')
plt.savefig('Plots/add_mult.png', dpi=600)

# Fit a curve to the data.
add_params = curve_fit(f=curve, xdata=num_qubits, ydata=add_array, p0=[0, 0], bounds=(-np.inf, np.inf))[0]
mult_params = curve_fit(f=curve, xdata=num_qubits, ydata=mult_array, p0=[0, 0], bounds=(-np.inf, np.inf))[0]

# Record the parameters of the fit curve.
print('Parameters for addition curve:', add_params)
print('Parameters for multiplication curve:', mult_params)

# Plot the raw data points and the fit curve.
domain = np.linspace(min_qubit, max_qubit, 1000)
fig = plt.figure()
plt.plot(num_qubits, add_array, 'ob', label='Addition Data')
plt.plot(domain, curve(domain, add_params[0], add_params[1]), 'b', label='Addition Fit Curve')
plt.plot(num_qubits, mult_array, 'og', label='Multiplication Data')
plt.plot(domain, curve(domain, mult_params[0], mult_params[1]), 'g', label='Multiplication Fit Curve')
plt.xlabel('Number of Qubits Per Operand')
plt.ylabel('Time (seconds)')
plt.legend(loc='best')
plt.savefig('Plots/add_mult_fit.png', dpi=600)