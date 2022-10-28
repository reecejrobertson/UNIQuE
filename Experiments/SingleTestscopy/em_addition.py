import sys
sys.path.append('/fslhome/reecejr/software/QuantumComputingEmulator') # Change this to match your installation location.
sys.path.append('/fslhome/reecejr/software/intel-qs/build/lib') # Change this to match your installation location.
import intelqs_py as simulator
import emulator
import numpy as np
import time
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from random import randint

# Define the number of times to repeat each following experiment.
M = 10

# Define the number of qubits to simulate for each experiment.
min_qubit = 2
max_qubit = 18

# Define the function that we will use to fit the curves.
def curve1(x, a, b):
    return a * (2 ** (b * x))

# Define the function that we will use to fit the curves.
def curve2(x, a, b):
    return (a * x) * (2 ** (b * x))

# Define a function to compute the Mean Squared Error between a curve and data points.
def MSE(func, param1, param2, data, points):
    pred = func(points, param1, param2)
    print("Pred: ", pred)
    error = pred - data
    error = error ** 2
    print("Error: ", error)
    print("Total error: ", sum(error))
    return sum(error)

# ---------------------------------------------------------------------------- #
#                                   Addition                                   #
# ---------------------------------------------------------------------------- #

# Create a list of various numbers of qubits <= N to simulate.
num_qubits = np.arange(min_qubit, max_qubit+1, 2)

# Define a list to hold the emulator and simulator times respectively.
em_times = []

# For each iteration of the experiment:
for m in range(1, M+1):

    print('m = ' + str(m) + '...', end='')
        
    # Define arrays to hold the results of this iteration (batch).
    em_batch = []

    # For each n, we will add two numbers of size n, as follows:
    for n in num_qubits:
        
        # Define a random initial state.
        a = np.random.uniform(0, 1, 2**n).astype(complex)
        b = np.random.uniform(0, 1, 2**n).astype(complex)
        
        # Perform addition with the emulator and time how long it takes.
        start_time = time.time()
        em_state = emulator.add(a, b)
        em_batch.append(time.time() - start_time)

    # Append the batch results to the main array.     
    em_times.append(em_batch)
    
    print('Done')

print('----------')

# Average the times over each batch.
em_array = np.array(em_times)
em_array = np.sum(em_times, axis=0)/m

# Record the raw data.
print("Emulator data:", em_array)
print('----------')

# Plot the times for each addition operation.
fig = plt.figure()
plt.plot(num_qubits, em_array, 'o-k', label='Emulator')   
plt.xlabel('Number of Qubits Per Addend')
plt.ylabel('Time (seconds)')
plt.legend(loc='best')
plt.savefig('Plots/addition.png', dpi=600)
    
# Fit a curve to the data.
em_params1 = curve_fit(f=curve1, xdata=num_qubits, ydata=em_array, p0=[0, 0], bounds=(-np.inf, np.inf))[0]

# Compute the mean squared error for this curve.
print('Emulator error on curve 1:')
em_err1 = MSE(curve1, em_params1[0], em_params1[1], em_array, num_qubits)

# Record the parameters of the fit curve.
print('Parameters for emulator curve1:', em_params1)
print('----------')

# Fit a curve to the data.
em_params2 = curve_fit(f=curve2, xdata=num_qubits, ydata=em_array, p0=[0, 0], bounds=(-np.inf, np.inf))[0]

# Compute the mean squared error for this curve.
print('Emulator error on curve 2:')
em_err2 = MSE(curve2, em_params2[0], em_params2[1], em_array, num_qubits)

# Record the parameters of the fit curve.
print('Parameters for emulator curve2:', em_params2)

# Plot the raw data points and the fit curve.
domain = np.linspace(min_qubit, max_qubit, 1000)
fig = plt.figure()
plt.plot(num_qubits, em_array, 'ok', label='Emulator Data')
if (em_err1 <= em_err2):
    print("Emulator array best fit by curve 1.")
    plt.plot(domain, curve1(domain, em_params1[0], em_params1[1]), 'k', label='Emulator Fit Curve')
else:
    print("Emulator array best fit by curve 2.")
    plt.plot(domain, curve2(domain, em_params2[0], em_params2[1]), 'k', label='Emulator Fit Curve')
plt.xlabel('Number of Qubits Per Addend')
plt.ylabel('Time (seconds)')
plt.legend(loc='best')
plt.savefig('Plots/addition_fit.png', dpi=600)