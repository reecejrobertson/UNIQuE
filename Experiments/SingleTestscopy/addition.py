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
max_qubit = 14

# Define the function that we will use to fit the curves.
def curve(x, a, b):
    return a * (2 ** (b * x))

# ---------------------------------------------------------------------------- #
#                                   Addition                                   #
# ---------------------------------------------------------------------------- #

# Create a list of various numbers of qubits <= N to simulate.
num_qubits = np.arange(min_qubit, max_qubit+1, 2)

# Define a list to hold the emulator and simulator times respectively.
em_times = []
sim_times = []

# For each iteration of the experiment:
for m in range(1, M+1):

    print('m = ' + str(m) + '...', end='')
        
    # Define arrays to hold the results of this iteration (batch).
    em_batch = []
    sim_batch = []

    # For each n, we will add two numbers of size n, as follows:
    for n in num_qubits:
        
        # We will randomly flip bits from 0 to 1 to generate random numbers.
        # This arrays keep track of which bits we flip.
        a_val = []
        b_val = []
        
        # Initialize the state of the simulator.
        # Note that to add two numbers of size n, we need 2n + 2 qubits.
        psi = simulator.QubitRegister(2*n+2, 'base', 0, 0)
        for i in range(1, 2*n+1):
            flip = randint(0, 1)
            if flip:
                psi.ApplyPauliX(i)
            if i % 2 == 1:
                a_val.append(flip)
            else:
                b_val.append(flip)
        
        # Initialize the state of the emulator. Ensure that it is consistent with the simulator values.
        a = np.zeros(2**n, dtype=complex)
        b = np.zeros(2**n, dtype=complex)
        idx = 0
        for i, x in enumerate(a_val):
            idx += 2**i * x
        a[idx] = 1
        idx = 0
        for i, x in enumerate(b_val):
            idx += 2**i * x
        b[idx] = 1
        
        # Perform addition with the emulator and time how long it takes.
        start_time = time.time()
        em_state = emulator.add(a, b)
        em_batch.append(time.time() - start_time)

        # Perform the addition with the simulator and time how long it takes.
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
        
        # Extract the sum from the simulator state. The process is derived from "A New Quantum Ripple-Carry Addition Circuit".
        sim_state = []
        for i in range(2**(2*n+2)):
            sim_state.append(psi[i])
        sim_state = np.array(sim_state, dtype=complex)
        sim_indx = bin(sim_state.nonzero()[0][0])
        sim_indx = sim_indx[2:]
        sim_indx = sim_indx[::-1]   # Note, we reverse the order of the Python binary string to match the convention we are using.
        if len(sim_indx) == 1:
            sim_sum = sim_indx
        else:
            sim_sum = sim_indx[1::2]

        # Extract the sum from the emulator state.
        em_indx = bin(em_state.nonzero()[0][0])
        em_sum = em_indx[2::1]
        em_sum = em_sum[::-1]       # Note, we reverse the order of the Python binary string to match the convention we are using.

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

print('----------')

# Average the times over each batch.
em_array = np.array(em_times)
sim_array = np.array(sim_times)
em_array = np.sum(em_times, axis=0)/m
sim_array = np.sum(sim_times, axis=0)/m

# Record the raw data.
print("Emulator data:", em_array)
print("Simulator data:", sim_array)
print('----------')

# Plot the times for each addition operation.
fig = plt.figure()
plt.plot(num_qubits, em_array, 'o-k', label='Emulator')   
plt.plot(num_qubits, sim_array, 'o-r', label='Simulator')
plt.xlabel('Number of Qubits Per Addend')
plt.ylabel('Time (seconds)')
plt.legend(loc='best')
plt.savefig('Plots/addition.png', dpi=600)
    
# Fit a curve to the data.
em_params = curve_fit(f=curve, xdata=num_qubits, ydata=em_array, p0=[0, 0], bounds=(-np.inf, np.inf))[0]
sim_params = curve_fit(f=curve, xdata=num_qubits, ydata=sim_array, p0=[0, 0], bounds=(-np.inf, np.inf))[0]

# Record the parameters of the fit curve.
print('Parameters for emulator curve:', em_params)
print('Parameters for simulator curve:', sim_params)

# Plot the raw data points and the fit curve.
domain = np.linspace(min_qubit, max_qubit, 1000)
fig = plt.figure()
plt.plot(num_qubits, em_array, 'ok', label='Emulator Data')
plt.plot(domain, curve(domain, em_params[0], em_params[1]), 'k', label='Emulator Fit Curve')
plt.plot(num_qubits, sim_array, 'or', label='Simulator Data')
plt.plot(domain, curve(domain, sim_params[0], sim_params[1]), 'r', label='Simulator Fit Curve')
plt.xlabel('Number of Qubits Per Addend')
plt.ylabel('Time (seconds)')
plt.legend(loc='best')
plt.savefig('Plots/addition_fit.png', dpi=600)