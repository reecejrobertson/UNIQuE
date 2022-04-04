import sys
sys.path.insert(0, '../intel-qs/build/lib')
import intelqs_py as simulator
import emulator
import numpy as np
import time
from matplotlib import pyplot as plt

# Define the number of times to repeat each following experiment.
M = 10

# Define the number of qubits to simulate for each experiment.
N_ADDITION = 35
N_QFT = 35
N_QPE = 35
N_ARITHMETIC = 3
N_MULTIPLICATION = 17
N_EM_QFT = 40
N_EM_QPE = 40

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
for m in range(M):

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

# Average the times over each batch.
em_times = np.array(em_times)
sim_times = np.array(sim_times)
em_times = np.sum(em_times, axis=0)/M
sim_times = np.sum(sim_times, axis=0)/M

# Plot the times for each addition operation.
fig = plt.figure()
plt.plot(num_qubits, em_times, 'o-k', label='Emulator')   
plt.plot(num_qubits, sim_times, 'o-r', label='Simulator')
plt.title('Speed Comparison for Addition')
plt.xlabel('Number of Qubits')
plt.ylabel('Time (seconds)')
plt.legend(loc='best')
plt.savefig('Plots/addition.png', dpi=600)

# Plot the times for each addition operation on a log plot.
fig = plt.figure()
plt.semilogy(num_qubits, em_times, 'o-k', label='Emulator')   
plt.semilogy(num_qubits, sim_times, 'o-r', label='Simulator')
plt.title('Speed Comparison for Addition on Log Plot')
plt.xlabel('Number of Qubits')
plt.ylabel('Time (seconds)')
plt.legend(loc='best')
plt.savefig('Plots/addition_log.png', dpi=600)

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

# ---------------------------------------------------------------------------- #
#                                      QPE                                     #
# ---------------------------------------------------------------------------- #

# Set the maximum number of qubits to simulate.
N = N_QPE

# Define a matrix (U) and an eigenvector.
z = np.random.uniform(0, 1)
U = np.array([[1, 0], [0, np.exp(1j*z)]])
phi = np.array([0, 1])

# Define the z-rotations needed for the simulation method.
Z = []
for n in range(1, N):
   Z.append(np.array([[1, 0], [0, np.exp(-1j*np.pi/2**n)]], dtype=complex))

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
        
        # Initialize the simulator to the 0 state.
        psi = simulator.QubitRegister(n+1, 'base', 0, 0)
        
        # Extract the simulator state to pass it into the emulator.
        state = []
        for i in range(2**n):
            state.append(psi[i])
        state = np.array(state, dtype=complex)
        
        # Perform the QPE with the emulator and time how long it takes.
        start_time = time.time()
        em_state = emulator.qpe(U, phi, n)
        em_batch.append(time.time() - start_time)

        # Perform the QPE with the simulator and time how long it takes.
        start_time = time.time()
        for i in range(n):
            psi.ApplyHadamard(i)
        psi.ApplyPauliX(n)
        for i in range(0, n):
            for j in range(2**i):
                psi.ApplyControlled1QubitGate(i, n, U)
        for i in range(n//2):
            psi.ApplySwap(i, n-i-1)
        for j in range(n):
            for m in range(j):
                psi.ApplyControlled1QubitGate(m, j, Z[j-m-1])
            psi.ApplyHadamard(j)
        for i in range(n):
            prob = psi.GetProbability(i)
            if prob < 0.5:
                psi.CollapseQubit(i, False)
            else:
                psi.CollapseQubit(i, True)
        sim_batch.append(time.time() - start_time)
        
        # Extract the resultant state from the simulator.
        sim_state = []
        for i in range(2**n, 2**(n+1)):
            sim_state.append(psi[i])
        sim_state = np.array(sim_state, dtype=complex)
        
        # Ensure that the final states of the emulator and simulator agree.
        if not np.allclose(np.nonzero(sim_state), np.nonzero(em_state)):
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

# Plot the times for each QPE operation.
fig = plt.figure()
plt.plot(num_qubits, em_times, 'o-k', label='Emulator')   
plt.plot(num_qubits, sim_times, 'o-r', label='Simulator')
plt.title('Speed Comparison for QPE')
plt.xlabel('Number of Qubits')
plt.ylabel('Time (seconds)')
plt.legend(loc='best')
plt.savefig('Plots/qpe.png', dpi=600)

# Plot the times for each QPE operation on a log plot.
fig = plt.figure()
plt.semilogy(num_qubits, em_times, 'o-k', label='Emulator')   
plt.semilogy(num_qubits, sim_times, 'o-r', label='Simulator')
plt.title('Speed Comparison for QPE on Log Plot')
plt.xlabel('Number of Qubits')
plt.ylabel('Time (seconds)')
plt.legend(loc='best')
plt.savefig('Plots/qpe_log.png', dpi=600)

# ---------------------------------------------------------------------------- #
#                                  Arithmetic                                  #
# ---------------------------------------------------------------------------- #

# Set the maximum number of qubits to simulate.
N = N_ARITHMETIC

# Create a list of various numbers of qubits <= N to simulate.
num_qubits = np.arange(1, N+1, 1)

# Define a list to hold the times of each operation.
add_times = []
mult_times = []
exp_times = []

# For each iteration of the experiment:
for m in range(M):
    
    # Define arrays to hold the results of this iteration (batch).
    add_batch = []
    mult_batch = []
    exp_batch = []

    # For each number of qubits:
    for n in num_qubits:

        # Define a random initial state.
        a = np.random.uniform(0, 1, 2**n).astype(complex)
        b = np.random.uniform(0, 1, 2**n).astype(complex)

        # Perform the addition with the emulator and time how long it takes.
        start_time = time.time()
        emulator.add(a, b)
        add_batch.append(time.time() - start_time)

        # Perform the multiplication with the emulator and time how long it takes.
        start_time = time.time()
        emulator.multiply(a, b)
        mult_batch.append(time.time() - start_time)

        # Perform the exponentiation with the emulator and time how long it takes.
        start_time = time.time()
        emulator.exponentiate(a, b)
        exp_batch.append(time.time() - start_time)

    # Append the batch results to the main array.
    add_times.append(add_batch)
    mult_times.append(mult_batch)
    exp_times.append(exp_batch)

# Average the times over each batch to get the average time for each operation.
add_times = np.array(add_times)
mult_times = np.array(mult_times)
exp_times = np.array(exp_times)
add_times = np.sum(add_times, axis=0)/M
mult_times = np.sum(mult_times, axis=0)/M
exp_times = np.sum(exp_times, axis=0)/M

# Plot the times for each operation.
fig = plt.figure()
plt.plot(num_qubits, add_times, 'o-k', label='Addition')
plt.plot(num_qubits, mult_times, 'o-r', label='Multiplication')
plt.plot(num_qubits, exp_times, 'o-b', label='Exponentiation')
plt.title('Emulator Speed for Arithmetic Operations')
plt.xlabel('Number of Qubits')
plt.ylabel('Time (seconds)')
plt.legend(loc='best')
plt.savefig('Plots/arithmetic.png', dpi=600)

# Plot the times for each operation on a log plot.
fig = plt.figure()
plt.semilogy(num_qubits, add_times, 'o-k', label='Addition')
plt.semilogy(num_qubits, mult_times, 'o-r', label='Multiplication')
plt.semilogy(num_qubits, exp_times, 'o-b', label='Exponentiation') 
plt.title('Emulator Speed for Arithmetic Operations on Log Plot')
plt.xlabel('Number of Qubits')
plt.ylabel('Time (seconds)')
plt.legend(loc='best')
plt.savefig('Plots/arithmetic_log.png', dpi=600)

# ---------------------------------------------------------------------------- #
#                                Multiplication                                #
# ---------------------------------------------------------------------------- #

# Set the maximum number of qubits to simulate.
N = N_MULTIPLICATION

# Create a list of various numbers of qubits <= N to simulate.
num_qubits = np.arange(1, N+1, 1)

# Define a list to hold the times of each operation.
add_times = []
mult_times = []

# For each iteration of the experiment:
for m in range(M):
    
    # Define arrays to hold the results of this iteration (batch).
    add_batch = []
    mult_batch = []

    # For each number of qubits:
    for n in num_qubits:

        # Define a random initial state.
        a = np.random.uniform(0, 1, 2**n).astype(complex)
        b = np.random.uniform(0, 1, 2**n).astype(complex)

        # Perform the addition with the emulator and time how long it takes.
        start_time = time.time()
        emulator.add(a, b)
        add_batch.append(time.time() - start_time)

        # Perform the multiplication with the emulator and time how long it takes.
        start_time = time.time()
        emulator.multiply(a, b)
        mult_batch.append(time.time() - start_time)
        
    # Append the batch results to the main array.
    add_times.append(add_batch)
    mult_times.append(mult_batch)

# Average the times over each batch to get the average time for each operation.
add_times = np.array(add_times)
mult_times = np.array(mult_times)
add_times = np.sum(add_times, axis=0)/M
mult_times = np.sum(mult_times, axis=0)/M

# Plot the times for each operation.
fig = plt.figure()
plt.plot(num_qubits, add_times, 'o-k', label='Addition')
plt.plot(num_qubits, mult_times, 'o-r', label='Multiplication')
plt.title('Emulator Speed for Arithmetic Operations')
plt.xlabel('Number of Qubits')
plt.ylabel('Time (seconds)')
plt.legend(loc='best')
plt.savefig('Plots/add_mult.png', dpi=600)

# Plot the times for each operation on a log plot.
fig = plt.figure()
plt.semilogy(num_qubits, add_times, 'o-k', label='Addition')
plt.semilogy(num_qubits, mult_times, 'o-r', label='Multiplication')
plt.title('Emulator Speed for Arithmetic Operations on Log Plot')
plt.xlabel('Number of Qubits')
plt.ylabel('Time (seconds)')
plt.legend(loc='best')
plt.savefig('Plots/add_mult_log.png', dpi=600)

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
for m in range(M_hat):
    
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
      
# Average the times over each batch to get the average time for each operation.
em_times = np.array(em_times)
em_times = np.sum(em_times, axis=0)/M_hat

# Plot the times for each QPE operation.
fig = plt.figure()
plt.plot(num_qubits, em_times, 'o-k')   
plt.title('Emulator Speed for QPE')
plt.xlabel('Number of Qubits')
plt.ylabel('Time (seconds)')
plt.savefig('Plots/em_qpe.png', dpi=600)

# Plot the times for each QPE operation on a log plot.
fig = plt.figure()
plt.semilogy(num_qubits, em_times, 'o-k')   
plt.title('Emulator Speed for QPE on Log Plot')
plt.xlabel('Number of Qubits')
plt.ylabel('Time (seconds)')
plt.savefig('Plots/em_qpe_log.png', dpi=600)