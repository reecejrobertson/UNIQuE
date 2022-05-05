import time
import emulator
import numpy as np
from math import gcd

def demo_shors(a, N, m, n, mults=10, iters=1000, display=False, outfile='shors_tests.txt'):
    """
    Demonstrate Shor's Algorithm with a given choice of a, N, m, n.
    Computes a^x mod N for all x in the interval [0, 2^m].

    Args:
        a (int): The number to use for factoring.
        N (int): The number to factor.
        m (int): The number of qubits to use on the counting register.
        n (int): The number of quibts to use on the storage register.
                 Must be chosen such that 2^n > N to guarentee success.
        mults (int, optional): The number of multiples to take the estimate 
                               of the period to. Defaults to 10.
        iters (int, optional): The number of times to repeat the process.
                               Defaults to 1000.
        display (bool, optional): Prints a detailed analysis of each iteration
                                  if True. Defaults to False.
        outfile (string, optional): The file to save the output to.
                                    Defaults to 'shors_tests.txt'.
    """
    
    # Define variables to hold the time and accuracy.
    times = []
    accuracy = 0
    
    # Repeat the following experiment iters times:
    for i in range(iters):
        
        # Run and time Shor's algorithm to estimate the period of a^x mod N.
        start_time = time.time()
        r = emulator.shors(N, a, m, n)
        times.append(time.time() - start_time)
        
        # Set the initial factor guesses to 1, 1, and success = False.
        guesses = np.array([1, 1])
        success = False
        
        # Compute a few small multiples of r.
        for j in range(mults):
            mult = j + 1
            R = r * mult
            
            # If the R (the power of r) is even:
            if R % 2 == 0:
                
                # Get gcd(a^(R/2)-1, N) and gcd(a^(R/2)+1, N) to guess at factors of N.
                guesses = np.array([gcd(a**(R//2) - 1, N), gcd(a**(R//2) + 1, N)])
                
                # If both guesses are proper factors of N:
                if (guesses[0] != 1 and N % guesses[0] == 0 and
                   guesses[1] != 1 and N % guesses[1] == 0):
                    
                    # Count the trial as a success.
                    accuracy += 1
                    success = True
                    break
        
        # Give a detailed account of each trial if requested.
        if display:
            print('r:' + str(r) + ',\tR:' + str(R) + ',\tmult:' + str(mult) +
                  ', \tGuesses:' + str(guesses) + ',\tPass: ' + str(success))
    
    # Save the output to a text file.
    with open(outfile, 'a') as out_file:
        out_file.write('----\n')
        out_file.write('a            = ' + str(a) + '\n')
        out_file.write('N            = ' + str(N) + '\n')
        out_file.write('m            = ' + str(m) + '\n')
        out_file.write('n            = ' + str(n) + '\n')
        out_file.write('mults        = ' + str(mults) + '\n')
        out_file.write('iters        = ' + str(iters) + '\n')
        out_file.write('Accuracy     = ' + str(accuracy/iters) + '\n')
        out_file.write('Average time = ' + str(sum(times)/iters) + '\n')


# Set the variables for the experiment.
a = 38
N = 67 * 127
n = 14
mults = 10
iters = 10
M = 29

# Run the experiment.
for m in range(1, M+1):
    print('m =', m)
    demo_shors(a, N, m, n, mults=mults, iters=iters)