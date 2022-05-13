import numpy as np
from fractions import Fraction
from scipy.fftpack import fft, ifft
from scipy import sparse as sp
from scipy.sparse import linalg as la


def normalize(x):
    """
    Accepts a state vector and normalizes it.

    Args:
        x (ndarray):    A state vector.

    Returns:
        ndarray:        The normaliztion of x.
    """

    return x / np.linalg.norm(x) # Divide x by the root of its squared sum.


def add(a, b):
    """
    Takes two arbitrary superposition states and returns their sum.

    Args:
        a (ndarray):    A state vector.
        b (ndarray):    A state vector.

    Returns:
        ndarray:        The state vector for the sum a+b.
    """

    # Create a c vector that is twice as long as the longest input.
    N = max(len(a), len(b))
    c = np.zeros(N+N, dtype=complex)

    # Compute the sum of the two quantum states.
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            c[i+j] += x*y

    # Normalize the result.
    c = normalize(c)

    # Return the sum.
    return c


def multiply(a, b):
    """
    Takes two arbitrary superposition states and returns their product.

    Args:
        a (ndarray):    A state vector.
        b (ndarray):    A state vector.

    Returns:
        ndarray:        The state vector for the sum a+b.
    """

    # Create a c vector that is the square of the size of the largest input.
    c = np.zeros(len(a)*len(b), dtype=complex)

    # Compute the product of the two quantum states.
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            c[i*j] += x*y

    # Normalize the result.
    c = normalize(c)

    # Return the product.
    return c


def exponentiate(a, b):
    """
    Takes two arbitrary superposition states and returns a**b.
    Note that in this case order matters (a**b != b**a in general).

    Args:
        a (ndarray):    A state vector.
        b (ndarray):    A state vector.

    Returns:
        ndarray:        The state vector for the sum a+b.
    """

    # Create a c vector that is large enough to hold the resultant state.
    c = np.zeros(len(a)**len(b), dtype=complex)   # Preserves a size that is a power of 2.

    # Compute the exponential of the two quantum states.
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            c[i**j] += x*y

    # Normalize the result.
    c = normalize(c)

    # Return the exponential.
    return c


def qft(x):
    """
    Performs the quantum Fourier transform on a given state vector.

    Args:
        x (ndarray):    A state vector.

    Returns:
        ndarray:        The normalized result of the qft (ifft) on x.
    """

    return normalize(ifft(x))    #Perform the ifft on x and normalize.


def inv_qft(x):
    """
    Performs the inverse quantum Fourier transform on a given state vector.

    Args:
        x (ndarray):    A state vector.

    Returns:
        ndarray:        The normalized result of the inverse qft (fft) on x.
    """

    return normalize(fft(x))    #Perform the fft on x and normalize.


def qpe(U, phi, b):
    """
    Performs quantum phase estimation on the matrix U with eigenvector |phi>
    such that U|phi>=e^{2*pi*i/theta}|phi>.

    Args:
        U (ndarray):    An NxN unitary operator.
        phi (ndarray):  An Nx1 eigenvector of U.
        b (int):        The number of qubits to use on the counting register.

    Returns:
        ndarray:        An approimation of the (2^b)*theta such that 
                        U|phi>=e^{2*pi*i/theta}|phi>. It is returned as a (2^b)x1
                        quantum state vector approximation.
    """
    
    # Assert that U is a square matrix of size NxN where N=2^n for some n.
    m, n = U.shape
    if m != n or np.log2(n) % 1 != 0:
        raise ValueError('U must be an NxN unitary matrix where N=2^n for some integer n.')
    
    # Get the eigenvalue (phase) corresponding to the given eigenvector phi.
    evals, evecs = np.linalg.eig(U)
    indx = list(filter(lambda x: np.allclose(evecs[x, :], phi), range(len(evecs))))
    phase = evals[indx[0]]
    
    # Compute the theta such that e^{2*pi*i/theta}=phase.
    theta = np.log(phase)/(2*np.pi*1j)
    
    # Multiply theta by 2^b, round the result, and convert it to an int.
    theta = int(np.round(theta.real * (2**b)))
    
    # Create a state of size 2^b with state[theta]=1. Return that state.
    state = np.zeros(2**b, dtype=complex)
    state[theta] = 1
    return state


def measure(x, include_index=False):
    """
    Performs a measurement operation on the vector |x>.

    Args:
        x (ndarray):                    A nx1 unitary vector
        include_index (bool, optional): Determines whether the index of the
                                        nonzero state of the measured vector is
                                        returned. Defaults to False.

    Returns:
        ndarray:        The measured vector x.
        int (optional): The index of the nonzero element of x.
                        Only returned if include_index=True.
    """

    # Choose a random number between 0 and the length of x,
    # with probabilities given by x**2.
    N = len(x)
    indx = np.random.choice(np.arange(0, N), p=np.abs(x)**2)

    # Create a new x vector of the size with 1 at the chosen number and 0 else.
    x_new = np.zeros(N)
    x_new[indx] = 1

    # Return the new x vector, and the chosen index if requested.
    if include_index:
        return x_new, indx
    else:
        return x_new


def shors(X, a, m, n):
    """
    Evaluates Shor's algorithm to attempt to find factors of X.
    By nature, this algorithm does not always succeed, so it is recommended
    that one run it several times to increase the probability of success.

    Args:
        X (int): The number to factor.
        a (int): The number used to find the period of X through modular exponentiation. 
        m (int): The number of qubits to use in the first (counting) register.
        n (int): The number of qubits to use in the second register.

    Returns:
        ndarray:    An array containing (ideally) one or more factors r of X,
                    such that a^r mod X = 1.
    """

    # Create the first register of m qubits. Note, normally this is put into 
    # superposition, but for this emulation algorithm this is unnecessary.
    M = 2**m
    first = np.ones(M)

    # Create the second empty register.
    # Note, this register is normally in the zero state.
    second = np.zeros(2**n)

    # Create a third register (which does not appear in the quantum Shor's)
    # to hold the result of a^x mod X for all x in [0, 1, ..., M-1].
    third = np.zeros(M)

    # Compute a^x mod X for all x in [0, 1, ..., M-1]. Increment each entry of
    # the second register by the number of times that entry appears.
    for i in range(0, M):
        mod_exp = pow(a, i, X)
        second[mod_exp] += 1
        third[i] = mod_exp

    # Normalize and measure the second register.
    second = normalize(second)
    second, indx = measure(second, include_index=True)

    # Filter out every element of the first register that does not correspond
    # to the measured state of the second element, and normalize.
    first[third != indx] = 0
    first = normalize(first)

    # Perform the inverse quantum Fourier transform on the first register.
    first = inv_qft(first)

    # Measure the first register, and use the continued fractions algorithm
    # to find the proper candidate for the period of a^x mod X.
    first, r = measure(first, include_index=True)
    frac = Fraction(r/M).limit_denominator(X)
    r = frac.denominator

    # Return our estimate for the period of a^x mod X.
    return r
    
    
def normalize_sparse(x):
    """
    Accepts a state vector and normalizes it.

    Args:
        x (ndarray):    A state vector.

    Returns:
        ndarray:        The normaliztion of x.
    """

    x_sum = 0
    for val in x.values():
        x_sum += np.abs(val)**2
    return x / np.sqrt(x_sum)   # Divide x by the root of its squared sum.


def add_sparse(a, b):
    """
    Takes two arbitrary superposition states and returns their sum.

    Args:
        a (ndarray):    A state vector.
        b (ndarray):    A state vector.

    Returns:
        ndarray:        The state vector for the sum a+b.
    """

    # Create a c vector that is twice as long as the longest input.
    N = max(a.shape[0], b.shape[0])
    c = sp.dok_matrix((N+N, 1), dtype=complex)

    # Compute the sum of the two quantum states.
    for i in a.nonzero()[0]:
        for j in b.nonzero()[0]:
            c[i+j] += a[i]*b[j]

    # Normalize the result.
    c = normalize_sparse(c)

    # Return the sum.
    return c


def multiply_sparse(a, b):
    """
    Takes two arbitrary superposition states and returns their product.

    Args:
        a (ndarray):    A state vector.
        b (ndarray):    A state vector.

    Returns:
        ndarray:        The state vector for the sum a+b.
    """

    # Create a c vector that is the square of the size of the largest input.
    c = sp.dok_matrix((a.shape[0]*b.shape[0], 1), dtype=complex)

    # Compute the product of the two quantum states.
    for i in a.nonzero()[0]:
        for j in b.nonzero()[0]:
            c[i*j] += a[i]*b[j]

    # Normalize the result.
    c = normalize_sparse(c)

    # Return the product.
    return c


def exponentiate_sparse(a, b):
    """
    Takes two arbitrary superposition states and returns a**b.
    Note that in this case order matters (a**b != b**a in general).

    Args:
        a (ndarray):    A state vector.
        b (ndarray):    A state vector.

    Returns:
        ndarray:        The state vector for the sum a+b.
    """

    # Create a c vector that is large enough to hold the resultant state.
    c = sp.dok_matrix((a.shape[0]**b.shape[0], 1), dtype=complex)   # Preserves a size that is a power of 2.

    # Compute the exponential of the two quantum states.
    for i in a.nonzero()[0]:
        for j in b.nonzero()[0]:
            c[i**j] += a[i]*b[j]

    # Normalize the result.
    c = normalize_sparse(c)

    # Return the exponential.
    return c