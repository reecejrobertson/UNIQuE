# Introduction
The Unconventional Noiseless Intermediate Quantum Emulator (UNIQuE) is the first open-source quantum computing emulator.
As an emulator, UNIQuE performs high level quantum subroutinues using optimized classical algorithms.
This is opposed to quantum simulators that perform all operations via matrix multiplications.

The functions provided by the emulator are found in the $\texttt{emulator.py}$ file.
The documenation for these functions is given below.
Additionally, experiments benchmarking UNIQuE against the Intel Quantum Simulator (Intel-QS, found at https://github.com/iqusoft/intel-qs) are found in the $\texttt{validate\\_qft.ipynb}$ and $\texttt{emulator\\_comparison.ipynb}$ files.
To run these files you will need to clone the Intel-QS repository and run the cmake files with Python binding enabled (see their documentation for details).
Then change the path variable at the top of the $\texttt{validate\\_qft.ipynb}$ and $\texttt{emulator\\_comparison.ipynb}$ files to match your installation location.

# Documentation

UNIQuE implements eight basic operations: $\texttt{normalize}$, $\texttt{add}$, $\texttt{multiply}$, $\texttt{exponentiate}$, $\texttt{qft}$, $\texttt{inv\\_qft}$, $\texttt{qpe}$, and $\texttt{measure}$.
It also implements a variation on the first four operations that employs sparse matrices to achieve greater spatial savings, which operations are $\texttt{normalize\\_sparse}$, $\texttt{add\\_sparse}$, $\texttt{multiply\\_sparse}$, $\texttt{exponentiate\\_sparse}$.
These operations are discussed below after their non-sparse counterparts.
Finally, UNIQuE implements Shor's algorithm, $\texttt{shors}$, the celebrated quantum factoring algorithm.
Note that because we are modeling state vectors of qubits, in all operations below it is assumed (but not required) that $N=2^n$ for some positive integer $n$. 

1. $\texttt{normalize(x)}$:
takes as input a numpy array $\texttt{x}$ of length $N$, computes the normalization of $\texttt{x}$, and returns a numpy array of length $N$ containing the result.

2. $\texttt{add(a, b)}$:
accepts two numpy arrays $\texttt{a}$ and $\texttt{b}$ of sizes $N_1$ and $N_2$, respectively, computes the sum $\texttt{a}+\texttt{b}$, and returns a numpy array $\texttt{c}$ of size $2\times\max(N_1 + N_2)$ containing the result.

3. $\texttt{multiply(a, b)}$:
accepts two numpy arrays $\texttt{a}$ and $\texttt{b}$ of sizes $N_1$ and $N_2$, respectively, computes the product $\texttt{a} \times \texttt{b}$, and returns a numpy array $\texttt{c}$ of length $N_1\times N_2$ containing the result.

4. $\texttt{exponentiate(a, b)}$:
accepts two numpy arrays $\texttt{a}$ and $\texttt{b}$ of sizes $N_1$ and $N_2$, respectively, computes $\texttt{a}^{\texttt{b}}$, and returns a numpy array $\texttt{c}$ of length $N_1^{N_2}$ containing the result.
Note that for this operation the ordering of $\texttt{a}$ and $\texttt{b}$ matters, as $\texttt{a}^{\texttt{b}}\neq\texttt{b}^{\texttt{a}}$ in general.

5. $\texttt{normalize\\_sparse(x)}$:
performs the same operation as the $\texttt{normalize}$ function on its input.
However, rather than using numpy arrays, this function uses the $\texttt{scipy.sparse.dok\\_matrix}$ framework.
It accepts a dok_matrix as input and returns a normalized dok_matrix for the output.

6. $\texttt{add\\_sparse(a, b)}$:
operates identically to the $\texttt{add}$ function above, however $\texttt{a}$, $\texttt{b}$, and $\texttt{c}$ are dok_matrix objects rather than numpy arrays.

7. $\texttt{multiply\\_sparse(a, b)}$:
operates identically to the $\texttt{multiply}$ function above, however $\texttt{a}$, $\texttt{b}$, and $\texttt{c}$ are dok_matrix objects rather than numpy arrays.

8. $\texttt{exponentiate\\_sparse(a, b)}$:
operates identically to the $\texttt{exponentiate}$ function above, however $\texttt{a}$, $\texttt{b}$, and $\texttt{c}$ are dok_matrix objects rather than numpy arrays.

9. $\texttt{qft(x)}$:
accepts a numpy array $\texttt{x}$ of length $N$, uses $\texttt{scipy.fftpack.ifft}$—the inverse discrete Fourier transform implemented by scipy—to classically compute the quantum Fourier transform, applies the $\texttt{normalize}$ function to preserve the vector's unitary property, and returns a numpy array of length $N$ containing the result.

10. $\texttt{inv\\_qft(x)}$:
accepts a numpy array $\texttt{x}$ of length $N$, uses $\texttt{scipy.fftpack.fft}$—the discrete Fourier transform implemented by scipy—to classically compute the inverse quantum Fourier transform, applies the $\texttt{normalize}$ function to preserve the vector's unitary property, and returns a numpy array of length $N$ containing the result.

11. $\texttt{qpe(U, phi, n)}$:
takes three inputs: an $M\times M$ unitary matrix $\texttt{U}$, an $M\times1$ eigenvector $\texttt{phi}$ in the form of a numpy array, and an integer $\texttt{n}$ which specifies the number of qubits of precision to use for the output of the function.
It is required that $M=2^m$ for some positive integer $m$.
The function finds all of the eigenvalues and eigenvectors of $\texttt{U}$ using $\texttt{numpy.linalg.eig(U)}$, and determines which eigenvalue corresponds to the eigenvector $\texttt{phi}$.
Because the eigenvalue is of the form $e^{2\pi i\theta}$, $\theta$ is extracted, and the integer $r\in[0,N]$ is found such that $r/N$ is the closest possible approximation of $\theta$, where $N=2^n$.
This value $r$ is encoded into a state vector (a dok_matrix specifically) of size $N$ and returned as the output of the function.

12. $\texttt{measure(x, return\\_index=False)}$:
takes a numpy array $\texttt{x}$ of length $N$ as input and returns a numpy array of the same size with a single nonzero entry, which entry is $1$.
The probability that any given index in the output will hold the value $1$ is given by the value of the corresponding entry of $\texttt{x}$ squared.
This random selection is made using the $\texttt{numpy.random.choice}$ function.
If $\texttt{return\\_index=True}$ then the index of the nonzero state is also returned.

13. $\texttt{shors(X, a, m, n)}$:
the inputs to this function are as follows: $\texttt{X}\in\mathbb{Z}$ is the number to factor, $\texttt{a}\in\mathbb{Z}$ is co-prime to $\texttt{X}$, and $\texttt{m},\texttt{n}\in\mathbb{N}$ represent the number of qubits in the first and second quantum registers, respectively.
It returns an estimate of the factors of $\texttt{X}$.
