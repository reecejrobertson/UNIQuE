from qiskit import IBMQ
from qiskit.utils import QuantumInstance
from qiskit.algorithms import Shor
import time
import datetime

def run_shors(N, a, shots, backend, outfile='Data/qiskit_shors.txt'):
    print(N, a, shots)
    print('Begin:', datetime.datetime.now())
    factors = Shor(QuantumInstance(backend, shots=shots, skip_qobj_validation=False))

    start_time = time.time()
    result_dict = factors.factor(N=N, a=a) # Where N is the integer to be factored
    execution_time = time.time() - start_time
    result = result_dict.factors

    with open(outfile, 'a') as out_file:
        out_file.write('----\n')
        out_file.write('a       = ' + str(a) + '\n')
        out_file.write('N       = ' + str(N) + '\n')
        out_file.write('Shots   = ' + str(shots) + '\n')
        out_file.write('Time    = ' + str(execution_time) + '\n')
        out_file.write('Factors = ' + str(result) + '\n')
    print('End:', datetime.datetime.now())
    print()

IBMQ.enable_account('99c0e0cf0828e07052aa037e95f66583b58e649173193b0743e9eeb3458cb4ec5ce2980423e36d2a13ca1a9e6cc1ac42c68799865ec41ea58d89964c662a1273') # Enter your API token here
provider = IBMQ.get_provider(hub='ibm-q')
backend = provider.get_backend('ibmq_qasm_simulator') # Specifies the quantum device

N =  15
a = 7
shots = 1
run_shors(N, a, shots, backend)

shots = 10
run_shors(N, a, shots, backend)

shots = 100
run_shors(N, a, shots, backend)

shots = 1000
run_shors(N, a, shots, backend)

N =  35
a = 13
shots = 10
run_shors(N, a, shots, backend)

N =  21
a = 5
shots = 10
run_shors(N, a, shots, backend)

N =  33
a = 7
shots = 10
run_shors(N, a, shots, backend)

# N =  67 * 127   # 8509
# a = 38
# shots = 1
# run_shors(N, a, shots, backend)

# N = 179 * 239   # 42781
# a = 10
# shots = 1
# run_shors(N, a, shots, backend)