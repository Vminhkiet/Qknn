import numpy as np
import base.encoding, base.constant, qiskit


def mahoa(vector1):
    vector1=base.encoding.Encoding(vector1,'dc_amplitude_encoding')
    return vector1.qcircuit

def chiso_fredkin_gate(N, padding = 0):
    """Get paramaters for log2(N) Fredkin gates

    Args:
        - N (int): dimensional of states
        - padding (int, optional): Defaults to 0.

    Returns:
        - list of int: params for the second and third Frekin gates
    """
    vector=[]
    for i in range(0,int(np.log2(N))):
        vector.append(2**i+padding)
    return vector

def tao_mach_fredkin_gate(vector1,vector2):
    N = len(vector1)
    cs1 = chiso_fredkin_gate(N)
    cs2 = chiso_fredkin_gate(N, N - 1)
    total_qubits = (N-1)*2+1
    # Construct circuit
    qc = qiskit.QuantumCircuit(total_qubits, 1)
    qc.h(0)
    qc.compose(mahoa(vector1), qubits=[*range(1, N)], inplace=True)
    qc.compose(mahoa(vector2), qubits=[*range(N, 2*N - 1)], inplace=True)
    for i in range(len(cs1)):
        qc.cswap(0, cs1[i], cs2[i])
    qc.h(0)
    qc.measure(0,0)
    return qc

def integrated_swap_test_circuit(vector1, vector2):
    """Return fidelity between two same - dimension vectors by cswaptest

    Args:
        - vector1 (numpy array): First vector, don't need to normalize
        - vector2 (numpy array): Second vector, don't need to normalize

    Returns:
        - float: Fidelity = sqrt((p(0) - p(1)) / n_shot)
    """
    if len(vector1) != len(vector2): 
        raise Exception('Two states must have the same dimensional')
    vector1 = vector1 / np.linalg.norm(vector1)
    vector2 = vector2 / np.linalg.norm(vector2)
    qc = tao_mach_fredkin_gate(vector1, vector2)
    counts = qiskit.execute(qc, backend = qiskit.Aer.get_backend('qasm_simulator'), shots = base.constant.num_shots).result().get_counts()
    return np.sqrt(np.abs((counts.get("0", 0) - counts.get("1", 0)) / base.constant.num_shots))


def do_muc_do(vector1,vector2,iteration: int=1):
    mangmucdo=np.array([])
    for i in range(0,iteration):
        mang=integrated_swap_test_circuit(vector1,vector2)
        mangmucdo=np.append(mangmucdo,mang)
    return np.average(mangmucdo)

