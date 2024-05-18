# NOTE: 答题时只能修改此文件，整合一切代码，因为评测机只会上传这单个脚本！！
# 一点也不能改 solution() 函数的签名，例如不能给 molecule 加类型注解，因为评测机做了硬编码的校验！！

if 'env':
    import os, sys
    sys.path.append(os.path.abspath(__file__))

    import warnings
    warnings.simplefilter(action='ignore', category=UserWarning)
    warnings.simplefilter(action='ignore', category=RuntimeWarning)
    warnings.simplefilter(action='ignore', category=DeprecationWarning)

from typing import *

import numpy as np
from numpy import ndarray
from scipy.optimize import minimize

from mindquantum.simulator import Simulator
from mindquantum.core.operators import QubitOperator, Hamiltonian, TimeEvolution
from mindquantum.core.gates import H, X, Y, Z, RX, RY, RZ, CNOT, BasicGate, ParameterGate, MeasureResult
from mindquantum.core.circuit import Circuit, UN
from mindquantum.core.parameterresolver import ParameterResolver
from mindquantum.algorithm.nisq import Transform
from mindquantum.utils.progress import SingleLoopProgress
from mindquantum.third_party.unitary_cc import uccsd_singlet_generator
from mindquantum.framework import MQAnsatzOnlyLayer
from mindspore.nn.optim import SGD, Adam
from mindspore.nn.wrap.cell_wrapper import TrainOneStepCell

from simulator import HKSSimulator
from utils import generate_molecule, get_molecular_hamiltonian

PauliTerm = Tuple[float, QubitOperator]


''' Ansatz '''

def get_HF_circuit(mol) -> Circuit:
    # In HF-state, the electrons always occupy the lowest orbitals :)
    # see https://pennylane.ai/blog/2022/09/how-to-use-the-hartree-fock-method-in-pennylane/
    return UN(X, mol.n_electrons)

def get_uccsd_circuit(mol) -> Circuit:
    ucc = Transform(uccsd_singlet_generator(mol.n_qubits, mol.n_electrons)).jordan_wigner().imag
    ucc = TimeEvolution(ucc).circuit
    return get_HF_circuit(mol) + ucc

def get_wtf_circit(mol, depth:int=2) -> Circuit:
    circ = Circuit()
    for d in range(depth):
        for i in range(mol.n_qubits):
            circ += RY(f'd{d}_q{i}').on(i)
        for i in range(mol.n_qubits):
            circ += CNOT.on((mol.n_qubits + mol.n_electrons + i - 1) % mol.n_qubits, i)
    return circ

def get_ry_HEA_circit_no_hf(mol, depth:int=1) -> Circuit:
    ''' impl from https://github.com/liwt31/QC-Contest-Demo '''

    circ = Circuit()
    for i in range(mol.n_qubits):
        circ += RY(f'd0_q{i}').on(i)
    for j in range(1, depth+1):
        for i in range(0, mol.n_qubits, 2):
            circ += CNOT.on(i+1, i)
        for i in range(1, mol.n_qubits - 1, 2):
            circ += CNOT.on(i+1, i)
        for i in range(mol.n_qubits):
            circ += RY(f'd{j}_q{i}').on(i)
    return circ

def get_ry_HEA_circit(mol, depth:int=1) -> Circuit:
    ''' impl from https://github.com/liwt31/QC-Contest-Demo '''

    return get_ry_HEA_circit_no_hf(mol, depth) + get_HF_circuit(mol)


def prune_circuit(circ:Circuit, pr:ParameterResolver) -> Tuple[Circuit, ParameterResolver]:
    # TODO: circuit prune: removing 2*pi rotation gate, round pi fractions to non-parameter gate
    # sanitize
    to_keep: List[BasicGate] = []
    to_remove_keys: List[str] = [k for k in pr.keys() if abs(pr[k]) < 1e-6]
    for gate in circ:
        gate: BasicGate
        if not gate.parameterized:
            to_keep.append(gate)
        else:
            gate: ParameterGate
            to_remove = False
            for gate_pr in gate.get_parameters():
                for key in gate_pr.keys():
                    if key in to_remove_keys:
                        to_remove = True
                        break
                if to_remove: break
            if not to_remove:
                to_keep.append(gate)
    # rebuild
    circ_new = Circuit()
    for gate in to_keep:
        circ_new += gate
    pr_new = ParameterResolver({k: pr[k] for k in circ_new.params_name})
    return circ_new, pr_new


''' Hamiltonian '''

def split_hamiltonian(ham: QubitOperator) -> Tuple[float, List[PauliTerm]]:
    const = 0.0
    split_ham: List[PauliTerm] = []
    for pr, ops in ham.split():
        if ops == 1:    # aka. I
            const = pr.const.real
        else:
            split_ham.append([pr.const.real, ops])
    return const, split_ham

def combine_hamiltonian(const: float, terms: List[PauliTerm]) -> QubitOperator:
    from re import compile as Regex
    R_str = Regex('\[([XYZ\d ]+)\]')

    ham = QubitOperator('', const)
    for coeff, ops in terms:
        coeff_string = str(ops)
        string = R_str.findall(coeff_string)[0]
        ham += QubitOperator(string, coeff)
    return ham

def prune_hamiltonian(split_ham:List[PauliTerm]) -> List[PauliTerm]:
    from mindquantum._math.ops import QubitOperator as QubitOperator_

    def filter_Z_only(ops:QubitOperator) -> bool:
        for term in QubitOperator_.get_terms(ops):
            for qubit, symbol in term[0]:
                if symbol.name in ['X', 'Y']:
                    return False
        return True

    #split_ham = [it for it in split_ham if filter_Z_only(it[1])]
    split_ham = [it for it in split_ham if abs(it[0]) > 1e-2]
    split_ham.sort(key=lambda it: abs(it[0]), reverse=True)
    return split_ham


''' Optimize '''

def trim_p(x:ndarray, eps:float=1e-15) -> ndarray:
    return np.where(np.abs(x) < eps, 0.0, x)

def norm_p(x:ndarray) -> ndarray:
    x = x % (2*np.pi)       # [0, 2*pi]
    return np.where(x < np.pi, x, x - 2*np.pi)

def optim_sp(method:str, circ:Circuit, ham:Hamiltonian, p0:ndarray, tol:float=1e-8):
    # tricks
    TRIM_P = 1e-5   # 1e-9
    NORM_P = True   # True

    def func(x:ndarray, grad_ops):
        nonlocal TRIM_P, NORM_P
        if TRIM_P: x = trim_p(x, TRIM_P)
        if NORM_P: x = norm_p(x)

        f, g = grad_ops(x)
        f = f.real[0, 0]
        g = g.real[0, 0]
        return f, g

    if TRIM_P: p0 = trim_p(p0, TRIM_P)
    if NORM_P: p0 = norm_p(p0)

    sim = Simulator('mqvector', circ.n_qubits)
    grad_ops = sim.get_expectation_with_grad(ham, circ)
    options = {'maxiter':1000, 'disp':False}
    if method == 'COBYLA':
        options.update({'rhobeg': 1.57})
    res = minimize(func, p0, (grad_ops,), method, jac=True, tol=tol, options=options)

    print('min. fval:', res.fun)
    #print('argmin. x:', res.x)
    px = res.x
    if TRIM_P:
        px = trim_p(px, TRIM_P)
        #print('argmin. x (trimmed):', px)
    if NORM_P:
        px = norm_p(px)
        #print('argmin. x (normed):', px)
    return res.fun, ParameterResolver(dict(zip(circ.params_name, px)))

def optim_ms(method:str, circ: Circuit, ham:Hamiltonian, p0:ndarray):
    ''' Model & Optim '''
    sim = Simulator('mqvector', circ.n_qubits)
    grad_ops = sim.get_expectation_with_grad(ham, circ)
    net = MQAnsatzOnlyLayer(grad_ops)
    net.weight.data[:] = p0.tolist()
    if method == 'Adam':
        opt = Adam(net.trainable_params(), learning_rate=0.01)
    elif method == 'SGD':
        opt = SGD(net.trainable_params(), learning_rate=0.01, momentum=0.8)
    train_net = TrainOneStepCell(net, opt)
    ''' Train '''
    best_E = 99999
    best_weight = None
    for i in range(2000):
        E = train_net()
        if E < best_E:
            best_E = E
            best_weight = train_net.weights[0]
        if i % 100 == 0:
            print(f'[step {i}] expect: {E}')
    px = best_weight

    '''
    px = [
        -0.172711, 0.000000, 0.000000, -0.000000, 0.000000, 0.000000, 0.015735, -0.007540, -0.000000,
        0.000000, 0.000000, -0.000000, -0.000000, -0.000001, 1.570925, 0.005027, -0.000000, 0.000000,
        0.000081, 0.000000, 0.000000, 0.000054, 0.004083, 0.019818, 0.000000, -0.000007, -0.000000,
        0.000000, 0.000000, 0.000256, -1.570796, 0.002514
    ]
    '''
    return E, ParameterResolver(dict(zip(circ.params_name, px)))

def get_best_params(circ: Circuit, ham:QubitOperator, method:str='BFGS', init:str='randu') -> Tuple[float, ParameterResolver]:
    if init == 'zeros':
        p0 = np.zeros(len(circ.params_name))
    elif init == 'randu':
        p0 = np.random.uniform(-np.pi, np.pi, len(circ.params_name))
    elif init == 'randn':
        p0 = np.random.normal(0, 0.02, len(circ.params_name))

    if method in ['BFGS', 'COBYLA']:
        return optim_sp(method, circ, Hamiltonian(ham), p0)
    else:
        return optim_ms(method, circ, Hamiltonian(ham), p0)


''' Measure '''

def rotate_to_z_axis_and_add_measure(circ: Circuit, ops: QubitOperator) -> Circuit:
    circ = circ.copy()
    assert ops.is_singlet
    for idx, o in list(ops.terms.keys())[0]:
        if o == 'X':
            circ.ry(-np.pi / 2, idx)
        elif o == 'Y':
            circ.rx(np.pi / 2, idx)
        circ.measure(idx)
    return circ

def measure_single_ham(sim: Simulator, circ: Circuit, pr: ParameterResolver, ops: QubitOperator, shots:int=100) -> float:
    circ_m = rotate_to_z_axis_and_add_measure(circ, ops)
    result: MeasureResult = sim.sampling(circ_m, pr, shots, seed=None)
    exp = 0.0
    for bits, cnt in result.data.items():
        exp += (-1)**bits.count('1') * cnt
    return exp / shots

def get_min_exp(sim:Simulator, circ:Circuit, pr:ParameterResolver, split_ham:List[PauliTerm], shots:int=100, n_repeat:int=10, use_exp_fix:bool=False) -> float:
    result_min = 99999
    for _ in range(n_repeat):
        result = 0.0
        with SingleLoopProgress(len(split_ham), '哈密顿量测量中') as bar:
            for idx, (coeff, ops) in enumerate(split_ham):
                exp = measure_single_ham(sim, circ, pr, ops, shots)
                if use_exp_fix:
                    # FIXME: when circ is non-entangled, and gates are all X
                    # the BitFlip noise on measure gate could be cheaty moved out :)
                    if   exp > 0: exp = +1
                    elif exp < 0: exp = -1
                #print('coeff=', coeff, 'term=', ops, 'exp=', exp)
                result += exp * coeff
                bar.update_loop(idx)
        result_min = min(result_min, result)
    return result_min


''' Entry '''

def solution(molecule, Simulator: HKSSimulator) -> float:
    ''' Molecule '''
    molecule: List[Tuple[str, List[float]]]     # i.e.: geometry
    mol = generate_molecule(molecule)
    print('[mol]')
    print('  name:', mol.name)
    print('  n_atoms:', mol.n_atoms)
    print('  n_electrons:', mol.n_electrons)
    print('  n_orbitals:', mol.n_orbitals)
    print('  n_qubits:', mol.n_qubits)
    print('  nuclear_repulsion:', mol.nuclear_repulsion)
    print('  orbital_energies:', mol.orbital_energies)
    print('  hf_energy:', mol.hf_energy)
    print('  fci_energy:', mol.fci_energy)

    ''' Hamiltionian '''
    ham = get_molecular_hamiltonian(mol)
    const, split_ham = split_hamiltonian(ham)
    print('[ham]')
    print('  const:', const)
    print('  n_terms:', len(split_ham))
    #split_ham = prune_hamiltonian(split_ham)
    print('  n_terms (pruned):', len(split_ham))
    ham = combine_hamiltonian(const, split_ham)

    ''' Circuit & Params '''
    #circ = get_HF_circuit(mol)
    #circ = get_wtf_circit(mol)
    circ = get_ry_HEA_circit(mol, 3)
    #circ = get_uccsd_circuit(mol)
    print('[circ]')
    print('   n_qubits:', circ.n_qubits)
    print('   n_gates:', len(circ))
    print('   n_params:', len(circ.params_name))
    print(circ)
    fmin = 99999
    pr = None
    if len(circ.params_name):
        for _ in range(10):
            fval, pr_new = get_best_params(circ, ham, method='Adam', init='randn')
            if fval < fmin:
                fmin = fval
                pr = pr_new
    #circ, pr = prune_circuit(circ, pr)
    print('   n_gates (pruned):', len(circ))
    print('   n_params (pruned):', len(circ.params_name))
    print(circ)
    print('[params]:', repr(pr))
    pr_empty = ParameterResolver(dict(zip(circ.params_name, np.zeros(len(circ.params_name))))) if pr is not None else None

    ''' Simulator '''
    from mindquantum.simulator import Simulator as OriginalSimulator
    #sim = OriginalSimulator('mqvector', mol.n_qubits)
    sim = Simulator('mqvector', mol.n_qubits)

    if not isinstance(sim, Simulator):
        sim: OriginalSimulator
        exp = sim.get_expectation(Hamiltonian(ham), circ, pr=pr).real
        print('exp (full ham):', exp)
        if pr_empty is not None:
            exp = sim.get_expectation(Hamiltonian(ham), circ, pr=pr_empty).real
            print('exp (full ham, zero param):', exp)
        print('exp (per-term):')
        for idx, (coeff, ops) in enumerate(split_ham):
            exp = sim.get_expectation(Hamiltonian(ops), circ, pr=pr).real
            print('coeff=', coeff, 'term=', ops, 'exp=', exp)

    ''' Measure '''
    SHOTS = 1000
    USE_EXP_FIX = False
    N_MEAS = 1 if USE_EXP_FIX else 1

    # best local case:
    # E_ref: -1.7572561834789797
    # E_vqc: -2.15495
    # best_param: [
    #     d0_q0: -0.172711,
    #     d0_q6: 0.015735,
    #     d0_q7: -0.007540,
    #     d1_q5: -0.000001,
    #     d1_q6: 1.570925,
    #     d1_q7: 0.005027,
    #     d2_q2: 0.000081,
    #     d2_q5: 0.000054,
    #     d2_q6: 0.004083,
    #     d2_q7: 0.019818,
    #     d3_q1: -0.000007,
    #     d3_q5: 0.000256,
    #     d3_q6: -1.570796,
    #     d3_q7: 0.002514
    # ]
    if pr_empty is not None:
        result_ref = const + get_min_exp(sim, circ, pr_empty, split_ham, shots=SHOTS, n_repeat=N_MEAS, use_exp_fix=USE_EXP_FIX)
        print('result_ref:', result_ref)
    else:
        result_ref = mol.hf_energy

    result_vqc = const + get_min_exp(sim, circ, pr, split_ham, shots=SHOTS, n_repeat=N_MEAS, use_exp_fix=USE_EXP_FIX)
    
    # Reference state error mitigation from https://pubs.acs.org/doi/10.1021/acs.jctc.2c00807
    return result_vqc + (mol.hf_energy - result_ref)
