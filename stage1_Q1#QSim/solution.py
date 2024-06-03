# NOTE: 答题时只能修改此文件，整合一切代码，因为评测机只会上传这单个脚本！！
# 一点也不能改 solution() 函数的签名，例如不能给 molecule 加类型注解，因为评测机做了硬编码的校验！！

if 'env':
    import os, sys
    sys.path.append(os.path.abspath(__file__))

    import warnings
    warnings.simplefilter(action='ignore', category=UserWarning)
    warnings.simplefilter(action='ignore', category=RuntimeWarning)
    warnings.simplefilter(action='ignore', category=DeprecationWarning)

from re import compile as Regex
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
from mindquantum import MQAnsatzOnlyLayer
from mindspore.nn.optim import SGD, Adam
from mindspore.nn.wrap.cell_wrapper import TrainOneStepCell

from simulator import HKSSimulator
from utils import generate_molecule, get_molecular_hamiltonian

R_str = Regex('\[([XYZ\d ]+)\]')
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

def get_pchc_circuit(mol, depth:int=1, order:str='sd') -> Circuit:
    ''' inspired by CHC from arXiv:2003.12578, we borrow the main structure and make it fully parametrized '''
    nq_half = mol.n_qubits // 2
    circ = Circuit()
    for d in range(depth):
        # layer idx
        l = 0
        # hardamard
        for i in range(mol.n_qubits):
            circ += RY(f'd{d}_l{l}_q{i}').on(i)
        l += 1
        # single excitation
        if 's' in order:
            for j in range(1, nq_half):
                for i in [0, nq_half]:
                    p, q = i, i + j
                    circ += CNOT.on(q, p)
                    circ += RZ(f'd{d}_l{l}_q{q}_c').on(q)
                    circ += CNOT.on(q, p)
                    circ += RY(f'd{d}_l{l}_q{p}').on(p)
                    circ += RY(f'd{d}_l{l}_q{q}').on(q)
                l += 1
        # double excitation
        if 'd' in order:
            for i in range(1, nq_half):
                for j in range(1, nq_half):
                    p, q, r, s = 0, i, nq_half, nq_half + j
                    circ += CNOT.on(q, p)
                    circ += CNOT.on(r, q)
                    circ += CNOT.on(s, r)
                    circ += RZ(f'd{d}_l{l}_q{s}_c').on(s)
                    circ += CNOT.on(s, r)
                    circ += CNOT.on(r, q)
                    circ += CNOT.on(q, p)
                    circ += RY(f'd{d}_l{l}_q{p}').on(p)
                    circ += RY(f'd{d}_l{l}_q{q}').on(q)
                    l += 1
    # HF appended
    return circ + get_HF_circuit(mol)

def get_hae_ry_circit(mol, depth:int=1) -> Circuit:
    ''' standard HAE(RY) '''
    circ = Circuit()
    for i in range(mol.n_qubits):
        circ += RY(f'd0_q{i}').on(i)
    for j in range(1, depth+1):
        for i in range(0, mol.n_qubits-1):
            circ += CNOT.on(i+1, i)
        for i in range(mol.n_qubits):
            circ += RY(f'd{j}_q{i}').on(i)
    return circ + get_HF_circuit(mol)

def get_hae_ry_compact_circit(mol, depth:int=1) -> Circuit:
    ''' impl from https://github.com/liwt31/QC-Contest-Demo
    The compact-HEA(RY) circuit is like:
        --RY--o-----RY--
              |
        --RY--x--o--RY--
                 |
        --RY--o--x--RY--
              |
        --RY--x-----RY--
    where two layers of CNOT placed zig-zagly, and RY is inserted around every CNOT block
    '''
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
    # HF appended at last, for reference error mitigation
    return circ + get_HF_circuit(mol)

def get_cnot_centric_circit(mol, depth:int=1) -> Circuit:
    '''The CNOT-centric circuit is like:
        --RY--o---------RY--
              |
        --RY--x--RY--o--RY--
                     |
        --RY--o--RY--x--RY--
              |
        --RY--x---------RY--
    where two layers of CNOT placed zig-zagly, and RY is inserted around each CNOT gate
    '''
    circ = Circuit()
    # flat RY
    for i in range(mol.n_qubits):
        circ += RY(f'd0_q{i}').on(i)
    for d in range(1, depth+1):
        # zig-zag CNOT with RY bridge
        qid = 0
        while qid + 1 < mol.n_qubits:   # even control: CNOT(1, 0)
            circ += CNOT.on(qid+1, qid)
            qid += 2
        qid = 1
        while qid + 1 < mol.n_qubits:   # odd control: CNOT(2, 1)
            circ += RY(f'd{d}_q{qid}_mid').on(qid)
            circ += RY(f'd{d}_q{qid+1}_mid').on(qid+1)
            circ += CNOT.on(qid+1, qid)
            qid += 2
        # flat RY
        for i in range(mol.n_qubits):
            circ += RY(f'd{d}_q{i}').on(i)
    return circ + get_HF_circuit(mol)


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
    ham = QubitOperator('', const)
    for coeff, ops in terms:
        string = R_str.findall(str(ops))[0]
        ham += QubitOperator(string, coeff)
    return ham

def prune_hamiltonian(split_ham:List[PauliTerm]) -> List[PauliTerm]:
    from mindquantum._math.ops import QubitOperator as QubitOperator_

    def filter_Z_only(ops:QubitOperator) -> bool:   # for HF_circuit
        for term in QubitOperator_.get_terms(ops):
            for qubit, symbol in term[0]:
                if symbol.name in ['X', 'Y']:
                    return False
        return True

    def combine_XY(split_ham:List[PauliTerm]) -> List[PauliTerm]:
        # parse pauli string of each term
        terms: List[Tuple[float, str]] = []
        for coeff, ops in split_ham:
            string = R_str.findall(str(ops))[0]
            terms.append((coeff, string))
        # ref: https://github.com/liwt31/QC-Contest-Demo
        # fix phase of Y raised by HF_ciruit
        phases: List[int] = []
        hf_state = '11110000'   # reversed of the bit order
        for i, (coeff, string) in enumerate(terms):
            phase = 1
            for seg in string.split(' '):
                sym = seg[0]
                if sym != 'Y': continue
                qid = int(seg[1:])
                if hf_state[qid] == '1':
                    phase *= 1j
                else:
                    phase *= -1j
            phases.append(phase.real)
        assert len(phases) == len(terms)
        # presuming X-Y is the same, aggregate the coeffs
        string_coeff: Dict[str, List[float]] = {}
        for i, (coeff, string) in enumerate(terms):
            string_XY = string.replace('X', 'Y')
            if string_XY not in string_coeff:
                string_coeff[string_XY] = []
            string_coeff[string_XY].append(coeff * phases[i])
        terms_agg: Dict[str, float] = {k: np.sum(v) for k, v in string_coeff.items()}
        # convert to SplitHam
        split_ham_combined = [[v, QubitOperator(k)] for k, v in terms_agg.items()]
        return split_ham_combined

    #split_ham = [it for it in split_ham if filter_Z_only(it[1])]
    split_ham = combine_XY(split_ham)
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
    return round(exp / shots, int(np.ceil(np.log10(shots))))

def estimate_rescaler(mol) -> float:
    # 测量门的比特翻转信道会使得测量值极差变小，尝试估计一下缩放比
    sim = HKSSimulator('mqvector', n_qubits=mol.n_qubits)
    circ = Circuit()
    circ += X.on(0)
    ops = QubitOperator('Z0')
    exp_GT = -1
    exp_actual = measure_single_ham(sim, circ, None, ops, 10000)
    return exp_actual / exp_GT

def get_min_exp(sim:Simulator, circ:Circuit, pr:ParameterResolver, split_ham:List[PauliTerm], shots:int=100, n_repeat:int=10, rescaler:float=1.0, use_exp_fix:bool=False) -> float:
    result_min = 99999
    for _ in range(n_repeat):
        result = 0.0
        with SingleLoopProgress(len(split_ham), '哈密顿量测量中') as bar:
            for idx, (coeff, ops) in enumerate(split_ham):
                #n_prec = int(np.ceil(np.log10(1/abs(coeff))))
                #var_shots = shots * 10**(n_prec-1)
                var_shots = shots
                exp = measure_single_ham(sim, circ, pr, ops, var_shots)
                if use_exp_fix:     # for HF_circuit only
                    # FIXME: when circ is non-entangled, and gates are all X
                    # the BitFlip noise on measure gate could be cheaty moved out :)
                    if   exp > +0.1: exp = +1
                    elif exp < -0.1: exp = -1
                    else:            exp = 0
                print('coeff=', coeff, 'term=', ops, 'exp=', exp)
                result += exp * coeff / rescaler
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
    split_ham = prune_hamiltonian(split_ham)
    print('  n_terms (pruned):', len(split_ham))
    #ham = combine_hamiltonian(const, split_ham)

    ''' Circuit & Params '''
    #circ = get_HF_circuit(mol)                 # concrete noisy baseline
    #circ = get_uccsd_circuit(mol)              # concrete noiseless topline
    #circ = get_pchc_circuit(mol)               # seems not work
    circ = get_hae_ry_circit(mol, 3)
    #circ = get_hae_ry_compact_circit(mol, 3)   # a bit worse than the standard HEA(RY) =_=||
    #circ = get_cnot_centric_circit(mol, 2)     # seems not work
    print('[circ]')
    print('   n_qubits:', circ.n_qubits)
    print('   n_gates:', len(circ))
    print('   n_params:', len(circ.params_name))
    print(circ)
    N_OPTIM_TRIAL = 3
    fmin = 99999
    fval = None
    pr = None
    if len(circ.params_name):
        for _ in range(N_OPTIM_TRIAL):
            fval, pr_new = get_best_params(circ, ham, method='Adam', init='randn')
            if fval < fmin:
                fmin = fval
                pr = pr_new
    print('   best fval:', fval)
    if pr is not None:
        circ, pr = prune_circuit(circ, pr)
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

    rescaler = estimate_rescaler(mol)
    if pr_empty is not None:
        result_ref = const + get_min_exp(sim, circ, pr_empty, split_ham, shots=SHOTS, n_repeat=N_MEAS, rescaler=rescaler, use_exp_fix=USE_EXP_FIX)
    else:
        result_ref = mol.hf_energy
    print('result_ref:', result_ref)
    print('ref_shift:', mol.hf_energy - result_ref)

    result_vqc = const + get_min_exp(sim, circ, pr, split_ham, shots=SHOTS, n_repeat=N_MEAS, rescaler=rescaler, use_exp_fix=USE_EXP_FIX)
    print('result_vqc:', result_vqc)
    
    # Reference state error mitigation from https://pubs.acs.org/doi/10.1021/acs.jctc.2c00807
    return result_vqc + (mol.hf_energy - result_ref)
