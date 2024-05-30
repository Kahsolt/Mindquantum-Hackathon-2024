from typing import Dict, Tuple

import numpy as np
from numpy import ndarray

from utils.path import LOG_PATH
from utils.lookup_table import load_lookup_table

lookup_table = load_lookup_table(LOG_PATH / 'lookup_table-iter=1400.json')


def ave_D(Jc, nq):      # average degree
    return 2 * len(Jc) / nq

def order(Jc):          # graph order
    return max([len(key) for key in Jc.keys()])

def trans_gamma(gammas, D):
    # Eq. 10 from arXiv:2201.11785, without 1/|w|
    return gammas * np.arctan(1 / np.sqrt(D - 1)) 

def rescale_factor_arXiv_2201_11785(Jc):
    # Eq. 10 from arXiv:2201.11785, i.e. 1/Σ|w|
    keys_len = {}
    for key in Jc.keys():
        if len(key) in keys_len:
            keys_len[len(key)] += 1
        else:
            keys_len[len(key)] = 1
    norm = 0
    for key, val in Jc.items():
        norm += abs(val/keys_len[len(key)])
    return 1 / norm

def rescale_factor(Jc):
    # Eq. 87 from arXiv:2305.15201, i.e. 1/sqrt(Σw**2)
    keys_len = {}
    for key in Jc.keys():
        if len(key) in keys_len:
            keys_len[len(key)] += 1
        else:
            keys_len[len(key)] = 1
    norm = 0
    for key, val in Jc.items():
        norm += val**2/keys_len[len(key)]
    norm = np.sqrt(norm)
    return 1 / norm


def main_baseline(Jc_dict:Dict[Tuple[int], float], p:int, Nq:int=12) -> Tuple[ndarray, ndarray]:
    '''
        The main function you need to change!!!
    Args:
        Jc_dict (dict): the ising model
        p (int): the depth of qaoa circuit
    Returns:
        gammas (Union[numpy.ndarray, List[float]]): the gamma parameters, the length should be equal to depth p.
        betas (Union[numpy.ndarray, List[float]]): the beta parameters, the length should be equal to depth p.
    '''
    D = ave_D(Jc_dict, Nq)
    k = order(Jc_dict)
    import csv
    # Read the parameters of infinite size limit, and the maximum order is 6 surported here.
    k = min(k, 6)
    # const data sheet from arXiv:2110.14206 Tbl. 4 & 5
    with open('utils/transfer_data.csv', 'r') as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            if row[0] == str(k):
                new_row = [item for item in row if item != '']
                length = len(new_row)
                if length == 3 + 2 * p:
                    gammas = np.array([float(new_row[i]) for i in range(3, 3 + p)] )
                    betas = np.array([float(new_row[i]) for i in range(3 + p, 3 + 2 * p)])
    # rescale the parameters for specific case
    gammas = trans_gamma(gammas, D)
    factor = rescale_factor(Jc_dict)
    return gammas * factor, betas

def main(Jc_dict:Dict[Tuple[int], float], p:int, Nq:int=12, rescaler:float=1.275) -> Tuple[ndarray, ndarray]:
    global lookup_table

    D = ave_D(Jc_dict, Nq)
    k = order(Jc_dict)
    k = min(k, 6)
    params = lookup_table[p][k]
    gammas, betas = np.split(params, 2)
    gammas = trans_gamma(gammas, D)
    factor = rescale_factor(Jc_dict) * rescaler
    return gammas * factor, betas
