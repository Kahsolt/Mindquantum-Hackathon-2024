import numpy as np


def main(Jc_dict, p, Nq=14):
    '''
        The main function you need to change!!!
    Args:
        Jc_dict (dict): the ising model
        p (int): the depth of qaoa circuit
    Returns:
        gammas (Union[numpy.ndarray, List[float]]): the gamma parameters, the length should be equal to depth p.
        betas (Union[numpy.ndarray, List[float]]): the beta parameters, the length should be equal to depth p.
    '''
    D = ave_D(Jc_dict,Nq)
    k = order(Jc_dict)
    import csv
    # Read the parameters of infinite size limit, and the maximum order is 6 surported here.
    k = min(k,6)
    with open('utils/transfer_data.csv', 'r') as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            if (row[0]) == str(k):
                new_row = [item for item in row if item != '']
                length = len(new_row)
                if length == 3+2*p:
                    gammas = np.array([float(new_row[i]) for i in range(3,3+p)] )
                    betas = np.array([float(new_row[i]) for i in range(3+p,3+2*p)])
    # rescale the parameters for specific case
    gammas = trans_gamma(gammas, D)
    factor = rescale_factor(Jc_dict)
    return gammas*factor, betas
        
def trans_gamma(gammas, D):
    return gammas*np.arctan(1/np.sqrt(D-1)) 

def rescale_factor(Jc):
    '''
    Get the rescale factor, a technique from arXiv:2305.15201v1
    '''
    import copy
    Jc_dict=copy.deepcopy(Jc)
    keys_len={}
    for key in Jc_dict.keys():
        if len(key) in keys_len:
            keys_len[len(key)]+=1
        else:
            keys_len[len(key)]=1
    norm=0
    for key,val in Jc_dict.items():        
        norm+= val**2/keys_len[len(key)]
    norm = np.sqrt(norm)
    return  1/norm    

def ave_D(Jc,nq):
    return 2*len(Jc)/nq

def order(Jc):
    return max([len(key)  for key in Jc.keys()])
        