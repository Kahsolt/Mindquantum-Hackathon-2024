# parse pauli string
R_str = Regex('\[([XYZ\d ]+)\]')
terms: List[Tuple[float, str]] = []
for coeff, ops in split_ham:
  terms.append((coeff, R_str.findall(str(ops))[0]))
# fix phase raised by HF_ciruit when replace Y -> X
phases: List[int] = []
hf_state = '11110000'   # reversed of the bit order
for i, (coeff, string) in enumerate(terms):
  phase = 1
  for seg in string.split(' '):
    sym = seg[0]
    if sym != 'Y': continue
    qid = int(seg[1:])
    phase *= 1j if hf_state[qid] == '1' else -1j
  phases.append(phase.real)
# approx YY ~= XX, hence aggregate the coeffs
string_coeff: Dict[str, List[float]] = {}
for i, (coeff, string) in enumerate(terms):
  string_XY = string.replace('Y', 'X')
  if string_XY not in string_coeff:
    string_coeff[string_XY] = []
  string_coeff[string_XY].append(coeff * phases[i])
terms_agg: Dict[str, float] = {k: sum(v) for k, v in string_coeff.items()}
# convert to SplitHam
split_ham_combined = [[v, QubitOperator(k)] for k, v in terms_agg.items()]
