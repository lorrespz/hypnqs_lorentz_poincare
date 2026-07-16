# New non-Euclidean neural quantum states from additional types of hyperbolic recurrent neural networks

This repo contains the codes and trained neural network weights for the work 2604.24337 (https://arxiv.org/2604.24337). In this work, we construct three new variants of hyperbolic neural quantum states (NQS): Poincare RNN, Lorentz RNN, Lorentz GRU, alongside the previously introduced Poincare GRU - the first type of non-Euclidean hyperbolic NQS in the literature. We benchmarked the performances of all four hyperbolic NQS against their Euclidean RNN/GRU counterparts in the quantum many-body settings of Heisenberg J1J2 and J1J2J3 models. Due to the hierarchical structure of these Hamiltonian systems in the form of various competing next nearest neighbor interactions, hyperbolic NQS were shown to definitively outperform Euclidean NQS. More details can be found in arxiv:2604.24337.

- `utility_poincare`: contains the Pytorch codes used to construct Poincare RNN/GRU NQS. These are constructed from scratch, with all necessary mathematical operations (Mobius addition, multiplication, nonlinear activation, parallel transport, exponential/logarithm maps) in the Poincare disk defined in the file `util_torch_poincare.py`.
  
- `utility_lorentz`: contains the Pytorch codes used to construct Lorentz RNN/GRU NQS. These are constructed using the predefined mathematical operations (addition, multiplication, parallel transport, exp/log maps) in Lorentz hyperboloid using the locally saved, modified version of `hypercore` (https://github.com/Graph-and-Geometric-Learning/HyperCore/tree/main/hypercore) files `hypercore/manifolds/lorentzian.py` and `hypercore/manifolds/lmath.py`. In particular, the original version of `lorentzian.py` (last accessed from `hypercore` github on Thu, Apr 30, 2026), contains a typo in the function `mobius_add`, `expmap(v,x)` which should be `expmap(x,v)`.  This typo caused a cascade of NaN errors in the earlier trainings of LorentzGRU/RNN, which disappeared after the typo was fixed.
  
- `training_notebook_examples` contain sample training notebooks for both J1J2 and J1J2J3 systems.

### To run the hyperbolic NQS training:

`!git clone https://github.com/lorrespz/hypnqs_lorentz_poincare.git`

- Lorentz NQS (J1J2 system)
`import sys
sys.path.append('/hypnqs_lorentz_poincare/utility_lorentz')
from j1j2_train_loop_lorentz import *`

- Poincare NQS (J1J2 system):
`import sys
sys.path.append('/hypnqs_lorentz_poincare/utility_poincare')
from j1j2_hyprnn_train_loop import *`



