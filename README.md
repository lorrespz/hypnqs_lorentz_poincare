# New non-Euclidean neural quantum states from additional types of hyperbolic recurrent neural networks

This repo contains the codes and trained neural network weights for the work 2604.24337 (https://arxiv.org/2604.24337v1). In this work, we construct three new variants of hyperbolic neural quantum states (NQS): Poincare RNN, Lorentz RNN, Lorentz GRU, alongside the previously introduced Poincare GRU - the first type of non-Euclidean hyperbolic NQS in the literature. We benchmarked the performances of all four hyperbolic NQS against their Euclidean RNN/GRU counterparts in the quantum many-body settings of Heisenberg J1J2 and J1J2J3 models. Due to the hierarchical structure of these Hamiltonian systems in the form of various competing next nearest neighbor interactions, hyperbolic NQS were shown to definitively outperform Euclidean NQS. More details can be found in arxiv:2604.24337.

- `1d_j1j2_inference_RNN_architecture.ipynb`: Inference notebook for the Heisenberg J1J2 model showing the performances of three RNN variants (out of the 6 NQS ansatzes): Euclidean RNN, Poincare RNN, Lorentz RNN, at four different couplings J2=0.0, 0.2, 0.5, 0.8.
  
- `1d_j1j2_inference_GRU_architecture.ipynb`: Inference notebook for the Heisenberg J1J2 model showing the performances of the three GRU variants (out of the 6 NQS ansatzes): Euclidean GRU, Poincare GRU, Lorentz GRU at four different couplings J2=0.0, 0.2, 0.5, 0.8.

- `1d_j1j2j3_inference_RNN_architecture.ipynb`:  Inference notebook for the Heisenberg J1J2J3 model showing the performances of the three RNN variants (out of the 6 NQS ansatzes): Euclidean RNN, Poincare RNN, Lorentz RNN at four different couplings (J2,J3)=(0.0,0.5), (0.2,0.2),  (0.2,0.5), (0.5,0.2).

- `1d_j1j2j3_inference_GRU_architecture.ipynb`:  Inference notebook for the Heisenberg J1J2J3 model showing the performances of the three GRU variants (out of the 6 NQS ansatzes): Euclidean GRU, Poincare GRU, Lorentz GRU at four different couplings (J2,J3)=(0.0,0.5), (0.2,0.2),  (0.2,0.5), (0.5,0.2).

- `utility_poincare`: contains the Pytorch codes used to construct Poincare RNN/GRU NQS. These are constructed from scratch, with all necessary mathematical operations (Mobius addition, multiplication, nonlinear activation, parallel transport, exponential/logarithm maps) in the Poincare disk defined in the file `util_torch_poincare.py`.
  
- `utility_lorentz`: contains the Pytorch codes used to construct Lorentz RNN/GRU NQS. These are constructed using the predefined mathematical operations (addition, multiplication, parallel transport, exp/log maps) in Lorentz hyperboloid using the locally saved, modified version of `hypercore` (https://github.com/Graph-and-Geometric-Learning/HyperCore/tree/main/hypercore) files `hypercore/manifolds/lorentzian.py` and `hypercore/manifolds/lmath.py`. In particular, the original version of `lorentzian.py` (last accessed from `hypercore` github on Thu, Apr 30, 2026), contains a typo in the function `mobius_add`, `expmap(v,x)` which should be `expmap(x,v)`.  This typo caused a cascade of NaN errors in the earlier trainings of LorentzGRU/RNN, which disappeared after the typo was fixed. 
  
- `J1J2` contains the trained weights of the 6 types of NQS used in the work: Euclidean/Poincare/Lorentz RNN/GRU for the Heisenberg J1J2 model at four different J2 couplings. A few training Jupyter notebooks showing the training at one particular $J_2$ coupling case are included.
  
-  `J1J2J3` contains the trained weights of the 6 types of NQS used in the work: Euclidean/Poincare/Lorentz RNN/GRU for the Heisenberg J1J2J3 model at four different (J2,J3) couplings. A few training Jupyter notebooks showing the training at one particular $(J_2, J_3)$ coupling case are included.

