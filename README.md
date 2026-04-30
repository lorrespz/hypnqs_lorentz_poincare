# New non-Euclidean neural quantum states from additional types of hyperbolic recurrent neural networks

This repo contains the codes and trained neural network weights for the work 2604.24337 (https://arxiv.org/2604.24337v1). 

- `utility_poincare': contains the Pytorch codes used to construct Poincare RNN/GRU NQS. These are constructed from scratch, with all necessary mathematical operations (Mobius addition, multiplication, nonlinear activation, parallel transport, exponential/logarithm maps) in the Poincare disk defined in the file `util_torch_poincare.py'.
- `utility_lorentz': contains the Pytorch codes used to construct Lorentz RNN/GRU NQS. These are constructed using the locally saved, modified version of `hypercore' files `lorentzian.py' and `lmath.py'.
- `J1J2' contains the trained weights of the 6 types of NQS used in the work: Euclidean/Poincare/Lorentz RNN/GRU for the Heisenberg J1J2 model at four different J2 couplings. A few training Jupyter notebooks showing the training at one particular $J_2$ coupling case are included.
-  `J1J2J3' contains the trained weights of the 6 types of NQS used in the work: Euclidean/Poincare/Lorentz RNN/GRU for the Heisenberg J1J2J3 model at four different (J2,J3) couplings. A few training Jupyter notebooks showing the training at one particular $(J_2, J_3)$ coupling case are included.

This repo is under active construction. Updates will be regularly made. 
