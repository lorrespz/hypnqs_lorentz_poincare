# This is part of the work arxiv:2607.24337 by HL Dao. 
# This code defines the LorentzRNN and LorentzGRU class (for J1J2 model)
# The LorentzGRU class in this file uses the manifold update method

from util_loading import *

# use 20.0 for LorentzRNN double clamp case + LorentzGRU (all cases)
#norm_spatial = 20
# use 18.0 for LorentzRNN single clamp case 
norm_spatial = 18.0

def project_lorentz_manual(x, k=1.0):
    spatial = x[..., 1:]
    
    # Clamp to prevent INF, not to restrict the manifold
    # Float32 max is ~3e38, but sinh/cosh blow up much earlier. 
    # Clamping spatial components to 15-20 is usually plenty for stability.
    spatial = torch.clamp(spatial, min=-norm_spatial, max=norm_spatial) 
    
    spatial_norm_sq = torch.sum(spatial**2, dim=-1, keepdim=True)
    
    # Based on lmath.py _inner0: -x0*sqrt(k) = -k => x0 = sqrt(k) at origin
    # Constraint: x0^2 - spatial^2 = k  => x0 = sqrt(spatial^2 + k)
    new_x0 = torch.sqrt(spatial_norm_sq + k)
    
    return torch.cat([new_x0, spatial], dim=-1)

# ---  LorentzRNN definitions ---
"""
    Hyperboloid Manifold class: for x in (d+1)-dimension Euclidean space
    -x0^2 + x1^2 + x2^2 + … + xd = -c, x0 > 0, c > 0
    negative curvature - 1 / c
    """
class LorentzRNN(nn.Module):
    def __init__(self, input_dim, num_units, spatial_clamp = 2.0, k=1.0):
            super().__init__()
            self.num_units = num_units
            self.k = k
            self.k_tensor= torch.as_tensor(self.k, dtype=torch.float32)
            self.name = f'LorentRNN_{num_units}'
            self.spatial_clamp = spatial_clamp

            self.register_buffer('max_h0', torch.tensor(1.0))            # Track h0 (the "Lorentz Factor")
            self.register_buffer('max_violation', torch.tensor(0.0))     # Track drift off the hyperboloid
            self.register_buffer('max_spatial_norm', torch.tensor(0.0))  # Track Euclidean spread

            self.W = nn.Parameter(torch.empty(num_units, num_units))
            self.U = nn.Parameter(torch.empty(num_units, input_dim))
            self.b = nn.Parameter(torch.zeros(num_units))
            self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W, gain=0.01)
        nn.init.xavier_uniform_(self.U, gain=0.01)

    def one_rnn_transform(self, W, state, U, x_hyp, b):
        Wh = project_lorentz_manual(hc_math.mobius_matvec(W, state))
        Ux = project_lorentz_manual(hc_math.mobius_matvec(U, x_hyp))
        self._check_manifold_violation(Wh, "Wh")
        self._check_manifold_violation(Ux, "Ux")
        # Map bias to manifold  (bias is Euclidean params optimized with Adam)  
        b_hyp = project_lorentz_manual(hc_math.expmap0(b,  k=self.k_tensor)) 
        # Combine on the manifold
        Wh_Ux = project_lorentz_manual(hc_math.mobius_add(Wh, Ux))
        self._check_manifold_violation(Wh_Ux, "Wh_Ux")
        Wh_Ux_b= project_lorentz_manual(hc_math.mobius_add(Wh_Ux , b_hyp))
        self._check_manifold_violation(Wh_Ux_b, "Wh_Ux_b")
        res = project_lorentz_manual(Wh_Ux_b)
        return res

    def forward(self, inputs, state):
        # 1. Map Euclidean input to manifold
        hyp_x = project_lorentz_manual(hc_math.expmap0(inputs, k=self.k_tensor))
        # Need spatial clamp to make LorentzRNN behave?
        # First spatial clamp
        state = self._apply_spatial_clamp(state)
        #state = project_lorentz_manual(state)

        #2. Apply the RNN transformation
        new_h = self.one_rnn_transform(self.W, state, self.U, hyp_x, self.b)
        # Second clamp (for single clamp, comment this line below out)
        new_h = self._apply_spatial_clamp(new_h)
        self._check_manifold_violation(new_h, "Final state")

        #3.  Final NaN check
        if torch.isnan(new_h).any():
             print(f"NaN triggered! Max spatial norm: {torch.max(torch.sum(new_h[..., 1:]**2, dim=-1))}")

        #4. Check all parameters relating the Lorentz geometry:
        with torch.no_grad():
            # a. Temporal Component (h0) - tells us how "curved" the state is
            # if h0=1.0 or close to 1.0: practically Euclidean space (near the origin)
            h0 = new_h[..., 0]
            self.max_h0.copy_(torch.max(h0).detach())

            # b. Spatial Spread
            spatial_norm = torch.norm(new_h[..., 1:], p=2, dim=-1)
            self.max_spatial_norm.copy_(torch.max(spatial_norm).detach())

            # c. Manifold Violation - tells us if the J2 frustration is breaking the math
            # Calculation: |-h0^2 + ||spatial||^2 + k|
            violation = torch.abs(-(h0**2) + (spatial_norm**2) + self.k)
            self.max_violation.copy_(torch.max(violation).detach())

        return new_h, new_h

    def _apply_spatial_clamp(self, x, eps=1e-5):
        # More aggressive scaling than 'projection_lorentz_manual'
        # 1. Force the hidden state to stay within a strict radius to avoid numerical explosion
        # We must constrain the norm BEFORE it hits the exponential functions.
        max_norm = (self.spatial_clamp - eps) / (self.k**0.5)
        
        # Calculate the spatial norm (sum of squares of all but the first component)
        spatial_norm = torch.norm(x[..., 1:], p=2, dim=-1, keepdim=True)
        
        # If the spatial norm is too high, scale it down
        mask = (spatial_norm > max_norm).float()
        scale = (max_norm / (spatial_norm + 1e-10)) * mask + (1.0 - mask)
        
        # Apply scaling to spatial components
        x_spatial = x[..., 1:] * scale
        x_0 = torch.sqrt(self.k + torch.sum(x_spatial**2, dim=-1, keepdim=True))
        
        return torch.cat([x_0, x_spatial], dim=-1)

    def _check_manifold_violation(self, x, name=""):
        # The Hyperboloid model satisfies B(x, x) = -c
        # We calculate the deviation from -c
        x0 = x[..., 0:1]
        spatial_sq = torch.sum(x[..., 1:]**2, dim=-1, keepdim=True)
        # B(x, x) = -x0^2 + ||x_spatial||^2
        # Violation = |B(x, x) - (-c)| = |-x0^2 + ||x_spatial||^2 + c|
        violation = torch.abs(-x0**2 + spatial_sq + self.k)
        
        if torch.isnan(violation).any():
            print(f"CRITICAL: NaN detected at {name}")
            return True
            
        if violation.max() > 1e-2: # Threshold for "significant" drift
            print(f"Warning: High manifold violation at {name}: {violation.max().item():.6f}")
        return False

    def get_manifold_parameters(self):
        # We CHOOSE these to be Euclidean because they act as transformation matrices
        eucl_vars = [self.W, self.U]
        
        # We CHOOSE this to be Hyperbolic because it represents a point 
        # being projected onto the manifold in the forward pass.
        hyp_vars = [self.b]
        
        return eucl_vars, hyp_vars

class LorentzGRU(nn.Module):

    def __init__(self, input_dim, num_units, spatial_clamp =4.0, k=1.0):
        super().__init__()
        self.num_units = num_units
        self.k = k
        self.name = f'LorentGRU_{num_units}'
        self.spatial_clamp=spatial_clamp
        self.k_tensor = torch.as_tensor(self.k,  dtype=torch.float32)

        self.register_buffer('max_h0', torch.tensor(1.0))            # Track h0 (the "Lorentz Factor")
        self.register_buffer('max_violation', torch.tensor(0.0))     # Track drift off the hyperboloid
        self.register_buffer('max_spatial_norm', torch.tensor(0.0))  # Track Euclidean spread


        for gate in ['z', 'r', 'h']:
            setattr(self, f'W{gate}', nn.Parameter(torch.empty(num_units, num_units)))
            setattr(self, f'U{gate}', nn.Parameter(torch.empty(num_units, input_dim)))
            setattr(self, f'b{gate}', nn.Parameter(torch.zeros(num_units)))

        self.reset_parameters()

    def reset_parameters(self):
        for gate in ['z', 'r', 'h']:
            nn.init.xavier_uniform_(getattr(self, f'W{gate}'), gain=1.0)
            nn.init.xavier_uniform_(getattr(self, f'U{gate}'), gain=1.0)

    def one_rnn_transform(self, W, state, U, x_hyp, b, gate):
        Wh = project_lorentz_manual(hc_math.mobius_matvec(W, state))
        self._check_manifold_violation(Wh, f"Wh_{gate}")
        Ux = project_lorentz_manual(hc_math.mobius_matvec(U, x_hyp))
        self._check_manifold_violation(Ux, f"Ux_{gate}")
        # Map bias to manifold  (bias is Euclidean params optimized with Adam) 
        #65 is a the event horizon for the exp() function for 32-bit floating point math: e^65 = 1.6x10^28
        #b_hyp = hc_math.expmap0(torch.clamp(b,-65,65),  k=self.k_tensor) 
        b_hyp = project_lorentz_manual(hc_math.expmap0(b,  k=self.k_tensor)) 
        # Combine on the manifold
        Wh_Ux = project_lorentz_manual(hc_math.mobius_add(Wh, Ux))
        self._check_manifold_violation(Wh_Ux, f"Wh_Ux_{gate}")
        Wh_Ux_b= project_lorentz_manual(hc_math.mobius_add(Wh_Ux, b_hyp))
        return Wh_Ux_b


    def forward(self, inputs, state):
        # 1. Map inputs to the manifold once
        hyp_x = project_lorentz_manual(hc_math.expmap0(inputs,k=self.k_tensor))
        # First spatial clamp
        state = self._apply_spatial_clamp(state)
        #state = project_lorentz_manual(state)
        #--------------------------------------------------------------------------------
        # 2. Gate Calculations (Strictly Hyperbolic)
        #--------------------------------------------------------------------------------
        # a. Update Gate (z)
        #--------------------------------------------------------------------------------
        z_manifold = self.one_rnn_transform(self.Wz, state, self.Uz, hyp_x, self.bz, gate = 'z_manifold')
        self._check_manifold_violation(z_manifold, "z_manifold")
        z = torch.sigmoid(hc_math.logmap0(z_manifold, k=self.k_tensor, is_tan_normalize=False))
        #--------------------------------------------------------------------------------
        # b. Reset Gate (r)
        #--------------------------------------------------------------------------------
        r_manifold = self.one_rnn_transform(self.Wr, state, self.Ur, hyp_x, self.br, gate='r_manifold')
        self._check_manifold_violation(r_manifold, "r_manifold")
        r = torch.sigmoid(hc_math.logmap0(r_manifold, k=self.k_tensor, is_tan_normalize=False))
        #--------------------------------------------------------------------------------
        # 3. Candidate State (h_tilde)
        #--------------------------------------------------------------------------------
        # h_tilde = Wh(r_i * h_{i-1}) + Uh*x_i + b_h
        #--------------------------------------------------------------------------------
        r_point_h = project_lorentz_manual(hc_math.mobius_scalar_mult(r,state))
        h_tilde = self.one_rnn_transform(self.Wh, r_point_h, self.Uh, hyp_x, self.bh, gate = 'h_manifold')    
        self._check_manifold_violation(h_tilde, "h_tilde")
        #--------------------------------------------------------------------------------
        # 4. Update equation: h_{i-1} + z_i*(-h_{i-1} + h_tilde_i), 
        # update_step = z_i*(-h_{i-1} + h_tilde_i), state_{i-1} = h_{i-1}, 
        # state_i = state_{i-1} + update_step
        #--------------------------------------------------------------------------------
        # Option 1: Manifold update
        #--------------------------------------------------------------------------------
        # first, create the correct Lorentz negative state: it's not simply -state
        state_inv = state.clone()
        # Negate only the spatial component, while the x0 component stays the same 
        # state_inv \oplus state = Lorentz_origin (1,0,0,...)
        state_inv[...,1:] = -state_inv[..., 1:]
        minus_h_oplus_htilde = project_lorentz_manual(hc_math.mobius_add(state_inv, h_tilde))
        # combine update_step and state on manifold
        update_step = project_lorentz_manual(hc_math.mobius_scalar_mult(z,minus_h_oplus_htilde))
        new_h = project_lorentz_manual(hc_math.mobius_add(state, update_step))
        self._check_manifold_violation(new_h, "final_new_h")
        #---------------------------------------------------------------------------------
        #Option 2: Tangent-space update: see j1j2_definitions_tangent_update.py
        #--------------------------------------------------------------------------------

        with torch.no_grad():
            # a. Temporal Component (h0) - tells us how "curved" the state is
            # if h0=1.0 or close to 1.0: practically Euclidean space (near the origin)
            h0 = new_h[..., 0]
            self.max_h0.copy_(torch.max(h0).detach())

            # b. Spatial Spread
            spatial_norm = torch.norm(new_h[..., 1:], p=2, dim=-1)
            self.max_spatial_norm.copy_(torch.max(spatial_norm).detach())

            # c. Manifold Violation - tells us if the J2 frustration is breaking the math
            # Calculation: |-h0^2 + ||spatial||^2 + k|
            violation = torch.abs(-(h0**2) + (spatial_norm**2) + self.k)
            self.max_violation.copy_(torch.max(violation).detach())

        #5. Final NaN check
        if torch.isnan(new_h).any():
             print(f"NaN triggered! Max spatial norm: {torch.max(torch.sum(new_h[..., 1:]**2, dim=-1))}")
        return new_h, new_h

    def _check_manifold_violation(self, x, name=""):
        # The Hyperboloid model satisfies B(x, x) = -c
        # We calculate the deviation from -c
        x0 = x[..., 0:1]
        spatial_sq = torch.sum(x[..., 1:]**2, dim=-1, keepdim=True)
        # B(x, x) = -x0^2 + ||x_spatial||^2
        # Violation = |B(x, x) - (-c)| = |-x0^2 + ||x_spatial||^2 + c|
        violation = torch.abs(-x0**2 + spatial_sq + self.k)
        
        if torch.isnan(violation).any():
            print(f"CRITICAL: NaN detected at {name}")
            return True
            
        if violation.max() > 1e-2: # Threshold for "significant" drift
            print(f"Warning: High manifold violation at {name}: {violation.max().item():.6f}")
        return False

    def _apply_spatial_clamp(self, x, eps=1e-5):
        # More aggressive scaling than 'projection_lorentz_manual'
        # 1. Force the hidden state to stay within a strict radius to avoid numerical explosion
        max_norm = (self.spatial_clamp - eps) / (self.k**0.5)

        # Calculate the spatial norm (sum of squares of all but the first component)
        spatial_norm = torch.norm(x[..., 1:], p=2, dim=-1, keepdim=True)
        
        # If the spatial norm is too high, scale it down 
        mask = (spatial_norm > max_norm).float()
        scale = (max_norm / (spatial_norm + 1e-10)) * mask + (1.0 - mask)
        
        # Apply scaling to spatial components
        x_spatial = x[..., 1:] * scale
        x_0 = torch.sqrt(self.k + torch.sum(x_spatial**2, dim=-1, keepdim=True))
        
        return torch.cat([x_0, x_spatial], dim=-1)

    def get_manifold_parameters(self):
        # Euclidean: Weight matrices
        eucl_vars = [
            self.Wz, self.Uz, 
            self.Wr, self.Ur, 
            self.Wh, self.Uh
        ]
        
        # Hyperbolic: All biases are now manifold points
        hyp_vars = [self.bz, self.br, self.bh]
        
        return eucl_vars, hyp_vars
