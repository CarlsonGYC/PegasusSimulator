#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Compare PX4Mixer (NumPy) vs TorchMixer (PyTorch) for normal mode, including the
final remapping from [0..1] to [-1..1] that PX4 does at the end of mix().
"""

import numpy as np
import math
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

###############################################################################
# 1) PX4 Mixer (NumPy) Implementation
###############################################################################
def compute_desaturation_gain_np(u, u_min, u_max, desaturation_vector):
    """
    Numpy version of compute_desaturation_gain():
      Summation of (k_min + k_max) from each rotor to unsaturate.
    """
    u = np.ravel(u)

    # Convert scalars to arrays so indexing is valid
    if np.isscalar(u_min) or np.ndim(u_min) == 0:
        u_min = np.full(u.shape, u_min)
    else:
        u_min = np.ravel(u_min)

    if np.isscalar(u_max) or np.ndim(u_max) == 0:
        u_max = np.full(u.shape, u_max)
    else:
        u_max = np.ravel(u_max)

    desaturation_vector = np.ravel(desaturation_vector)

    d_u_plus = u_max - u
    d_u_minus = u_min - u

    k_min = 0.0
    k_max = 0.0

    for i in range(u.size):
        if abs(desaturation_vector[i]) < 1e-6:
            continue

        # If output < min => we can push up
        if u[i] < u_min[i]:
            k = (u_min[i] - u[i]) / desaturation_vector[i]
            if k < k_min:
                k_min = k
            if k > k_max:
                k_max = k

        # If output > max => we can pull down
        if u[i] > u_max[i]:
            k = (u_max[i] - u[i]) / desaturation_vector[i]
            if k < k_min:
                k_min = k
            if k > k_max:
                k_max = k

    return k_min + k_max

def minimize_sat_np(u, u_min, u_max, desaturation_vector):
    """
    Numpy version of minimize_sat():
      1) compute k1 => u_1 = u + k1*vec
      2) compute k2 => final = u + (k1 + 0.5*k2)*vec
    """
    k1 = compute_desaturation_gain_np(u, u_min, u_max, desaturation_vector)
    u_1 = u + k1 * desaturation_vector
    k2 = compute_desaturation_gain_np(u_1, u_min, u_max, desaturation_vector)
    k_opt = k1 + 0.5*k2
    return u + k_opt * desaturation_vector

def mix_yaw_np(m_sp, u, P, u_min, u_max):
    """
    Add yaw in a separate step, then saturate yaw up to +15%,
    then saturate thrust only. If final is bigger => revert.
    """
    m_sp = np.ravel(m_sp)
    m_sp_yaw_only = np.zeros_like(m_sp)
    m_sp_yaw_only[2] = m_sp[2]

    u_p = u + (P @ m_sp_yaw_only.reshape(-1,1)).flatten()

    # saturate yaw => up to +0.15
    u_r_dot = P[:, 2].flatten()
    u_pp = minimize_sat_np(u_p, u_min, u_max + 0.15, u_r_dot)

    # saturate thrust only
    u_T = P[:, 3].flatten()
    u_ppp = minimize_sat_np(u_pp, 0.0, u_max, u_T)

    # revert if final is bigger
    if (u_ppp > u_pp).any():
        u_ppp = u_pp.copy()

    return u_ppp

def normal_mode_np(m_sp, P, u_min, u_max):
    """
    Normal mode in PX4:
      1) remove yaw
      2) saturate thrust (reduce only)
      3) saturate roll
      4) saturate pitch
      5) add yaw + saturations
    """
    m_sp_no_yaw = m_sp.copy()
    m_sp_no_yaw[2, 0] = 0.0
    u = (P @ m_sp_no_yaw).flatten()

    # saturate thrust (reduce only)
    u_T = P[:, 3].flatten()
    u_prime = minimize_sat_np(u, u_min, u_max, u_T)
    if (u_prime > u).any():
        u_prime = u.copy()

    # saturate roll
    u_p_dot = P[:, 0].flatten()
    u_p2 = minimize_sat_np(u_prime, u_min, u_max, u_p_dot)

    # saturate pitch
    u_q_dot = P[:, 1].flatten()
    u_p3 = minimize_sat_np(u_p2, u_min, u_max, u_q_dot)

    # add yaw
    u_final = mix_yaw_np(m_sp, u_p3, P, u_min, u_max)
    return (u, u_final)

class PX4Mixer:
    """
    A NumPy-based class that replicates PX4 normal mode mixing.
    """
    def __init__(self, P, airmode="none"):
        self.P = np.array(P, dtype=np.float32)
        self.airmode = airmode
        self.B = np.linalg.pinv(self.P)

    def forward(self, m_sp):
        """
        m_sp: shape (4,) => [roll, pitch, yaw, thrust], in [-1..1] except thrust in [0..1].
        Returns motor outputs in [0..1].
        """
        m_sp = m_sp.reshape(4,1)
        if self.airmode == "none":
            _, u_new = normal_mode_np(m_sp, self.P, 0.0, 1.0)
        else:
            raise ValueError("Only 'none' airmode is implemented in this demo")

        # clamp final to [0..1]
        return np.clip(u_new, 0.0, 1.0)

###############################################################################
# 2) TorchMixer Implementation (with final [-1..1] remap)
###############################################################################
class TorchMixer(nn.Module):
    """
    TorchMixer replicates PX4 normal mode. The final step transforms [0..1] => [-1..1],
    just like PX4's "2f * output - 1f" in C++ code. 
    """
    def __init__(self, P_matrix, airmode="none"):
        super().__init__()
        self.airmode = airmode
        P_tensor = torch.as_tensor(P_matrix, dtype=torch.float32)
        self.register_buffer("P", P_tensor)
        pseudo_inv = torch.pinverse(P_tensor)
        self.register_buffer("B", pseudo_inv)
        self.eps = 1e-6

        # For simplicity, we assume _thrust_factor=0 (as in default PX4).
        # If needed, replicate the polynomial step from C++:
        # if thrust_factor > 0:
        #    output = -((1 - thrust_factor)/(2 * thrust_factor)) + ...
        self.thrust_factor = 0.0

    def forward(self, m_sp_batch, u_min=0.0, u_max=1.0):
        """
        m_sp_batch: [batch_size, 4], each row => [roll, pitch, yaw, thrust].
        Return motor signals in [-1..1], matching final PX4 transform.
        """
        batch_size = m_sp_batch.shape[0]
        num_rotors = self.P.shape[0]

        # (A) Basic multiply: P*m_sp => in [0..1]
        P_expanded = self.P.unsqueeze(0).expand(batch_size, num_rotors, 4)
        m_sp_3d = m_sp_batch.view(batch_size, 4, 1)
        u_init = torch.bmm(P_expanded, m_sp_3d).squeeze(-1)

        # (B) Normal mode logic
        if self.airmode == "none":
            u_final = self._normal_mode(m_sp_batch, u_init, u_min, u_max)
        else:
            raise ValueError("Only 'none' airmode is implemented in TorchMixer demo")

        # (C) clamp intermediate to [0..1]
        outputs_clamped = torch.clamp(u_final, min=0.0, max=1.0)

        # (D) replicate final PX4 transform: [0..1] => [-1..1]
        outputs_remapped = 2.0 * outputs_clamped - 1.0
        outputs_final = torch.clamp(outputs_remapped, -1.0, 1.0)

        # If you have a non-zero thrust_factor, you'd replicate that polynomial step here.
        # e.g. if self.thrust_factor > 0:
        #   outputs_final = self._apply_thrust_factor(outputs_final, self.thrust_factor)

        return outputs_final

    ###### normal_mode logic ######
    def _normal_mode(self, m_sp_batch, u_init, u_min, u_max):
        """
        Reproduces PX4 normal_mode steps (no airmode):
          1) remove yaw
          2) saturate only thrust
          3) saturate roll
          4) saturate pitch
          5) mix yaw (and revert if needed)
        """
        batch_size = m_sp_batch.shape[0]
        num_rotors = self.P.shape[0]

        # 1) remove yaw
        m_sp_no_yaw = m_sp_batch.clone()
        m_sp_no_yaw[:, 2] = 0.0

        # re-multiply => P*m_sp_no_yaw
        P_expanded = self.P.unsqueeze(0).expand(batch_size, num_rotors, 4)
        m_sp_3d = m_sp_no_yaw.view(batch_size,4,1)
        u_no_yaw = torch.bmm(P_expanded, m_sp_3d).squeeze(-1)

        # 2) saturate thrust => reduce_only
        thr_vec = self.P[:, 3].unsqueeze(0).expand(batch_size, num_rotors)
        u_prime = self._minimize_sat(u_no_yaw, thr_vec, u_min, u_max, reduce_only=True)
        # do not revert if bigger => matches PX4 code

        # 3) saturate roll
        roll_vec = self.P[:, 0].unsqueeze(0).expand(batch_size, num_rotors)
        u_p2 = self._minimize_sat(u_prime, roll_vec, u_min, u_max, reduce_only=False)

        # 4) saturate pitch
        pitch_vec = self.P[:, 1].unsqueeze(0).expand(batch_size, num_rotors)
        u_p3 = self._minimize_sat(u_p2, pitch_vec, u_min, u_max, reduce_only=False)

        # 5) mix yaw
        out_final = self._mix_yaw(m_sp_batch, u_p3, u_min, u_max)
        return out_final

    def _mix_yaw(self, m_sp_batch, u_in, u_min, u_max):
        """
        1) add yaw
        2) saturate yaw up to +0.15
        3) saturate thrust => reduce_only
        4) if final is bigger => revert
        """
        batch_size, num_rotors = u_in.shape
        yaw_sp = m_sp_batch[:, 2]
        yaw_mat = self.P[:, 2].unsqueeze(0).expand(batch_size, num_rotors)

        # 1) add yaw
        u_p = u_in + yaw_sp.unsqueeze(-1) * yaw_mat

        # 2) saturate yaw => up to 1.15
        u_p_unsat = self._minimize_sat(u_p, yaw_mat, 0.0, u_max + 0.15, reduce_only=False)

        # 3) saturate thrust => reduce_only
        thr_mat = self.P[:, 3].unsqueeze(0).expand(batch_size, num_rotors)
        u_pp = self._minimize_sat(u_p_unsat, thr_mat, 0.0, u_max, reduce_only=True)

        # 4) revert if bigger
        if torch.any(u_pp > u_p_unsat):
            final = u_p_unsat.clone()
        else:
            final = u_pp

        return final

    ###### saturations ######
    def _compute_desaturation_gain(self, u, desat_vec, u_min, u_max):
        """
        Gains to fix saturations => k_min + k_max approach, one rotor at a time.
        """
        batch_size, num_rotors = u.shape
        k_min_all = torch.zeros(batch_size, dtype=u.dtype, device=u.device)
        k_max_all = torch.zeros(batch_size, dtype=u.dtype, device=u.device)

        for i in range(num_rotors):
            mask = (desat_vec[:, i].abs() > self.eps)
            d_u_plus = (u_max - u[:, i])  # how much we can go up
            d_u_minus = (u_min - u[:, i]) # how much we can go down

            k_plus = torch.zeros_like(d_u_plus)
            k_minus = torch.zeros_like(d_u_minus)

            valid_minus = (d_u_minus > 0) & mask
            k_minus[valid_minus] = d_u_minus[valid_minus] / desat_vec[valid_minus, i]

            valid_plus = (d_u_plus < 0) & mask
            k_plus[valid_plus] = d_u_plus[valid_plus] / desat_vec[valid_plus, i]

            k_candidates = torch.stack([k_plus, k_minus], dim=1)
            k_min_i = k_candidates.min(dim=1).values
            k_max_i = k_candidates.max(dim=1).values

            k_min_all = torch.min(k_min_all, k_min_i)
            k_max_all = torch.max(k_max_all, k_max_i)

        return k_min_all + k_max_all

    def _minimize_sat(self, u, desat_vec, u_min, u_max, reduce_only=False):
        """
        1) compute k1 => add to u
        2) compute k2 => final = u + (k1 + 0.5*k2)*desat_vec
        if reduce_only => clamp k1 >= 0 => no upward push
        """
        k1 = self._compute_desaturation_gain(u, desat_vec, u_min, u_max)
        if reduce_only:
            k1 = torch.minimum(k1, torch.zeros_like(k1))

        u_1 = u + k1.unsqueeze(-1)*desat_vec
        k2 = self._compute_desaturation_gain(u_1, desat_vec, u_min, u_max)
        k_opt = k1 + 0.5*k2
        return u + k_opt.unsqueeze(-1)*desat_vec

# class TorchMixer(nn.Module):
#     """
#     TorchMixer implements the mixing logic using PyTorch.
#     Modified so that if any element of the thrust-only corrected output exceeds
#     the unsaturated output, the entire vector is reverted (to match PX4 C++ behavior).
#     Supports airmode "none" only in this demo.
#     """
#     def __init__(self, P_matrix, airmode="none"):
#         super().__init__()
#         self.airmode = airmode
#         P_tensor = torch.as_tensor(P_matrix, dtype=torch.float32)
#         self.register_buffer("P", P_tensor)  # shape: [num_rotors, 4]
#         pseudo_inv = torch.pinverse(P_tensor)
#         self.register_buffer("B", pseudo_inv)
#         self.eps = 1e-6

#     def forward(self, m_sp_batch, u_min=0.0, u_max=1.0):
#         """
#         m_sp_batch: tensor of shape [batch_size, 4] where each row is [roll, pitch, yaw, thrust]
#         Returns: tensor of shape [batch_size, num_rotors] motor outputs.
#         """
#         batch_size = m_sp_batch.shape[0]
#         num_rotors = self.P.shape[0]
#         P_expanded = self.P.unsqueeze(0).expand(batch_size, num_rotors, 4)
#         m_sp_3d = m_sp_batch.view(batch_size, 4, 1)
#         u_init = torch.bmm(P_expanded, m_sp_3d).squeeze(-1)

#         if self.airmode == "none":
#             u_final = self._normal_mode(m_sp_batch, u_init, u_min, u_max)
#         else:
#             raise ValueError("Only 'none' airmode is implemented in TorchMixer demo")
#         return torch.clamp(u_final, u_min, u_max)

#     def _compute_desaturation_gain(self, u, desat_vec, u_min, u_max):
#         batch_size, num_rotors = u.shape
#         k_min_all = torch.zeros(batch_size, device=u.device, dtype=u.dtype)
#         k_max_all = torch.zeros(batch_size, device=u.device, dtype=u.dtype)
#         for i in range(num_rotors):
#             mask = (desat_vec[:, i].abs() > self.eps)
#             d_u_plus = (u_max - u[:, i])
#             d_u_minus = (u_min - u[:, i])
#             k_plus = torch.zeros_like(d_u_plus)
#             k_minus = torch.zeros_like(d_u_minus)
#             valid_minus = (d_u_minus > 0) & mask
#             k_minus[valid_minus] = d_u_minus[valid_minus] / desat_vec[valid_minus, i]
#             valid_plus = (d_u_plus < 0) & mask
#             k_plus[valid_plus] = d_u_plus[valid_plus] / desat_vec[valid_plus, i]
#             k_candidates = torch.cat([k_plus.unsqueeze(-1), k_minus.unsqueeze(-1)], dim=1)
#             k_min_i = k_candidates.min(dim=1).values
#             k_max_i = k_candidates.max(dim=1).values
#             k_min_all = torch.min(k_min_all, k_min_i)
#             k_max_all = torch.max(k_max_all, k_max_i)
#         return k_min_all + k_max_all

#     def _minimize_sat(self, u, desat_vec, u_min, u_max, reduce_only=False):
#         k1 = self._compute_desaturation_gain(u, desat_vec, u_min, u_max)
#         if reduce_only:
#             k1 = torch.minimum(k1, torch.zeros_like(k1))
#         u_1 = u + k1.unsqueeze(-1) * desat_vec
#         k2 = self._compute_desaturation_gain(u_1, desat_vec, u_min, u_max)
#         k_opt = k1 + 0.5 * k2
#         u_prime = u + k_opt.unsqueeze(-1) * desat_vec
#         return u_prime

#     def _mix_yaw(self, m_sp_batch, u_in, u_min, u_max):
#         batch_size, num_rotors = u_in.shape
#         yaw_sp = m_sp_batch[:, 2]
#         yaw_mat = self.P[:, 2].unsqueeze(0).expand(batch_size, num_rotors)
#         u_p = u_in + yaw_sp.unsqueeze(-1) * yaw_mat

#         u_p_unsat = self._minimize_sat(u_p, yaw_mat, 0.0, u_max + 0.15, reduce_only=False)
#         thr_mat = self.P[:, 3].unsqueeze(0).expand(batch_size, num_rotors)
#         u_pp = self._minimize_sat(u_p_unsat, thr_mat, 0.0, u_max, reduce_only=True)
#         if torch.any(u_pp > u_p_unsat):
#             final = u_p_unsat.clone()
#         else:
#             final = u_pp
#         return final

#     def _normal_mode(self, m_sp_batch, u_init, u_min, u_max):
#         batch_size = m_sp_batch.shape[0]
#         num_rotors = self.P.shape[0]
#         m_sp_no_yaw = m_sp_batch.clone()
#         m_sp_no_yaw[:, 2] = 0.0
#         P_expanded = self.P.unsqueeze(0).expand(batch_size, num_rotors, 4)
#         u_no_yaw = torch.bmm(P_expanded, m_sp_no_yaw.view(batch_size, 4, 1)).squeeze(-1)
#         thr_vec = self.P[:, 3].unsqueeze(0).expand(batch_size, num_rotors)
#         u_prime = self._minimize_sat(u_no_yaw, thr_vec, u_min, u_max, reduce_only=True)
#         if torch.any(u_prime > u_no_yaw):
#             u_prime = u_no_yaw.clone()
#         roll_vec = self.P[:, 0].unsqueeze(0).expand(batch_size, num_rotors)
#         u_p2 = self._minimize_sat(u_prime, roll_vec, u_min, u_max, reduce_only=False)
#         pitch_vec = self.P[:, 1].unsqueeze(0).expand(batch_size, num_rotors)
#         u_p3 = self._minimize_sat(u_p2, pitch_vec, u_min, u_max, reduce_only=False)
#         out_final = self._mix_yaw(m_sp_batch, u_p3, u_min, u_max)
#         return out_final

###############################################################################
# 3) Compare Function
###############################################################################
def compare_mixers():
    """
    Generates 100 test inputs (roll from -0.2 to 0.2, pitch from 0.0 to 0.2, yaw=0, thrust=0)
    and compares the final outputs from:
      - PX4Mixer (NumPy, normal mode, final in [0..1])
      - TorchMixer (PyTorch, normal mode, final in [-1..1])
    """
    # Quad_x geometry
    quad_x = np.array([
        [-0.71,  0.71,  1.0,  1.0],
        [ 0.71, -0.71,  1.0,  1.0],
        [ 0.71,  0.71, -1.0,  1.0],
        [-0.71, -0.71, -1.0,  1.0]
    ], dtype=np.float32)

    # Create mixers
    px4_mixer = PX4Mixer(quad_x, airmode="none")
    torch_mixer = TorchMixer(quad_x, airmode="none").to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    # Create 100 test inputs
    N = 100
    p_sp = np.linspace(-0.2, 0.2, N)
    q_sp = np.linspace(0.0, 0.2, N)
    r_sp = np.zeros(N)
    T_sp = np.ones(N)
    inputs_np = np.stack([p_sp, q_sp, r_sp, T_sp], axis=-1)  # shape (N,4)

    # PX4 mixer => returns in [0..1]
    px4_outputs = []
    for i in range(N):
        out = px4_mixer.forward(inputs_np[i])  # shape (4,)
        px4_outputs.append(out)
    px4_outputs = np.array(px4_outputs)  # shape (N,4)

    # Torch mixer => final in [-1..1]
    inputs_torch = torch.tensor(inputs_np, dtype=torch.float32, device=torch_mixer.P.device)
    with torch.no_grad():
        torch_outputs = torch_mixer(inputs_torch, 0.0, 1.0).cpu().numpy()  # shape (N,4)

    # We can transform PX4 outputs to match the final [-1..1] if we want direct 1:1 comparison:
    #   final_px4 = 2 * px4_outputs - 1
    # or we can keep them separate and see the difference.
    # Let's do final_px4 in [-1..1]:
    final_px4 = 2.0 * px4_outputs - 1.0
    final_px4 = np.clip(final_px4, -1.0, 1.0)

    # Compare numerically
    num_motors = px4_outputs.shape[1]
    for motor in range(num_motors):
        # Compare final_px4 vs torch_outputs
        rmse = np.sqrt(np.mean((final_px4[:, motor] - torch_outputs[:, motor])**2))
        corr = np.corrcoef(final_px4[:, motor], torch_outputs[:, motor])[0,1]
        print(f"Motor {motor}: RMSE={rmse:.6f}, Corr={corr:.6f}")

    # Plot subplots
    fig, axs = plt.subplots(2, 2, figsize=(10,8))
    axs = axs.flatten()
    for motor in range(num_motors):
        ax = axs[motor]
        ax.plot(final_px4[:, motor], 'o-', label=f"PX4 (Remapped) Motor {motor}")
        ax.plot(torch_outputs[:, motor], 's--', label=f"Torch Motor {motor}")
        ax.set_title(f"Motor {motor}")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Output [-1..1]")
        ax.legend()
        ax.grid(True)

    fig.suptitle("PX4 vs Torch Mixer (Normal Mode), final in [-1..1]", fontsize=14)
    fig.tight_layout()
    plt.show()

###############################################################################
# 4) Main
###############################################################################
if __name__ == "__main__":
    compare_mixers()
