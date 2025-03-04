#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script compares two mixer implementations for a multirotor:

  1. PX4Mixer (a NumPy implementation mimicking the PX4 C++ mixer)
  2. TorchMixer (a PyTorch implementation, modified to match the PX4 behavior)

Both mixers are given the same test inputs and their outputs are compared
numerically (RMSE and correlation per motor) and visually.
"""

import numpy as np
import math
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

###############################################################################
# 1) PX4 Mixer (NumPy Implementation)
###############################################################################
def compute_desaturation_gain_np(u, u_min, u_max, desaturation_vector):
    """
    Computes the gain k by which desaturation_vector must be multiplied
    to unsaturate the output with the greatest saturation.
    
    u, u_min, u_max, desaturation_vector are all converted to 1D arrays.
    If u_min or u_max are scalars, they are converted to arrays of the same shape as u.
    """
    u = np.ravel(u)
    if np.isscalar(u_min) or np.ndim(u_min) == 0:
        u_min = np.full(u.shape, u_min)
    else:
        u_min = np.ravel(u_min)
    if np.isscalar(u_max) or np.ndim(u_max) == 0:
        u_max = np.full(u.shape, u_max)
    else:
        u_max = np.ravel(u_max)
    desaturation_vector = np.ravel(desaturation_vector)
    
    d_u_sat_plus = u_max - u   # How much can we increase?
    d_u_sat_minus = u_min - u   # How much can we decrease?
    
    k_min = 0.0
    k_max = 0.0
    for i in range(u.size):
        if abs(desaturation_vector[i]) < 1e-6:
            continue
        if u[i] < u_min[i]:
            k = (u_min[i] - u[i]) / desaturation_vector[i]
            if k < k_min:
                k_min = k
            if k > k_max:
                k_max = k
        if u[i] > u_max[i]:
            k = (u_max[i] - u[i]) / desaturation_vector[i]
            if k < k_min:
                k_min = k
            if k > k_max:
                k_max = k
    return k_min + k_max

def minimize_sat_np(u, u_min, u_max, desaturation_vector):
    k1 = compute_desaturation_gain_np(u, u_min, u_max, desaturation_vector)
    u_1 = u + k1 * desaturation_vector
    k2 = compute_desaturation_gain_np(u_1, u_min, u_max, desaturation_vector)
    k_opt = k1 + 0.5 * k2
    return u + k_opt * desaturation_vector

def mix_yaw_np(m_sp, u, P, u_min, u_max):
    """
    Mix yaw independently:
      - Create a vector with only the yaw command.
      - Add it to the existing output.
      - Then perform two saturation corrections:
          first with the yaw scale (allowing +15% overhead),
          then with the thrust scale (only reducing thrust).
      - If any element of the thrust-only corrected vector is higher than the previous, revert entirely.
    """
    m_sp = np.ravel(m_sp)
    m_sp_yaw_only = np.zeros_like(m_sp)
    m_sp_yaw_only[2] = m_sp[2]
    u_p = u + (P @ m_sp_yaw_only.reshape(-1, 1)).flatten()

    u_r_dot = np.ravel(P[:, 2])
    u_pp = minimize_sat_np(u_p, u_min, u_max + 0.15, u_r_dot)
    u_T = np.ravel(P[:, 3])
    u_ppp = minimize_sat_np(u_pp, 0.0, u_max, u_T)
    if np.any(u_ppp > u_pp):
        u_ppp = u_pp.copy()
    return u_ppp

def normal_mode_np(m_sp, P, u_min, u_max):
    """
    Implements the 'normal_mode' mixing behavior:
      - Mix without yaw (set yaw to zero).
      - Use thrust to unsaturate (only reducing thrust).
      - Then adjust roll and pitch if needed.
      - Finally, mix yaw independently.
    """
    m_sp_no_yaw = m_sp.copy()
    m_sp_no_yaw[2, 0] = 0.0
    u = (P @ m_sp_no_yaw).flatten()
    u_T = np.ravel(P[:, 3])
    u_prime = minimize_sat_np(u, u_min, u_max, u_T)
    if np.any(u_prime > u):
        u_prime = u.copy()
    u_p_dot = np.ravel(P[:, 0])
    u_p2 = minimize_sat_np(u_prime, u_min, u_max, u_p_dot)
    u_q_dot = np.ravel(P[:, 1])
    u_p3 = minimize_sat_np(u_p2, u_min, u_max, u_q_dot)
    u_final = mix_yaw_np(m_sp, u_p3, P, u_min, u_max)
    return (u, u_final)

class PX4Mixer:
    """
    PX4Mixer uses NumPy to implement the mixing logic.
    """
    def __init__(self, P, airmode="none"):
        self.P = np.array(P)
        self.airmode = airmode
        self.B = np.linalg.pinv(self.P)

    def forward(self, m_sp):
        """
        m_sp: a 1D array of shape (4,) representing [roll, pitch, yaw, thrust]
        Returns a 1D array of motor outputs.
        """
        m_sp = np.ravel(m_sp).reshape(4, 1)
        if self.airmode == "none":
            _, u_new = normal_mode_np(m_sp, self.P, 0.0, 1.0)
        else:
            raise ValueError("Only 'none' airmode is implemented in PX4Mixer demo")
        u_new_sat = np.clip(u_new, 0.0, 1.0)
        return u_new_sat

###############################################################################
# 2) TorchMixer (Modified PyTorch Implementation)
###############################################################################
class TorchMixer(nn.Module):
    """
    TorchMixer implements the mixing logic using PyTorch.
    Modified so that if any element of the thrust-only corrected output exceeds
    the unsaturated output, the entire vector is reverted (to match PX4 C++ behavior).
    Supports airmode "none" only in this demo.
    """
    def __init__(self, P_matrix, airmode="none"):
        super().__init__()
        self.airmode = airmode
        P_tensor = torch.as_tensor(P_matrix, dtype=torch.float32)
        self.register_buffer("P", P_tensor)  # shape: [num_rotors, 4]
        pseudo_inv = torch.pinverse(P_tensor)
        self.register_buffer("B", pseudo_inv)
        self.eps = 1e-6

    def forward(self, m_sp_batch, u_min=0.0, u_max=1.0):
        """
        m_sp_batch: tensor of shape [batch_size, 4] where each row is [roll, pitch, yaw, thrust]
        Returns: tensor of shape [batch_size, num_rotors] motor outputs.
        """
        batch_size = m_sp_batch.shape[0]
        num_rotors = self.P.shape[0]
        P_expanded = self.P.unsqueeze(0).expand(batch_size, num_rotors, 4)
        m_sp_3d = m_sp_batch.view(batch_size, 4, 1)
        u_init = torch.bmm(P_expanded, m_sp_3d).squeeze(-1)

        if self.airmode == "none":
            u_final = self._normal_mode(m_sp_batch, u_init, u_min, u_max)
        else:
            raise ValueError("Only 'none' airmode is implemented in TorchMixer demo")
        return torch.clamp(u_final, u_min, u_max)

    def _compute_desaturation_gain(self, u, desat_vec, u_min, u_max):
        batch_size, num_rotors = u.shape
        k_min_all = torch.zeros(batch_size, device=u.device, dtype=u.dtype)
        k_max_all = torch.zeros(batch_size, device=u.device, dtype=u.dtype)
        for i in range(num_rotors):
            mask = (desat_vec[:, i].abs() > self.eps)
            d_u_plus = (u_max - u[:, i])
            d_u_minus = (u_min - u[:, i])
            k_plus = torch.zeros_like(d_u_plus)
            k_minus = torch.zeros_like(d_u_minus)
            valid_minus = (d_u_minus > 0) & mask
            k_minus[valid_minus] = d_u_minus[valid_minus] / desat_vec[valid_minus, i]
            valid_plus = (d_u_plus < 0) & mask
            k_plus[valid_plus] = d_u_plus[valid_plus] / desat_vec[valid_plus, i]
            k_candidates = torch.cat([k_plus.unsqueeze(-1), k_minus.unsqueeze(-1)], dim=1)
            k_min_i = k_candidates.min(dim=1).values
            k_max_i = k_candidates.max(dim=1).values
            k_min_all = torch.min(k_min_all, k_min_i)
            k_max_all = torch.max(k_max_all, k_max_i)
        return k_min_all + k_max_all

    def _minimize_sat(self, u, desat_vec, u_min, u_max, reduce_only=False):
        k1 = self._compute_desaturation_gain(u, desat_vec, u_min, u_max)
        if reduce_only:
            k1 = torch.minimum(k1, torch.zeros_like(k1))
        u_1 = u + k1.unsqueeze(-1) * desat_vec
        k2 = self._compute_desaturation_gain(u_1, desat_vec, u_min, u_max)
        k_opt = k1 + 0.5 * k2
        u_prime = u + k_opt.unsqueeze(-1) * desat_vec
        return u_prime

    def _mix_yaw(self, m_sp_batch, u_in, u_min, u_max):
        batch_size, num_rotors = u_in.shape
        yaw_sp = m_sp_batch[:, 2]
        yaw_mat = self.P[:, 2].unsqueeze(0).expand(batch_size, num_rotors)
        u_p = u_in + yaw_sp.unsqueeze(-1) * yaw_mat

        u_p_unsat = self._minimize_sat(u_p, yaw_mat, 0.0, u_max + 0.15, reduce_only=False)
        thr_mat = self.P[:, 3].unsqueeze(0).expand(batch_size, num_rotors)
        u_pp = self._minimize_sat(u_p_unsat, thr_mat, 0.0, u_max, reduce_only=True)
        if torch.any(u_pp > u_p_unsat):
            final = u_p_unsat.clone()
        else:
            final = u_pp
        return final

    def _normal_mode(self, m_sp_batch, u_init, u_min, u_max):
        batch_size = m_sp_batch.shape[0]
        num_rotors = self.P.shape[0]
        m_sp_no_yaw = m_sp_batch.clone()
        m_sp_no_yaw[:, 2] = 0.0
        P_expanded = self.P.unsqueeze(0).expand(batch_size, num_rotors, 4)
        u_no_yaw = torch.bmm(P_expanded, m_sp_no_yaw.view(batch_size, 4, 1)).squeeze(-1)
        thr_vec = self.P[:, 3].unsqueeze(0).expand(batch_size, num_rotors)
        u_prime = self._minimize_sat(u_no_yaw, thr_vec, u_min, u_max, reduce_only=True)
        if torch.any(u_prime > u_no_yaw):
            u_prime = u_no_yaw.clone()
        roll_vec = self.P[:, 0].unsqueeze(0).expand(batch_size, num_rotors)
        u_p2 = self._minimize_sat(u_prime, roll_vec, u_min, u_max, reduce_only=False)
        pitch_vec = self.P[:, 1].unsqueeze(0).expand(batch_size, num_rotors)
        u_p3 = self._minimize_sat(u_p2, pitch_vec, u_min, u_max, reduce_only=False)
        out_final = self._mix_yaw(m_sp_batch, u_p3, u_min, u_max)
        return out_final

# class TorchMixer(nn.Module):
#     def __init__(self, P_matrix, airmode="none"):
#         super().__init__()
#         self.airmode = airmode
#         P_tensor = torch.as_tensor(P_matrix, dtype=torch.float32)
#         self.register_buffer("P", P_tensor)  # shape: [num_rotors, 4]

#         pseudo_inv = torch.pinverse(P_tensor)
#         self.register_buffer("B", pseudo_inv)
#         self.eps = 1e-6

#     def forward(self, m_sp_batch, u_min=0.0, u_max=1.0):
#         batch_size = m_sp_batch.shape[0]
#         num_rotors = self.P.shape[0]

#         # Basic multiply: u_init = P * m_sp
#         P_expanded = self.P.unsqueeze(0).expand(batch_size, num_rotors, 4)
#         m_sp_3d = m_sp_batch.view(batch_size, 4, 1)
#         u_init = torch.bmm(P_expanded, m_sp_3d).squeeze(-1)

#         if self.airmode == "none":
#             u_final = self._normal_mode(m_sp_batch, u_init, u_min, u_max)
#         else:
#             raise ValueError("Only 'none' airmode is implemented here")

#         return torch.clamp(u_final, u_min, u_max)

#     ###### PX4-like normal_mode ######
#     def _normal_mode(self, m_sp_batch, u_init, u_min, u_max):
#         """
#         Reproduces PX4 normal mode steps:
#           1) remove yaw
#           2) saturate only thrust (reduce_only)
#           3) saturate roll
#           4) saturate pitch
#           5) mix yaw in a separate step, revert if new is bigger
#         """
#         batch_size = m_sp_batch.shape[0]
#         num_rotors = self.P.shape[0]

#         # (1) remove yaw from the input
#         m_sp_no_yaw = m_sp_batch.clone()
#         m_sp_no_yaw[:, 2] = 0.0

#         # re-multiply => P * m_sp_no_yaw
#         P_expanded = self.P.unsqueeze(0).expand(batch_size, num_rotors, 4)
#         u_no_yaw = torch.bmm(P_expanded, m_sp_no_yaw.view(batch_size, 4, 1)).squeeze(-1)

#         # (2) saturate only thrust
#         thr_vec = self.P[:, 3].unsqueeze(0).expand(batch_size, num_rotors)
#         u_prime = self._minimize_sat(u_no_yaw, thr_vec, u_min, u_max, reduce_only=True)
#         # NOTE: do *not* revert if new is bigger. PX4 doesn't revert here.

#         # (3) saturate roll
#         roll_vec = self.P[:, 0].unsqueeze(0).expand(batch_size, num_rotors)
#         u_p2 = self._minimize_sat(u_prime, roll_vec, u_min, u_max, reduce_only=False)

#         # (4) saturate pitch
#         pitch_vec = self.P[:, 1].unsqueeze(0).expand(batch_size, num_rotors)
#         u_p3 = self._minimize_sat(u_p2, pitch_vec, u_min, u_max, reduce_only=False)

#         # (5) mix yaw (separate pass)
#         out_final = self._mix_yaw(m_sp_batch, u_p3, u_min, u_max)
#         return out_final

#     def _mix_yaw(self, m_sp_batch, u_in, u_min, u_max):
#         """
#         1) add yaw
#         2) saturate yaw up to +0.15 overhead
#         3) saturate thrust (reduce_only)
#         4) if the final result is bigger than the previous => revert
#         """
#         batch_size, num_rotors = u_in.shape
#         yaw_sp = m_sp_batch[:, 2]
#         yaw_mat = self.P[:, 2].unsqueeze(0).expand(batch_size, num_rotors)

#         # 1) add yaw
#         u_p = u_in + yaw_sp.unsqueeze(-1) * yaw_mat

#         # 2) saturate with yaw, up to 1.15
#         u_p_unsat = self._minimize_sat(u_p, yaw_mat, 0.0, u_max + 0.15, reduce_only=False)

#         # 3) saturate thrust (reduce_only)
#         thr_mat = self.P[:, 3].unsqueeze(0).expand(batch_size, num_rotors)
#         u_pp = self._minimize_sat(u_p_unsat, thr_mat, 0.0, u_max, reduce_only=True)

#         # 4) if the new version is bigger => revert
#         if torch.any(u_pp > u_p_unsat):
#             final = u_p_unsat.clone()
#         else:
#             final = u_pp
#     ###### saturations ######
#     def _compute_desaturation_gain(self, u, desat_vec, u_min, u_max):
#         """
#         Same logic as the PX4 approach: sum (k_min + k_max) across all rotors.
#         """
#         batch_size, num_rotors = u.shape
#         k_min_all = torch.zeros(batch_size, dtype=u.dtype, device=u.device)
#         k_max_all = torch.zeros(batch_size, dtype=u.dtype, device=u.device)

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
#         """
#         1) compute k1
#         2) u_1 = u + k1*desat_vec
#         3) compute k2
#         4) final = u + (k1 + 0.5*k2)*desat_vec
#         If reduce_only=True, we clamp k1 >= 0 => never let motors go above original.
#         """
#         k1 = self._compute_desaturation_gain(u, desat_vec, u_min, u_max)
#         if reduce_only:
#             k1 = torch.minimum(k1, torch.zeros_like(k1))

#         u_1 = u + k1.unsqueeze(-1)*desat_vec
#         k2 = self._compute_desaturation_gain(u_1, desat_vec, u_min, u_max)
#         k_opt = k1 + 0.5*k2

#         return u + k_opt.unsqueeze(-1)*desat_vec


###############################################################################
# 3) Compare Mixers
###############################################################################
def compare_mixers():
    """
    Creates a quadrotor geometry (quad_x), then generates 100 test inputs
    (a ramp in roll and pitch, with zero yaw and zero thrust).
    It runs both PX4Mixer (NumPy) and TorchMixer on these inputs,
    computes RMSE and correlation for each motor channel,
    and plots the outputs in 4 subplots for visual comparison.
    """
    # Quadrotor geometry (quad_x)
    quad_x = np.array([
        [-0.71,  0.71,  1.0,  1.0],
        [ 0.71, -0.71,  1.0,  1.0],
        [ 0.71,  0.71, -1.0,  1.0],
        [-0.71, -0.71, -1.0,  1.0]
    ], dtype=np.float32)

    # Create mixers (airmode "none")
    px4_mixer = PX4Mixer(quad_x, airmode="none")
    torch_mixer = TorchMixer(quad_x, airmode="none")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch_mixer.to(device)

    # Create 100 test inputs:
    # Roll: -0.2 to 0.2, Pitch: 0.0 to 0.2, Yaw: 0, Thrust: 0.
    N = 100
    p_sp = np.linspace(-2, 2, N)
    q_sp = np.linspace(0.0, 2, N)
    r_sp = np.zeros(N)
    T_sp = np.ones(N)
    inputs_np = np.stack([p_sp, q_sp, r_sp, T_sp], axis=-1)  # shape (N,4)

    # Run PX4 mixer (NumPy) for each input:
    px4_outputs = []
    for i in range(N):
        out = px4_mixer.forward(inputs_np[i])
        px4_outputs.append(out)
    px4_outputs = np.array(px4_outputs)  # shape (N, num_rotors)

    # Run Torch mixer in batch:
    inputs_torch = torch.tensor(inputs_np, device=device, dtype=torch.float32)
    with torch.no_grad():
        torch_outputs = torch_mixer(inputs_torch, u_min=0.0, u_max=1.0).cpu().numpy()  # shape (N, num_rotors)

    # Compute numerical metrics per motor:
    num_motors = px4_outputs.shape[1]
    for motor in range(num_motors):
        m_px4 = px4_outputs[:, motor]
        m_torch = torch_outputs[:, motor]
        rmse = np.sqrt(np.mean((m_px4 - m_torch) ** 2))
        corr = np.corrcoef(m_px4, m_torch)[0, 1]
        print(f"Motor {motor}: RMSE = {rmse:.6f}, Corr = {corr:.6f}")

    # Plot results in 4 subplots (one per motor)
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))  # 2x2 grid
    axs = axs.flatten()  # flatten to easily iterate [0..3]
    for motor in range(num_motors):
        ax = axs[motor]
        ax.plot(px4_outputs[:, motor], 'o-', label=f"PX4 Motor {motor}")
        ax.plot(torch_outputs[:, motor], 's--', label=f"Torch Motor {motor}")
        ax.set_title(f"Motor {motor}")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Motor Output")
        ax.legend()
        ax.grid(True)

    fig.suptitle("Mixer Outputs Comparison (airmode='none')", fontsize=14)
    fig.tight_layout()
    plt.show()


###############################################################################
# 4) Main
###############################################################################
if __name__ == "__main__":
    compare_mixers()
