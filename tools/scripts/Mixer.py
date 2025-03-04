import math
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class RateController(nn.Module):
    def __init__(self, num_envs, dt):
        super().__init__()
        # default gain set up
        default_Kp = torch.tensor([0.15, 0.15, 0.20], dtype=torch.float32)
        default_Ki = torch.tensor([0.20, 0.20, 0.10], dtype=torch.float32)
        default_Kd = torch.tensor([0.003, 0.003, 0.00], dtype=torch.float32)
        default_Kff = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
        default_int_lim = torch.tensor([0.3, 0.3, 0.3], dtype=torch.float32)
        default_out_lim = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
        default_slew_rate = torch.full((3,), float('inf'), dtype=torch.float32)

        self.num_envs = num_envs    # ?????   input number of envs manually for now
        self.dt = dt

        self.register_buffer("gain_p", default_Kp)  # register_buffer to keep gain unchanged
        self.register_buffer("gain_i", default_Ki)
        self.register_buffer("gain_d", default_Kd)
        self.register_buffer("gain_ff", default_Kff)
        self.register_buffer("lim_int", default_int_lim)
        self.register_buffer("lim_out", default_out_lim)
        self.register_buffer("slew_rate", default_slew_rate)

        self.register_buffer("rate_int", torch.zeros((num_envs, 3), dtype=torch.float32))
        self.register_buffer("prev_output", torch.zeros((num_envs, 3), dtype=torch.float32))

    def forward(self, rate, rate_sp, angular_accel, landed = False, saturation_positive=None, saturation_negative=None):
        if saturation_positive is None:     # in PX4 this is given by allocator
            saturation_positive = torch.zeros_like(rate, dtype=torch.bool)
        if saturation_negative is None:
            saturation_negative = torch.zeros_like(rate, dtype=torch.bool)

        rate_error = rate_sp - rate
        p_term = self.gain_p * rate_error
        d_term = self.gain_d * angular_accel
        ff_term = self.gain_ff * rate_sp
        raw_output = p_term + self.rate_int - d_term + ff_term  # without anti-windup

        clipped_output = torch.clamp(raw_output, -self.lim_out, self.lim_out)   # output limit
        delta_output = clipped_output - self.prev_output
        max_delta = self.slew_rate * self.dt
        limited_delta = torch.clamp(delta_output, -max_delta, max_delta)
        final_output = self.prev_output + limited_delta

        update_mask = ~landed
        rate_error_for_i = rate_error.clone()
        rate_error_for_i[saturation_positive & (rate_error_for_i > 0)] = 0.0
        rate_error_for_i[saturation_negative & (rate_error_for_i < 0)] = 0.0

        radians_400 = 400.0 * math.pi / 180.0
        i_factor = 1.0 - (rate_error_for_i / radians_400) ** 2
        i_factor = torch.clamp(i_factor, min=0.0, max=1.0)

        delta_int = i_factor * (self.gain_i * rate_error_for_i) * self.dt
        delta_int = delta_int * update_mask.unsqueeze(-1)
        new_rate_int = self.rate_int + delta_int
        new_rate_int = torch.clamp(new_rate_int, -self.lim_int, self.lim_int)
        self.rate_int.copy_(new_rate_int)
        self.prev_output.copy_(final_output)

        return final_output

    def reset(self, env_ids=None):
        with torch.no_grad():
            if env_ids is None:
                self.rate_int.zero_()
                self.prev_output.zero_()
            else:
                env_ids_t = torch.as_tensor(env_ids, dtype=torch.long, device=self.rate_int.device)
                self.rate_int[env_ids_t] = 0.0
                self.prev_output[env_ids_t] = 0.0


###############################################################################
#  2) MULTIROTOR MIXER (Thrust Allocator) in Torch
###############################################################################
class TorchMixer(nn.Module):
    """
    TorchMixer is a multi-rotor thrust allocator that reproduces the PX4 logic:
      - normal_mode (airmode disabled)
      - airmode_rp
      - airmode_rpy
    The approach is the same as Python reference script, but fully vectorized in Torch for GPU usage.
    """

    def __init__(self, P_matrix, airmode="none"):
        """
        P_matrix: Tensor of shape [num_rotors, 4], describing the geometry / control effectiveness:
                  Each row is [roll_scale, pitch_scale, yaw_scale, thrust_scale]
        airmode: one of ["none", "rp", "rpy"], corresponds to:
                 - none => normal_mode
                 - rp   => airmode_rp
                 - rpy  => airmode_rpy
        """
        super().__init__()
        self.airmode = airmode

        # Register P as buffer so it can remain on GPU
        P_tensor = torch.as_tensor(P_matrix, dtype=torch.float32)
        self.register_buffer("P", P_tensor)  # shape: [num_rotors, 4]

        # Build pseudo-inverse for "allocated accelerations" debug if needed
        pseudo_inv = torch.pinverse(P_tensor)  # shape [4, num_rotors]
        self.register_buffer("B", pseudo_inv)

        # A small epsilon to avoid divide-by-zero
        self.eps = 1e-6

    def forward(self, m_sp_batch, u_min=0.0, u_max=1.0):
        """
        m_sp_batch: [batch_size, 4], each row is [p_dot_sp, q_dot_sp, r_dot_sp, T_sp]
                    They are all in [-1, 1] for p,q,r, and [0,1] for thrust, same as in PX4.

        Returns:
           allocated_outputs: [batch_size, num_rotors]
        """

        # Expand shape info
        batch_size = m_sp_batch.shape[0]
        num_rotors = self.P.shape[0]

        # Step 1) Basic multiply:  u = P * m_sp  (but we need a vector for each in the batch)
        # We'll expand P to shape [1, num_rotors, 4] and do a batched matmul with [batch_size, 4, 1]
        P_expanded = self.P.unsqueeze(0).expand(batch_size, num_rotors, 4)        # [batch_size, num_rotors, 4]
        m_sp_3d = m_sp_batch.view(batch_size, 4, 1)                              # [batch_size, 4, 1]
        u_init = torch.bmm(P_expanded, m_sp_3d).squeeze(-1)                      # [batch_size, num_rotors]

        # According to airmode, we do different saturation-handling logic
        if self.airmode == "none":
            u_final = self._normal_mode(m_sp_batch, u_init, u_min, u_max)
        elif self.airmode == "rp":
            u_final = self._airmode_rp(m_sp_batch, u_init, u_min, u_max)
        elif self.airmode == "rpy":
            u_final = self._airmode_rpy(m_sp_batch, u_init, u_min, u_max)
        else:
            # default to normal_mode if the user passes something invalid
            u_final = self._normal_mode(m_sp_batch, u_init, u_min, u_max)

        # Finally clamp to [0,1]
        outputs_clamped = torch.clamp(u_final, min=0.0, max=1.0)

        return outputs_clamped

    ####################### Helper Routines ########################
    def _compute_desaturation_gain(self, u, desat_vec, u_min, u_max):
        """
        Vectorized version of compute_desaturation_gain(u, u_min, u_max, desaturation_vector)
        from the PX4 logic. For each environment in the batch, returns a single k scalar
        that attempts to reduce saturation as much as possible.
        """
        # shape checks
        #   u, desat_vec => [batch_size, num_rotors]
        # We'll compute for each rotor how much k is needed to unsaturate that rotor
        # Then we take min(k) and max(k) from the entire set, then sum them => result is k.
        batch_size, num_rotors = u.shape

        # We skip those with near-zero desat_vec to avoid division by zero
        # We'll create arrays for k_min and k_max across all rotors, init them at zero
        k_min_all = torch.zeros(batch_size, dtype=u.dtype, device=u.device)
        k_max_all = torch.zeros(batch_size, dtype=u.dtype, device=u.device)

        # We can do the math in a for-loop or a vector approach
        # For clarity, let's do a direct loop over rotors.
        for i in range(num_rotors):
            mask = (desat_vec[:, i].abs() > self.eps)  # only compute where nonzero
            # d_u_plus  = u_max - u
            d_u_plus = (u_max - u[:, i])
            # d_u_minus = u_min - u
            d_u_minus = (u_min - u[:, i])

            # Gains that would fix saturation (k = d_u / desat_vec)
            k_plus = torch.zeros_like(d_u_plus)
            k_minus = torch.zeros_like(d_u_minus)

            # For those below u_min => we can solve for how much k to push up
            valid_minus = (d_u_minus > 0) & mask
            k_minus[valid_minus] = d_u_minus[valid_minus] / desat_vec[valid_minus, i]

            # For those above u_max => we can solve for how much k to pull down
            valid_plus = (d_u_plus < 0) & mask
            k_plus[valid_plus] = d_u_plus[valid_plus] / desat_vec[valid_plus, i]

            # We want min and max from the union of [k_plus, k_minus]
            k_candidates = torch.cat([k_plus.unsqueeze(-1), k_minus.unsqueeze(-1)], dim=-1)  # shape [batch_size,2]

            # Get min, max per env
            k_min_i = k_candidates.min(dim=1).values
            k_max_i = k_candidates.max(dim=1).values

            # We update the global k_min_all and k_max_all
            #   k_min_all = min( k_min_all, k_min_i )
            #   k_max_all = max( k_max_all, k_max_i )
            # But note we want the "minimum of all negative" and "maximum of all positive"
            # Actually we want to keep track of the "lowest" across the entire rotor set
            #   => so k_min_all is the min across all rotors
            #   => and k_max_all is the max across all rotors
            k_min_all = torch.min(k_min_all, k_min_i)
            k_max_all = torch.max(k_max_all, k_max_i)

        # The final gain in the PX4 script is k_min + k_max
        k = k_min_all + k_max_all  # shape [batch_size]
        return k

    def _minimize_sat(self, u, desat_vec, u_min, u_max, reduce_only=False):
        """
        Vectorized version of the "minimize_sat" function:
          1) compute k1 from _compute_desaturation_gain
          2) add k1 * desat_vec to "u"  (try to unsaturate)
          3) compute k2 again on the updated result
          4) final solution = u + (k1 + 0.5 * k2) * desat_vec
        If reduce_only=True, we do nothing if k1>0, i.e. we never allow motors to go above the original "u".

        Returns new allocated outputs [batch_size, num_rotors].
        """
        batch_size, num_rotors = u.shape
        k1 = self._compute_desaturation_gain(u, desat_vec, u_min, u_max)  # [batch_size]
        if reduce_only:
            # If k1>0 => means we would push motors upward => not allowed => skip
            k1_clamped = torch.minimum(k1, torch.zeros_like(k1))
        else:
            k1_clamped = k1

        # Update once
        u_1 = u + k1_clamped.view(-1, 1) * desat_vec
        # Compute again
        k2 = self._compute_desaturation_gain(u_1, desat_vec, u_min, u_max)
        k_opt = k1_clamped + 0.5 * k2

        u_prime = u + k_opt.view(-1,1) * desat_vec
        return u_prime

    def _mix_yaw(self, m_sp_batch, u_in, u_min, u_max):
        """
        Yaw mixing in a separate pass, as in Python script:
          1) add yaw to the existing u_in
          2) saturate by adjusting yaw, allowing up to +15% on upper side
          3) saturate by adjusting thrust only on the lower side
        Then if the final result is bigger than the previous in any dimension,
        we keep the old one (i.e. "we never raise thrust"?).
        """
        batch_size, num_rotors = u_in.shape
        # m_sp_batch shape [batch_size, 4]
        #   => yaw_sp = m_sp_batch[:,2]
        yaw_sp = m_sp_batch[:, 2]
        yaw_scale = self.P[:, 2]  # shape [num_rotors]

        # Add yaw
        #    u_p = u_in + yaw_sp * (vector of shape [num_rotors])
        yaw_mat = yaw_scale.unsqueeze(0).expand(batch_size, num_rotors)  # [batch_size, num_rotors]
        u_p = u_in + yaw_sp.view(-1,1) * yaw_mat

        # step a) allow some yaw response at max thrust => we do a "minimize_sat" with range [0,1.15]
        # shape of 'yaw_mat' is the "desaturation_vector"
        u_p_unsat = self._minimize_sat(u_p, yaw_mat, u_min=0.0, u_max=1.15, reduce_only=False)

        # step b) reduce thrust only => we do a "minimize_sat" with range [0,1], but the "desaturation_vector" is
        # the thrust scale => P[:,3]
        thr_mat = self.P[:, 3].unsqueeze(0).expand(batch_size, num_rotors)
        u_pp = self._minimize_sat(u_p_unsat, thr_mat, u_min=0.0, u_max=1.0, reduce_only=True)

        # If the new version is bigger than old => revert
        # (the reference code does "if (u_ppp > (u_pp)).any(): revert", but that's per-element:
        #  we replicate that logic in a vector sense)
        # We'll do an elementwise check: use "torch.where"
        bigger_mask = (u_pp > u_p_unsat)
        final = torch.where(bigger_mask, u_p_unsat, u_pp)
        return final

    ####################### Different modes ########################
    def _airmode_rp(self, m_sp_batch, u_init, u_min, u_max):
        """
        'airmode_rp' => thrust can be used to keep roll/pitch authority,
        but Yaw is treated separately (similar to the Python reference).
        """
        # 1) zero out yaw in the input
        m_sp_no_yaw = m_sp_batch.clone()
        m_sp_no_yaw[:, 2] = 0.0  # remove yaw

        # re-multiply => P * m_sp_no_yaw
        #   we actually already have "u_init" = P*m_sp if we want
        #   but let's do a direct version for clarity
        batch_size = m_sp_batch.shape[0]
        num_rotors = self.P.shape[0]

        P_expanded = self.P.unsqueeze(0).expand(batch_size, num_rotors, 4)  # [batch, num_rotors,4]
        m_sp_no_yaw_3d = m_sp_no_yaw.view(batch_size, 4, 1)                 # [batch,4,1]
        u_no_yaw = torch.bmm(P_expanded, m_sp_no_yaw_3d).squeeze(-1)       # [batch, num_rotors]

        # 2) use thrust to unsaturate
        thr_vec = self.P[:, 3].unsqueeze(0).expand(batch_size, num_rotors)
        u_unsat = self._minimize_sat(u_no_yaw, thr_vec, u_min, u_max, reduce_only=False)

        # 3) mix yaw axis independently
        out_final = self._mix_yaw(m_sp_batch, u_unsat, u_min, u_max)
        return out_final

    def _airmode_rpy(self, m_sp_batch, u_init, u_min, u_max):
        """
        'airmode_rpy' => thrust can be used for roll/pitch/yaw as well.
        """
        # step 1) direct full mixing => P * m_sp
        #   but we've already done that as "u_init"
        u = u_init

        # step 2) use thrust to unsaturate
        batch_size = m_sp_batch.shape[0]
        thr_vec = self.P[:, 3].unsqueeze(0).expand(batch_size, -1)
        u_prime = self._minimize_sat(u, thr_vec, u_min, u_max, reduce_only=False)

        # step 3) unsaturate yaw => do not let yaw push us outside [u_min, u_max] if already saturated
        yaw_vec = self.P[:, 2].unsqueeze(0).expand(batch_size, -1)
        u_prime2 = self._minimize_sat(u_prime, yaw_vec, u_min, u_max, reduce_only=False)

        return u_prime2

    def _normal_mode(self, m_sp_batch, u_init, u_min, u_max):
        """
        'normal_mode' => thrust is never increased to meet roll/pitch/yaw. We can only reduce thrust if we saturate.
        Yaw mixing is done separately, with up to +15% overhead, but we do not let thrust go up from the original.
        This reproduces the Python code's 'normal_mode' function.
        """
        batch_size = m_sp_batch.shape[0]
        num_rotors = self.P.shape[0]

        # 1) remove yaw
        m_sp_no_yaw = m_sp_batch.clone()
        m_sp_no_yaw[:, 2] = 0.0
        P_expanded = self.P.unsqueeze(0).expand(batch_size, num_rotors, 4)
        u_no_yaw = torch.bmm(P_expanded, m_sp_no_yaw.view(batch_size,4,1)).squeeze(-1)

        # 2) only reduce thrust
        thr_vec = self.P[:, 3].unsqueeze(0).expand(batch_size, num_rotors)
        u_prime = self._minimize_sat(u_no_yaw, thr_vec, u_min, u_max, reduce_only=True)

        # 3) reduce roll/pitch acceleration if needed
        #    first roll: P[:,0]
        roll_vec = self.P[:, 0].unsqueeze(0).expand(batch_size, num_rotors)
        u_p2 = self._minimize_sat(u_prime, roll_vec, u_min, u_max, reduce_only=False)
        #    then pitch: P[:,1]
        pitch_vec = self.P[:, 1].unsqueeze(0).expand(batch_size, num_rotors)
        u_p3 = self._minimize_sat(u_p2, pitch_vec, u_min, u_max, reduce_only=False)

        # 4) mix yaw
        out_final = self._mix_yaw(m_sp_batch, u_p3, u_min, u_max)
        return out_final


###############################################################################
#  3) DEMO / TEST
###############################################################################
def demo_torch_mixer():
    """
    A small test that shows how the TorchMixer can be used
    with different airmodes. We'll do a step in roll/pitch, see how it saturates.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Example of a standard quad_X geometry (like P1 in the original script).
    # Each row is [roll_scale, pitch_scale, yaw_scale, thrust_scale].
    # This is "quad_x": 4 rotors.
    quad_x = np.array([
        [-0.71,  0.71,  1.0,  1.0],
        [ 0.71, -0.71,  1.0,  1.0],
        [ 0.71,  0.71, -1.0,  1.0],
        [-0.71, -0.71, -1.0,  1.0]
    ], dtype=np.float32)

    # Create the mixer
    # You can switch between "none", "rp", "rpy"
    mixer = TorchMixer(quad_x, airmode="none").to(device)

    # We'll create a batch of N=100 sets of control demands.
    # For example, let's ramp from small roll+pitch to larger roll+pitch
    N = 100
    p_sp = torch.linspace(-0.2, 0.2, N, device=device)
    q_sp = torch.linspace( 0.0, 0.2, N, device=device)
    r_sp = torch.zeros(N, device=device)
    T_sp = torch.zeros(N, device=device)  # we keep thrust=0 to see the effect

    # Combine them: shape [N,4]
    m_sp_batch = torch.stack([p_sp, q_sp, r_sp, T_sp], dim=-1)

    # Forward through the mixer
    outputs = mixer(m_sp_batch, u_min=0.0, u_max=1.0)  # shape [N,4]

    # We'll plot the results
    outputs_np = outputs.detach().cpu().numpy()  # [N,4]
    plt.figure()
    for i in range(outputs_np.shape[1]):
        plt.plot(outputs_np[:, i], label=f"Motor {i}")
    plt.title("Mixer outputs (airmode='none'), ramp in roll/pitch, thrust=0")
    plt.xlabel("Timestep")
    plt.ylabel("Motor output")
    plt.legend()
    plt.grid(True)
    plt.show()

def test_rate_controller():
    """
    Just re-uses the RateController test from the question for completeness.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device for RateController test:", device)

    num_envs = 10
    dt = 0.001
    controller = RateController(num_envs, dt).to(device)

    T = 3000
    alpha = 0.1

    outputs_all = []
    integrators_all = []
    errors_all = []

    current_rate = torch.zeros(num_envs, 3, device=device)

    for t in range(T):
        if t < 50:
            target_rate = torch.zeros(num_envs, 3, device=device)
        else:
            target_rate = torch.tensor([0.1, 0.0, 0.0], device=device).expand(num_envs, 3)

        angular_accel = torch.zeros_like(current_rate, device=device)
        landed = torch.zeros(num_envs, dtype=torch.bool, device=device)

        output = controller(current_rate, target_rate, angular_accel, landed)
        outputs_all.append(output.clone().cpu().numpy())
        integrators_all.append(controller.rate_int.clone().cpu().numpy())
        errors_all.append((target_rate - current_rate).clone().cpu().numpy())

        # Simulate some process
        current_rate = alpha * current_rate + (1.0 - alpha) * output

    # Plot a single environment's data
    outputs_all = np.array(outputs_all)  # [T, num_envs, 3]
    integrators_all = np.array(integrators_all)
    errors_all = np.array(errors_all)

    roll_output = outputs_all[:, 0, 0]       # env=0
    roll_integrator = integrators_all[:, 0, 0]
    roll_error = errors_all[:, 0, 0]

    step_input = np.zeros(T)
    step_input[50:] = 0.1

    plt.figure(figsize=(10, 6))

    plt.subplot(3,1,1)
    plt.plot(step_input, 'k--', label="Target Rate (Step=0.1)")
    plt.plot(roll_output, label="Roll Output")
    plt.title("Roll Output vs. Target (Env=0)")
    plt.xlabel("Timestep")
    plt.ylabel("Output")
    plt.grid(True)
    plt.legend()

    plt.subplot(3,1,2)
    plt.plot(roll_integrator, color="orange", label="Roll Integrator")
    plt.title("Integrator Value (Env=0)")
    plt.xlabel("Timestep")
    plt.ylabel("Integrator")
    plt.grid(True)
    plt.legend()

    plt.subplot(3,1,3)
    plt.plot(roll_error, color="red", label="Roll Error")
    plt.title("Roll Error (Env=0)")
    plt.xlabel("Timestep")
    plt.ylabel("Error")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Demo for the Mixer
    demo_torch_mixer()

    # Demo for the Rate Controller
    test_rate_controller()
