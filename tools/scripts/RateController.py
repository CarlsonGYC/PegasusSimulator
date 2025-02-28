import torch
import torch.nn as nn

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


import numpy as np
import matplotlib.pyplot as plt

def test_rate_controller():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    num_envs = 10
    dt = 0.001
    controller = RateController(num_envs, dt).to(device)

    T = 30000
    outputs_all = []
    integrators_all = []
    errors_all = []

    current_rate = torch.zeros(num_envs, 3, device=device)
    alpha = 0.1

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

        current_rate = alpha * current_rate + (1.0 - alpha) * output

    outputs_all = np.array(outputs_all)
    integrators_all = np.array(integrators_all)
    errors_all = np.array(errors_all)

    roll_output = outputs_all[:, 0, 0]
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
    test_rate_controller()
