import torch
import torch.nn as nn

class BodyRateController(nn.Module):
    def __init__(self, num_envs: int, Kp=None, Ki=None, Kd=None, Kff=None, integrator_limit=None, output_limit=None, device=None, dt=None):
        self.device = torch.device("cuda")
        default_Kp = torch.tensor([0.15, 0.15, 0.2], dtype=torch.float32)
        default_Ki = torch.tensor([0.2, 0.2, 0.1], dtype=torch.float32)
        default_Kd = torch.tensor([0.003, 0.003, 0.0], dtype=torch.float32)  # roll/pitch
        default_Kff= torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)  # feedforward, default 0
        default_int_lim = torch.tensor([0.3, 0.3, 0.3], dtype=torch.float32)  # output max
        default_out_lim = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)  # normalize
        default_dt = torch.tensor(0.001)
        
        def init_param(param, default):
            if param is None:
                tensor = default.clone()
            else:
                tensor = torch.tensor(param, dtype=torch.float32)
            return tensor.to(self.device)
        self.Kp = init_param(Kp, default_Kp)
        self.Ki = init_param(Ki, default_Ki)
        self.Kd = init_param(Kd, default_Kd)
        self.Kff= init_param(Kff, default_Kff)
        self.dt = init_param(dt, default_dt)
        # Limits
        self.integrator_limit = init_param(integrator_limit, default_int_lim)  # max integrator effect
        self.output_limit = init_param(output_limit, default_out_lim)      # absolute output limit
        self.num_envs = num_envs    # input number of envs ????????
        self.integrator = torch.zeros((num_envs, 3), dtype=torch.float32, device=self.device)   # I term
        self.last_error = torch.zeros((num_envs, 3), dtype=torch.float32, device=self.device)   
    
    # def update_gains(self, Kp=None, Ki=None, Kd=None, Kff=None, integrator_limit=None, output_limit=None):
    #     if Kp is not None:
    #         self.Kp = torch.tensor(Kp, dtype=torch.float32).view(-1)[:3].to(self.device)
    #     if Ki is not None:
    #         self.Ki = torch.tensor(Ki, dtype=torch.float32).view(-1)[:3].to(self.device)
    #     if Kd is not None:
    #         self.Kd = torch.tensor(Kd, dtype=torch.float32).view(-1)[:3].to(self.device)
    #     if Kff is not None:
    #         self.Kff = torch.tensor(Kff, dtype=torch.float32).view(-1)[:3].to(self.device)
    #     if integrator_limit is not None:
    #         self.integrator_limit = torch.tensor(integrator_limit, dtype=torch.float32).view(-1)[:3].to(self.device)
    #     if output_limit is not None:
    #         self.output_limit = torch.tensor(output_limit, dtype=torch.float32).view(-1)[:3].to(self.device)
    #     # Ensure all internal tensors are on correct device
    #     self.Kp = self.Kp.to(self.device)
    #     self.Ki = self.Ki.to(self.device)
    #     self.Kd = self.Kd.to(self.device)
    #     self.Kff= self.Kff.to(self.device)
    #     self.integrator_limit = self.integrator_limit.to(self.device)
    #     self.output_limit = self.output_limit.to(self.device)
    
    def reset(self, env_indices=None):
        if env_indices is None:
            self.integrator.zero_()
            self.last_error.zero_()
        else:
            env_indices = torch.tensor(env_indices)
            self.integrator.index_fill_(0, env_indices, 0.0)
            self.last_error.index_fill_(0, env_indices, 0.0)
    
    def control(self, rates: torch.Tensor, setpoints: torch.Tensor):
        dt = self.dt
        rates = rates.to(self.device).view(self.num_envs, 3)    # body rate, shape (N,3)
        setpoints = setpoints.to(self.device).view(self.num_envs, 3)    # desired rate, shape (N,3)
        error = setpoints - rates  

        P_term = error * self.Kp  # Broadcast, 3 * [N, 3]
        D_term = (error - self.last_error) * (self.Kd / dt)  
        FF_term = setpoints * self.Kff  # Feedforward, (N,3)
        I_term = self.integrator * self.Ki  # Integral term: update integrator with anti-windup

        output_raw = P_term + I_term + D_term + FF_term  # (N,3), without anti-windup
        
        high_limit = self.output_limit 
        low_limit  = -self.output_limit
        satur_high = output_raw > high_limit   # boolean 
        satur_low  = output_raw < low_limit    # boolean (N,3)
        
        # Anti-windup
        error_dt = error * dt
        inhibit_high = satur_high & (error_dt > 0)
        inhibit_low = satur_low & (error_dt < 0)
        inhibit_integration = inhibit_high | inhibit_low  # (N,3), NOT integrate
        effective_error_dt = torch.where(inhibit_integration, torch.zeros_like(error_dt), error_dt) # (N,3)
        
        # Update integrator state
        self.integrator += effective_error_dt  
        # Clamp integrator to prevent windup: 
        integrator_max = torch.where(self.Ki != 0, self.integrator_limit / (self.Ki + 1e-8), 0.0)  # (3,) avoid divide by zero.
        high_int_mask = self.integrator > integrator_max   # (N,3)
        low_int_mask = self.integrator < -integrator_max
        self.integrator = torch.where(high_int_mask, integrator_max.expand_as(self.integrator), self.integrator)
        self.integrator = torch.where(low_int_mask, (-integrator_max).expand_as(self.integrator), self.integrator)
        
        I_term = self.integrator * self.Ki  # (N,3) updated integral contribution
        output = P_term + I_term + D_term + FF_term
        output = torch.max(torch.min(output, high_limit), low_limit)  # clamp each axis between its low and high limit
        
        self.last_error = error.detach().clone()
        
        return output


import numpy as np
import matplotlib.pyplot as plt

def test_rate_controller():
    # 选择设备：GPU优先，没有则使用CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 创建控制器
    num_envs = 10
    dt = 0.001
    controller = BodyRateController(num_envs = num_envs, dt = dt)

    # 测试时长
    T = 30000
    # 记录
    outputs_all = []
    integrators_all = []
    errors_all = []

    # 初始化系统状态: current_rate
    current_rate = torch.zeros(num_envs, 3, device=device)

    # 一个简单的一阶系统常数, 用于模拟实际系统（使输出对控制量有个缓冲作用）
    alpha = 0.8

    for t in range(T):
        # 从第50步开始，目标从0变为1（只给roll通道一个阶跃；pitch、yaw为0）
        if t < 50:
            target_rate = torch.zeros(num_envs, 3, device=device)
        else:
            # 目标为 [1, 0, 0]
            target_rate = torch.tensor([0.1, 0.0, 0.0], device=device).expand(num_envs, 3)

        # 控制器输出
        output = controller.control(current_rate, target_rate)

        # 记录
        outputs_all.append(output.clone().cpu().numpy())  
        integrators_all.append(controller.integrator.clone().cpu().numpy())
        errors_all.append((target_rate - current_rate).clone().cpu().numpy())

        # 用简单的一阶模型更新 current_rate，模拟系统行为
        #   next_rate = alpha * current_rate + (1 - alpha) * output
        # 这样可以看到控制器如何把当前速率逼近 target_rate = [1, 0, 0]
        current_rate = alpha * current_rate + (1.0 - alpha) * output

    # 转为 numpy 方便可视化
    outputs_all = np.array(outputs_all)       # (T, num_envs, 3)
    integrators_all = np.array(integrators_all)
    errors_all = np.array(errors_all)

    # 只绘制第0号环境的 roll 通道 为演示
    #    roll = channel 0
    roll_output = outputs_all[:, 0, 0]
    roll_integrator = integrators_all[:, 0, 0]
    roll_error = errors_all[:, 0, 0]

    # 构造阶跃输入序列 (0 ~ 49 为0，50~T-1 为1)
    step_input = np.zeros(T)
    step_input[50:] = 1.0

    # --- 画图 ---
    plt.figure(figsize=(10, 6))

    plt.subplot(3,1,1)
    # plt.plot(step_input, 'k--', label="Target (step=1)")
    plt.plot(roll_output, label="Roll output")
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