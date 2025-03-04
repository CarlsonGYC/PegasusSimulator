#!/usr/bin/env python
"""
| File: 1_px4_single_vehicle.py
| Author: Marcelo Jacinto (marcelo.jacinto@tecnico.ulisboa.pt)
| License: BSD-3-Clause
"""
import carb
from isaacsim import SimulationApp

# 创建 SimulationApp
simulation_app = SimulationApp({"headless": False})

import omni.timeline
from omni.isaac.core.world import World
from omni.isaac.dynamic_control import _dynamic_control as dc
from pxr import UsdGeom, PhysxSchema
import csv

# 如果需要用 Pegasus 的相关类，可以自行 import
from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface


class PegasusApp:
    def __init__(self):
        # 获取 timeline
        self.timeline = omni.timeline.get_timeline_interface()

        # 初始化 Pegasus 接口
        self.pg = PegasusInterface()
        self.pg._world = World(**self.pg._world_settings)
        self.world = self.pg.world

        # 加载环境和模型
        self.pg.load_environment(SIMULATION_ENVIRONMENTS["Curved Gridroom"])
        self.pg.load_asset(ROBOTS["Single Cable"], "/World/cable")

        # 基础物理设置
        stage = self.world.stage
        physics_context = self.world.get_physics_context()
        physics_context.enable_gpu_dynamics(True)
        PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath("/physicsScene"))
        physxSceneAPI = PhysxSchema.PhysxSceneAPI.Get(stage, "/physicsScene")
        # physxSceneAPI.CreateMinPositionIterationCountAttr(25)
        # physxSceneAPI.CreateMinVelocityIterationCountAttr(0)

        # 重置场景
        self.world.reset()

        # CSV 初始化：写表头
        self.csv_filename = "transform_log.csv"
        self.log_file = open(self.csv_filename, mode="w", newline="")
        self.csv_writer = csv.writer(self.log_file)
        self.csv_writer.writerow(["time", "z_val"])  # 这里只保存 time 和 z 值

        # 记录上一次时间
        self.last_time = 0.0
        self.stop_sim = False

    def run(self):
        self.timeline.play()
        dynamic_control = dc.acquire_dynamic_control_interface()

        capsule_prim_path = "/World/cable/CommonPayload/Payload"
        
        while simulation_app.is_running() and not self.stop_sim:
            # 执行物理步
            self.world.step(render=True)

            current_time = self.world.current_time
            xformCache = UsdGeom.XformCache()
        
            # 如果发现时间“跳回”，视为 reset：重新写 CSV
            if current_time < self.last_time:
                self.log_file.close()
                self.log_file = open(self.csv_filename, mode="w", newline="")
                self.csv_writer = csv.writer(self.log_file)
                self.csv_writer.writerow(["time", "z_val"])

            # ========== 只在 time > 0 时才写入 CSV ========== 
            if self.last_time >= 0 and current_time > 0:
                prim = self.world.stage.GetPrimAtPath(capsule_prim_path)
                world_transform = xformCache.GetLocalToWorldTransform(prim)

                # 提取4x4矩阵中第四行前3列即 (tx, ty, tz)
                z_val = world_transform[3][2]
                print(z_val)
                print(world_transform)
                self.csv_writer.writerow([current_time, z_val])

            self.last_time = current_time

        # 清理与关闭
        carb.log_warn("PegasusApp Simulation App is closing.")
        self.timeline.stop()
        self.log_file.close()
        simulation_app.close()


def main():
    app = PegasusApp()
    app.run()


if __name__ == "__main__":
    main()