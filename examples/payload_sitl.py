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
from pegasus.simulator.logic.backends.px4_mavlink_backend import PX4MavlinkBackend, PX4MavlinkBackendConfig
from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS, ROBOTS_CONFIG
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
import os.path
from scipy.spatial.transform import Rotation


class PegasusApp:
    def __init__(self):
        # 获取 timeline
        self.timeline = omni.timeline.get_timeline_interface()

        # 初始化 Pegasus 接口
        self.pg = PegasusInterface()
        self.pg._world = World(**self.pg._world_settings)
        self.world = self.pg.world

        stage = self.world.stage
        physics_context = self.world.get_physics_context()
        physics_context.enable_gpu_dynamics(True)
        physics_context.enable_stablization(False)
        physics_context.enable_ccd(False)
        PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath("/physicsScene"))
        physxSceneAPI = PhysxSchema.PhysxSceneAPI.Get(stage, "/physicsScene")

        # 加载环境和模型
        self.pg.load_environment(SIMULATION_ENVIRONMENTS["Curved Gridroom"])
        # self.pg.load_asset(ROBOTS["Single Cable"], "/World/cable")

        config_multirotor = MultirotorConfig()
        # config_multirotor = self.pg.generate_quadrotor_config_from_yaml(ROBOTS_CONFIG["Raynor"]) # Load Raynor's configuration
        # self.pg.load_asset(ROBOTS["Single Cable"], "/World/cable")
        # self.pg.load_asset(ROBOTS["Cable"], "/World/quadrotor")
        
        # Create the multirotor configuration
        vehicle_id = 0
        mavlink_config = PX4MavlinkBackendConfig({
            "vehicle_id": vehicle_id,
            "px4_autolaunch": True,
            "px4_dir": self.pg.px4_path,
            "px4_vehicle_model": 'iris' # CHANGE this line to 'iris' if using PX4 version bellow v1.14
        })
        config_multirotor.backends = [PX4MavlinkBackend(mavlink_config)]

        Multirotor(
            "/World/quadrotor",
            # ROBOTS['Raynor'],
            # ROBOTS['Iris'],
            ROBOTS['Cable'],
            0,
            [0.0, 0.0, 0.11],
            Rotation.from_euler("XYZ", [0.0, 0.0, 0.0], degrees=True).as_quat(),
            config=config_multirotor,
        )
        
        
        # Reset the simulation environment so that all articulations (aka robots) are initialized
        self.world.reset()

        # Auxiliar variable for the timeline callback example
        self.stop_sim = False

    def run(self):
        """
        Method that implements the application main loop, where the physics steps are executed.
        """

        # Start the simulation
        self.timeline.play()

        # The "infinite" loop
        while simulation_app.is_running() and not self.stop_sim:

            # Update the UI of the app and perform the physics step
            self.world.step(render=True)
        
        # Cleanup and stop
        carb.log_warn("PegasusApp Simulation App is closing.")
        self.timeline.stop()
        simulation_app.close()

def main():

    # Instantiate the template app
    pg_app = PegasusApp()

    # Run the application loop
    pg_app.run()

if __name__ == "__main__":
    main()