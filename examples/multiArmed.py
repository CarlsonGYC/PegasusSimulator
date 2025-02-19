import asyncio
from mavsdk import System


async def arm_and_takeoff(drone: System, drone_id: int):
    # 等待无人机连接成功
    print(f"[Drone {drone_id}] 正在等待连接...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print(f"[Drone {drone_id}] 已连接!")
            break

    # 等待全球定位和 home 位置确认，确保无人机准备好执行任务
    print(f"[Drone {drone_id}] 正在等待全球位置和 home 位置确认...")
    async for health in drone.telemetry.health():
        print(f"[Drone {drone_id}] health:", health)
        # if health.is_global_position_ok and health.is_home_position_ok:
        if health.is_global_position_ok and health.is_armable:
            print(f"[Drone {drone_id}] 全球位置和 arm 确认 OK")
            break
        else:
            print(f"[Drone {drone_id}] 未通过健康检查，等待中...")
            await asyncio.sleep(1)

    # 解锁无人机（arm）
    print(f"[Drone {drone_id}] 正在解锁 (arm)...")
    await drone.action.arm()
    # 等待确保解锁完成
    await asyncio.sleep(10)

    # 起飞（takeoff）
    print(f"[Drone {drone_id}] 正在起飞 (takeoff)...")
    await drone.action.takeoff()
    # 等待5秒让无人机有足够时间起飞到指定高度
    # await asyncio.sleep(5)


async def main():
    drone_addresses = [
        "udp://127.0.0.1:18570",
        "udp://127.0.0.1:18571",
        "udp://127.0.0.1:18572",
        "udp://127.0.0.1:18573",
        "udp://127.0.0.1:18574",
        "udp://127.0.0.1:18575",
    ]

    drones = []
    for addr in drone_addresses:
        drone = System()
        print(f"正在连接 {addr} ...")
        await drone.connect(system_address=addr)
        drones.append(drone)

    # 为每台无人机创建任务，依次进行解锁和起飞操作
    tasks = [arm_and_takeoff(drone, idx + 1)
             for idx, drone in enumerate(drones)]

    # 使用 asyncio.gather 实现并发执行，使所有无人机能够同时控制
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())
