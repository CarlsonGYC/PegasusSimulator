################################################################################################
# @File DroneKitPX4.py
# Example usage of DroneKit with PX4
#
# @author Sander Smeets <sander@droneslab.com>
#
# Code partly based on DroneKit (c) Copyright 2015-2016, 3D Robotics.
################################################################################################

# Import DroneKit-Python
from dronekit import connect, Command, LocationGlobal
from pymavlink import mavutil
import time
import sys
import argparse
import math


def PX4setMode(mavMode, vehicle):
    vehicle._master.mav.command_long_send(vehicle._master.target_system, vehicle._master.target_component,
                                          mavutil.mavlink.MAV_CMD_DO_SET_MODE, 0,
                                          mavMode,
                                          0, 0, 0, 0, 0, 0)


def get_location_offset_meters(original_location, dNorth, dEast, alt):
    """
    Returns a LocationGlobal object containing the latitude/longitude `dNorth` and `dEast` metres from the
    specified `original_location`. The returned Location adds the entered `alt` value to the altitude of the `original_location`.
    The function is useful when you want to move the vehicle around specifying locations relative to
    the current vehicle position.
    The algorithm is relatively accurate over small distances (10m within 1km) except close to the poles.
    For more information see:
    http://gis.stackexchange.com/questions/2951/algorithm-for-offsetting-a-latitude-longitude-by-some-amount-of-meters
    """
    earth_radius = 6378137.0  # Radius of "spherical" earth
    # Coordinate offsets in radians
    dLat = dNorth/earth_radius
    dLon = dEast/(earth_radius*math.cos(math.pi*original_location.lat/180))

    # New position in decimal degrees
    newlat = original_location.lat + (dLat * 180/math.pi)
    newlon = original_location.lon + (dLon * 180/math.pi)
    return LocationGlobal(newlat, newlon, original_location.alt+alt)


def load_commands(vehicle, index):
    # Load commands
    cmds = vehicle.commands
    cmds.clear()

    home = vehicle.location.global_relative_frame
    print("vehicle{}".format(index))
    print("Home lat: %s, lon: %s" % (home.lat, home.lon))

    # takeoff to 10 meters
    wp = get_location_offset_meters(home, 0, 0, 10)
    cmd = Command(0, 0, 0, mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                  mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 1, 0, 0, 0, 0, wp.lat, wp.lon, wp.alt)
    cmds.add(cmd)

    # move 2 meters north
    wp = get_location_offset_meters(wp, 2, 0, 0)
    cmd = Command(0, 0, 0, mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                  mavutil.mavlink.MAV_CMD_NAV_WAYPOINT, 0, 1, 0, 0, 0, 0, wp.lat, wp.lon, wp.alt)
    cmds.add(cmd)

    # move 2 meters east
    wp = get_location_offset_meters(wp, 0, 2, 0)
    cmd = Command(0, 0, 0, mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                  mavutil.mavlink.MAV_CMD_NAV_WAYPOINT, 0, 1, 0, 0, 0, 0, wp.lat, wp.lon, wp.alt)
    cmds.add(cmd)

    # move 2 meters south
    wp = get_location_offset_meters(wp, -2, 0, 0)
    cmd = Command(0, 0, 0, mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                  mavutil.mavlink.MAV_CMD_NAV_WAYPOINT, 0, 1, 0, 0, 0, 0, wp.lat, wp.lon, wp.alt)
    cmds.add(cmd)

    # move 2 meters west
    wp = get_location_offset_meters(wp, 0, -2, 0)
    cmd = Command(0, 0, 0, mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                  mavutil.mavlink.MAV_CMD_NAV_WAYPOINT, 0, 1, 0, 0, 0, 0, wp.lat, wp.lon, wp.alt)
    cmds.add(cmd)

    # land
    wp = get_location_offset_meters(home, 0, 0, 10)
    cmd = Command(0, 0, 0, mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                  mavutil.mavlink.MAV_CMD_NAV_LAND, 0, 1, 0, 0, 0, 0, wp.lat, wp.lon, wp.alt)
    cmds.add(cmd)

    # Upload mission
    cmds.upload()
    print(cmds)
    time.sleep(2)
    print("Mission uploaded for vehicle{}".format(index))

# Command sequence to make the vehicle perform a circle


def load_circle(vehicle, index, radius=5, num_waypoints=36):
    cmds = vehicle.commands
    cmds.clear()

    home = vehicle.location.global_relative_frame
    print("vehicle{}".format(index))
    print("Home lat: %s, lon: %s" % (home.lat, home.lon))

    wp = get_location_offset_meters(home, 0, 0, 10)
    cmd = Command(0, 0, 0, mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                  mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 1, 0, 0, 0, 0, wp.lat, wp.lon, wp.alt)
    cmds.add(cmd)

    for i in range(num_waypoints):
        angle = i * (360.0 / num_waypoints)
        dNorth = radius * math.cos(math.radians(angle))
        dEast = radius * math.sin(math.radians(angle))
        wp = get_location_offset_meters(home, dNorth, dEast, 10)
        cmd = Command(0, 0, 0, mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                      mavutil.mavlink.MAV_CMD_NAV_WAYPOINT, 0, 1, 0, 0, 0, 0, wp.lat, wp.lon, wp.alt)
        cmds.add(cmd)

    wp = get_location_offset_meters(home, 0, 0, 10)
    cmd = Command(0, 0, 0, mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                  mavutil.mavlink.MAV_CMD_NAV_LAND, 0, 1, 0, 0, 0, 0, wp.lat, wp.lon, wp.alt)
    cmds.add(cmd)

    cmds.upload()
    print(cmds)
    time.sleep(2)
    print("Mission uploaded for vehicle{}".format(index))

# Arm all vehicle


def arm_vehicles(vehicle_list):
    for i in range(UAV_NUM):
        print("Arming vehicle{}".format(i))
        vehicle_list[i].armed = True
        time.sleep(0.1)

# Change to AUTO mode


def change_mode_auto(vehicle_list):
    for i in range(UAV_NUM):
        PX4setMode(MAV_MODE_AUTO, vehicle_list[i])
        time.sleep(1)


def wait_vehicles_ready(vehicle_list):
    for i in range(UAV_NUM):
        while not vehicle_list[i].armed:
            print("Waiting for vehicle{} to be ready".format(i))
            time.sleep(1)
    print("All vehicles ready")


def wait_home_position(vehicle_list):
    for i in range(UAV_NUM):
        while not home_position_set:
            print("Waiting for home position for vehicle{}".format(i))
            time.sleep(1)
    print("All home position set")

################################################################################################
# Settings
################################################################################################


connection_string = '127.0.0.1:14540'
UAV_NUM = 6
MAV_MODE_AUTO = 4
# https://github.com/PX4/PX4-Autopilot/blob/master/Tools/mavlink_px4.py

################################################################################################
# start
################################################################################################

# Connect to the Vehicle
print("Connecting")
vehicle_list = []
for i in range(UAV_NUM):
    connection_string = 'udpin:localhost:{}'.format(14540+i)
    print('conneting to vehicle on: %s' % connection_string)
    vehicle = connect(connection_string, wait_ready=True)

    vehicle_list.append(vehicle)

################################################################################################
# Listeners
################################################################################################

home_position_set = False
for i in range(UAV_NUM):
    vehicle_list[i].home_position_set = False


def listener(self, name, home_position):
    self.home_position_set = True


# Create a message listener for home position fix
for i in range(UAV_NUM):
    vehicle_list[i].add_message_listener('HOME_POSITION', listener)

################################################################################################
# Start mission example
################################################################################################

# wait for a home position lock
while not vehicle_list[0].home_position_set or not vehicle_list[1].home_position_set or not vehicle_list[2].home_position_set:
    print("Waiting for home position...")
    print("vehicle0: %s" % vehicle_list[0].home_position_set)
    print("vehicle1: %s" % vehicle_list[1].home_position_set)
    print("vehicle2: %s" % vehicle_list[2].home_position_set)
    time.sleep(1)


# Display basic vehicle state
for i in range(UAV_NUM):
    print('Vehicle{}'.format(i))
    print(" Type: %s" % vehicle_list[i]._vehicle_type)
    print(" Armed: %s" % vehicle_list[i].armed)
    print(" System status: %s" % vehicle_list[i].system_status.state)
    print(" GPS: %s" % vehicle_list[i].gps_0)
    print(" Alt: %s" % vehicle_list[i].location.global_relative_frame.alt)

# upload mission to each vehicle
for i in range(UAV_NUM):
    load_commands(vehicle_list[i], i)
    # load_circle(vehicle_list[i], i)


# arm all vehicles
change_mode_auto(vehicle_list)
arm_vehicles(vehicle_list)

# monitor mission execution
nextwaypoint = vehicle.commands.next
while nextwaypoint < len(vehicle.commands):
    if vehicle.commands.next > nextwaypoint:
        display_seq = vehicle.commands.next+1
        print("Moving to waypoint %s" % display_seq)
        nextwaypoint = vehicle.commands.next
    time.sleep(1)

# wait for the vehicle to land
while vehicle.commands.next > 0:
    time.sleep(1)


# # Disarm vehicle
# vehicle.armed = False
# time.sleep(1)

# # Close vehicle object before exiting script
# vehicle.close()
# time.sleep(1)
