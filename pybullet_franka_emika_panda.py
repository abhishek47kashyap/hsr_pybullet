"""
Reference:
- Part 1: https://www.etedal.net/2020/04/pybullet-panda.html
- Part 2: https://www.etedal.net/2020/04/pybullet-panda_2.html
- Part 3: https://www.etedal.net/2020/04/pybullet-panda_3.html
"""

import os
import math

import pybullet as p
import pybullet_data

def get_filepath(urdf: str) -> str:
    return os.path.join(pybullet_data.getDataPath(), urdf)

def spawn_robot():
    return p.loadURDF(
        get_filepath("franka_panda/panda.urdf"),
        useFixedBase=True,
        flags=p.URDF_USE_SELF_COLLISION
        )

def spawn_table():
    return p.loadURDF(
        get_filepath("table/table.urdf"),
        basePosition=[0.5,0,-0.65]
        )

def spawn_items_on_table():
    trayUid = p.loadURDF(get_filepath("tray/traybox.urdf"), basePosition=[0.65,0,0])
    objectUid = p.loadURDF(get_filepath("random_urdfs/000/000.urdf"), basePosition=[0.7,0,0.1])

    return trayUid, objectUid

p.connect(p.GUI)
p.setGravity(0,0,-9.81)
p.resetDebugVisualizerCamera(
    cameraDistance=1.5,
    cameraYaw=0,
    cameraPitch=-40,
    cameraTargetPosition=[0.55,-0.35,0.2]
    )

pandaUid = spawn_robot()
tableUid = spawn_table()
trayUid, objectUid = spawn_items_on_table()

state_durations = [1,1,1,1]
control_dt = 1./240.
p.setTimestep = control_dt
state_t = 0.
current_state = 0

while True:
    p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING) 
    
    state_t += control_dt

    if current_state == 0:
        p.setJointMotorControl2(pandaUid, 0, 
                        p.POSITION_CONTROL,0)
        p.setJointMotorControl2(pandaUid, 1, 
                        p.POSITION_CONTROL,math.pi/4.)
        p.setJointMotorControl2(pandaUid, 2, 
                        p.POSITION_CONTROL,0)
        p.setJointMotorControl2(pandaUid, 3, 
                        p.POSITION_CONTROL,-math.pi/2.)
        p.setJointMotorControl2(pandaUid, 4, 
                        p.POSITION_CONTROL,0)
        p.setJointMotorControl2(pandaUid, 5, 
                        p.POSITION_CONTROL,3*math.pi/4)
        p.setJointMotorControl2(pandaUid, 6, 
                        p.POSITION_CONTROL,-math.pi/4.)
        p.setJointMotorControl2(pandaUid, 9, 
                        p.POSITION_CONTROL, 0.08)
        p.setJointMotorControl2(pandaUid, 10, 
                        p.POSITION_CONTROL, 0.08)
    if current_state == 1:
        p.setJointMotorControl2(pandaUid, 1, 
                        p.POSITION_CONTROL,math.pi/4.+.15)
        p.setJointMotorControl2(pandaUid, 3, 
                        p.POSITION_CONTROL,-math.pi/2.+.15)
    if current_state == 2:
        p.setJointMotorControl2(pandaUid, 9, 
                        p.POSITION_CONTROL, 0.0, force = 200)
        p.setJointMotorControl2(pandaUid, 10, 
                        p.POSITION_CONTROL, 0.0, force = 200)
    if current_state == 3:
        p.setJointMotorControl2(pandaUid, 1, 
                        p.POSITION_CONTROL,math.pi/4.-1)
        p.setJointMotorControl2(pandaUid, 3, 
                        p.POSITION_CONTROL,-math.pi/2.-1)

    if state_t >state_durations[current_state]:
        current_state += 1
        if current_state >= len(state_durations):
            current_state = 0
        state_t = 0

    p.stepSimulation()