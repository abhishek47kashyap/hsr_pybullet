
import time
from copy import deepcopy
from enum import Enum

import numpy as np
from scipy.spatial.transform import Rotation as R
import trimesh

import pybullet as p
import pybullet_data
import pybullet_utils.bullet_client as bc
import pybulletX as px

def get_joint_limits(robot_body_unique_id, verbose=False):
    joint_infos = robot_body_unique_id.get_joint_infos()
    joint_names = [x.decode('utf-8') for x in joint_infos["joint_name"]]
    joint_lower_limits = joint_infos["joint_lower_limit"]
    joint_upper_limits = joint_infos["joint_upper_limit"]

    if verbose:
        print("Joint limits:")
        for name, lower, upper in zip(joint_names, joint_lower_limits, joint_upper_limits):
            print(f"\t{name}: [{lower}, {upper}]")

    return joint_names, joint_lower_limits, joint_upper_limits

def set_joint_positions(bullet_client, robot_body_unique_id, joint_positions, joint_velocities, set_joint_motor_control_array: bool = True):
    if set_joint_motor_control_array:
        bullet_client.setJointMotorControlArray(
            targetPositions=joint_positions,
            bodyUniqueId=robot_body_unique_id.id,
            jointIndices=robot_body_unique_id.free_joint_indices,
            controlMode=p.POSITION_CONTROL,
            targetVelocities=joint_velocities
        )

    else:
        joint_infos = robot_body_unique_id.get_joint_infos()
        all_joints_max_velocities = joint_infos['joint_max_velocity']
        all_joints_max_torques = joint_infos['joint_max_force']
        joint_indices_that_overshoot = [0, 1]  # joint_x, joint_y

        for joint_idx, joint_pos, joint_target_vel, joint_max_vel, joint_torque in zip(robot_body_unique_id.free_joint_indices, joint_positions, joint_velocities, all_joints_max_velocities, all_joints_max_torques):
            if joint_idx in joint_indices_that_overshoot:
                bullet_client.setJointMotorControl2(
                    bodyIndex=robot_body_unique_id.id,
                    jointIndex=joint_idx,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=joint_pos,
                    targetVelocity=joint_target_vel, # joint_max_vel,
                    positionGain=0.028,
                    # maxVelocity=joint_max_vel,   # if uncommented, base moves very slowly
                    # velocityGain=0.001,   # has little/no effect on overshoot
                    force=joint_torque,
                    physicsClientId=bullet_client._client
                    )
            else:
                bullet_client.setJointMotorControl2(
                    bodyIndex=robot_body_unique_id.id,
                    jointIndex=joint_idx,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=joint_pos,
                    targetVelocity=joint_max_vel,
                    # maxVelocity=joint_max_vel,   # if uncommented, base moves very slowly
                    force=joint_torque,
                    physicsClientId=bullet_client._client
                    )

def main():
    urdf_file_path = "hsrb_description/robots/hsrb.urdf"
    use_fixed_base = True
    connection_mode = p.GUI

    bullet_client = bc.BulletClient(connection_mode=connection_mode)
    px_client = px.Client(client_id=bullet_client._client)

    bullet_client.setAdditionalSearchPath(pybullet_data.getDataPath())
    bullet_client.loadURDF("plane.urdf")

    bullet_client.setGravity(0, 0, -9.8)

    robot_body_unique_id = px.Robot(
        urdf_file_path,
        use_fixed_base=use_fixed_base,
        physics_client=px_client,
        base_position=[0.0, 0.0, 0.0],
        flags=p.URDF_USE_SELF_COLLISION
        )
    robot_body_unique_id.torque_control = False

    n_joints = robot_body_unique_id.num_joints
    n_dofs = robot_body_unique_id.num_dofs
    print(f"Number of joints: {n_joints}, number of dofs: {n_dofs}")
    joint_names, joint_lower_limits, joint_upper_limits = get_joint_limits(robot_body_unique_id, verbose=True)

    setJointMotorControlArray_args = {
        "bodyUniqueId": robot_body_unique_id.id,
        "jointIndices": robot_body_unique_id.free_joint_indices,
        "controlMode": p.POSITION_CONTROL,
        "targetVelocities": np.zeros(n_dofs)
    }

    debug_param_ids = [bullet_client.addUserDebugParameter(paramName=name, rangeMin=lower, rangeMax=upper, startValue=0.0, physicsClientId=bullet_client._client) 
                            for name, lower, upper in zip(joint_names, joint_lower_limits, joint_upper_limits)]

    while True:
        bullet_client.stepSimulation()
        joint_values_from_sliders = [bullet_client.readUserDebugParameter(itemUniqueId=id, physicsClientId=bullet_client._client) for id in debug_param_ids]

        set_joint_positions(
            bullet_client = bullet_client,
            robot_body_unique_id = robot_body_unique_id,
            joint_positions = joint_values_from_sliders,
            joint_velocities = np.zeros(n_dofs),
            set_joint_motor_control_array=True
        )


if __name__ == "__main__":
    main()