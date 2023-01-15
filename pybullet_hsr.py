
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

        bullet_client.setJointMotorControlArray(
            targetPositions=joint_values_from_sliders,
            **setJointMotorControlArray_args
        )


if __name__ == "__main__":
    main()