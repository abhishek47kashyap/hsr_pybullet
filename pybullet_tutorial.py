import time

import numpy as np
import pybullet as p
import pybullet_data

P_GAIN = 50
# desired_joint_positions = np.array([1.218, 0.507, -0.187, 1.235, 0.999, 1.279, 0])

def main():
    p.connect(p.GUI)

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf")

    robot_id = p.loadURDF("hsrb_description/robots/hsrb.urdf", useFixedBase=True, flags=p.URDF_USE_SELF_COLLISION)

    n_joints = p.getNumJoints(robot_id)
    joints = [p.getJointInfo(robot_id, i) for i in range(n_joints)]
    joint_indices = range(n_joints)

    print("Joint information:")
    for joint in joints:
        idx = joint[0]
        name = joint[1]
        joint_type = joint[2]
        lower_limit = joint[8]
        upper_limit = joint[9]
        link_name = joint[12]
        print(f"  Joint {idx}: {name}, type: {joint_type}, limits: lower {lower_limit}, upper: {upper_limit}, link-name: {link_name}")

    # The magic that enables torque control
    p.setJointMotorControlArray(
        bodyIndex=robot_id,
        jointIndices=joint_indices,
        controlMode=p.TORQUE_CONTROL,
        forces=np.zeros(n_joints),
    )

    

    while True:
        time.sleep(0.01)

        # joint_states = p.getJointStates(robot_id, joint_indices)
        # joint_positions = np.array([j[0] for j in joint_states])
        # error = desired_joint_positions - joint_positions
        # torque = error * P_GAIN

        # p.setJointMotorControlArray(
        #     bodyIndex=robot_id,
        #     jointIndices=joint_indices,
        #     controlMode=p.TORQUE_CONTROL,
        #     forces=torque,
        # )

        p.stepSimulation()

if __name__ == "__main__":
    main()