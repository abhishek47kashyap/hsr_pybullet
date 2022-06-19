"""
Ref:
- https://towardsdatascience.com/simulate-images-for-ml-in-pybullet-the-quick-easy-way-859035b2c9dd
"""

import time
from copy import deepcopy

import numpy as np
import trimesh

import pybullet as p
import pybullet_data

CAMERA_XTION_CONFIG = {
    'image_size': (480, 640),
    'intrinsics': (537.4933389299223, 0.0, 319.9746375212718, 0.0, 536.5961755975517, 244.54846607953, 0.0, 0.0, 1.0),
    'position': None,
    'orientation': None,
    'zrange': (0.5, 10.),
    'noise': False
}

CAMERA_REALSENSE_CONFIG = {
    'image_size': (480, 640),
    'intrinsics': (
    607.3814086914062, 0.0, 315.9123840332031, 0.0, 607.2514038085938, 233.77308654785156, 0.0, 0.0, 1.0),
    'position': None,
    'orientation': None,
    'zrange': (0.3, 10.),
    'noise': False
}

hand = True   # if false, CAMERA_XTION_CONFIG config will be used

def main():
    urdf_file_path = "hsrb_description/robots/hsrb.urdf"
    use_fixed_base = True
    connection_mode = p.GUI

    camera_joint_frame = 'hand_camera_gazebo_frame_joint' if hand else 'head_rgbd_sensor_gazebo_frame_joint'
    camera_config = deepcopy(CAMERA_REALSENSE_CONFIG if hand else CAMERA_XTION_CONFIG)

    pybullet_client = p.connect(connection_mode)

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf")

    robot_body_unique_id = robot_id = p.loadURDF(
        urdf_file_path,
        useFixedBase=use_fixed_base,
        flags=p.URDF_USE_SELF_COLLISION
        )

    n_joints = p.getNumJoints(robot_id)
    joints = [p.getJointInfo(robot_id, i) for i in range(n_joints)]
    joint_indices = range(n_joints)

    print("Joint information:")
    camera_link_idx = None
    for joint in joints:
        idx = joint[0]
        name = joint[1]
        joint_type = joint[2]
        lower_limit = joint[8]
        upper_limit = joint[9]
        link_name = joint[12]
        print(f"  Joint {idx}: {name}, type: {joint_type}, limits: lower {lower_limit}, upper: {upper_limit}, link-name: {link_name}")

        if name.decode('ASCII') == camera_joint_frame:   # decode() required because name is a byte-string
            camera_link_idx = idx
            print(f"Camera link name is {camera_joint_frame}, link index is {camera_link_idx}")
    if not camera_link_idx:
        print(f"Camera link index not found for frame named {camera_joint_frame}")

    # The magic that enables torque control
    p.setJointMotorControlArray(
        bodyIndex=robot_id,
        jointIndices=joint_indices,
        controlMode=p.TORQUE_CONTROL,
        forces=np.zeros(n_joints),
    )

    model_name = "007_tuna_fish_can"
    mesh_path = 'assets/ycb/{}/google_16k/nontextured.stl'.format(model_name)
    collision_path = 'assets/ycb/{}/google_16k/collision.obj'.format(model_name)
    mesh = trimesh.load(mesh_path, force='mesh', process=False)
    mesh.density = 150
    scale = 1

    centroid = mesh.centroid
    scale = 1
    scale = [scale, scale, scale]

    visual_shape_id = p.createVisualShape(
        shapeType=p.GEOM_MESH,
        rgbaColor=[1, 0, 0, 1],
        specularColor=[0.4, .4, 0],
        fileName=mesh_path, meshScale=scale,
        visualFramePosition=-centroid,
    )
    collision_shape_id = p.createCollisionShape(
        shapeType=p.GEOM_MESH,
        fileName=collision_path, meshScale=scale,
        collisionFramePosition=-centroid,
    )

    obj_id = p.createMultiBody(
        baseMass=mesh.mass,
        basePosition=(1.0, 0, 0),
        baseOrientation=(0, 0, 0, 1),
        baseCollisionShapeIndex=collision_shape_id,
        baseVisualShapeIndex=visual_shape_id,
        baseInertialFramePosition=np.array(mesh.center_mass - centroid),
    )

    p.changeDynamics(obj_id, -1, lateralFriction=0.25)
    p.setGravity(0, 0, -9.8)

    width = 128
    height = 128

    fov = 60
    aspect = width / height
    near = 0.02
    far = 1
    view_matrix = p.computeViewMatrix(
        cameraEyePosition=[0, 0, 0.5],
        cameraTargetPosition=[0, 0, 0],
        cameraUpVector=[1, 0, 0]
        )
    projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)
    renderer = p.ER_BULLET_HARDWARE_OPENGL   # or p.ER_TINY_RENDERER

    first_iteration = True

    while True:
        camera_link_state = p.getLinkState(
            robot_body_unique_id,
            linkIndex=camera_link_idx,
            computeForwardKinematics=1,  # True
            physicsClientId=pybullet_client
        )
        camera_link_position = camera_link_state[0]      # tuple of 3 floats, same values as index [4]
        camera_link_orientation = camera_link_state[1]   # tuple of 4 floats, same values as index [5]

        camera_config["position"] = list(camera_link_position)
        camera_config["orientation"] = list(camera_link_orientation)

        if first_iteration:
            first_iteration = False
            print(f"{camera_joint_frame} info:")
            for elem in camera_link_state:
                print(f"\t{elem}")

            print("Camera config:")
            print(camera_config)


        images = p.getCameraImage(width, height, view_matrix, projection_matrix, renderer=renderer)
        depth_buffer_opengl = np.reshape(images[3], [width, height])
        depth_opengl = far * near / (far - (far - near) * depth_buffer_opengl)

        time.sleep(0.01)

        p.stepSimulation()

if __name__ == "__main__":
    main()