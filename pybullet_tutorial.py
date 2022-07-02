"""
Ref:
- https://towardsdatascience.com/simulate-images-for-ml-in-pybullet-the-quick-easy-way-859035b2c9dd
"""

import time
from copy import deepcopy

import numpy as np
from scipy.spatial.transform import Rotation as R
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

hand = False   # if false, CAMERA_XTION_CONFIG config will be used

def get_view_matrix(camera_config: dict):
    # OpenGL camera settings.
    lookdir = np.float32([0, 0, 1]).reshape(3, 1)
    updir = np.float32([0, -1, 0]).reshape(3, 1)

    rotm = p.getMatrixFromQuaternion(camera_config['orientation'])
    rotm = np.float32(rotm).reshape(3, 3)
    lookdir = (rotm @ lookdir).reshape(-1)
    lookat = camera_config['position'] + lookdir
    updir = (rotm @ updir).reshape(-1)

    return p.computeViewMatrix(camera_config['position'], lookat, updir)

def get_projection_matrix(camera_config: dict):
    focal_len = camera_config['intrinsics'][0]
    fovh = (camera_config['image_size'][0] / 2) / focal_len
    fovh = 180 * np.arctan(fovh) * 2 / np.pi

    # Notes: 1) FOV is vertical FOV 2) aspect must be float
    znear, zfar = camera_config['zrange']
    aspect_ratio = camera_config['image_size'][1] / camera_config['image_size'][0]

    return p.computeProjectionMatrixFOV(fovh, aspect_ratio, znear, zfar)

def main():
    urdf_file_path = "hsrb_description/robots/hsrb.urdf"
    use_fixed_base = True
    connection_mode = p.GUI

    camera_joint_frame = 'hand_camera_gazebo_frame_joint' if hand else 'head_rgbd_sensor_gazebo_frame_joint'
    camera_config = deepcopy(CAMERA_REALSENSE_CONFIG if hand else CAMERA_XTION_CONFIG)

    pybullet_client = p.connect(connection_mode)

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf")

    robot_body_unique_id = p.loadURDF(
        urdf_file_path,
        useFixedBase=use_fixed_base,
        flags=p.URDF_USE_SELF_COLLISION
        )

    n_joints = p.getNumJoints(robot_body_unique_id)
    joints = [p.getJointInfo(robot_body_unique_id, i) for i in range(n_joints)]
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

    joint_states = p.getJointStates(robot_body_unique_id, range(n_joints), pybullet_client)
    print("\nJoint values as obtained from p.getJointStates():")
    print(joint_states)
    print()

    # The magic that enables torque control
    p.setJointMotorControlArray(
        bodyIndex=robot_body_unique_id,
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

        camera_config["orientation"] = (R.from_quat(camera_config["orientation"]) * R.from_euler('YZ', [0.5 * np.pi, -0.5 * np.pi])).as_quat()

        if first_iteration:
            first_iteration = False
            print(f"{camera_joint_frame} info:")
            for elem in camera_link_state:
                print(f"\t{elem}")

            print("Camera config:")
            print(camera_config)

        width, height, rgb_img, depth_img, seg_img = p.getCameraImage(
            camera_config["image_size"][1],
            camera_config['image_size'][0], 
            viewMatrix=get_view_matrix(deepcopy(camera_config)),
            projectionMatrix=get_projection_matrix(deepcopy(camera_config)),
            shadow=0,
            flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
            renderer=renderer
            )

        time.sleep(0.01)
        p.stepSimulation()

if __name__ == "__main__":
    main()