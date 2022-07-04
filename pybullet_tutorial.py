"""
Ref:
- https://towardsdatascience.com/simulate-images-for-ml-in-pybullet-the-quick-easy-way-859035b2c9dd
"""

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

all_joint_names = [  # order is important
    'joint_x',
    'joint_y',
    'joint_rz',
    'torso_lift_joint',
    'head_pan_joint',
    'head_tilt_joint',
    'arm_lift_joint',
    'arm_flex_joint',
    'arm_roll_joint',
    'wrist_flex_joint',
    'wrist_roll_joint',
    'hand_l_proximal_joint',
    'hand_l_distal_joint',
    'hand_r_proximal_joint',
    'hand_r_distal_joint'
    ]

class JointName(Enum):
    joint_x = all_joint_names[0]
    joint_y = all_joint_names[1]
    joint_rz = all_joint_names[2]
    torso_lift_joint = all_joint_names[3]
    head_pan_joint = all_joint_names[4]
    head_tilt_joint = all_joint_names[5]
    arm_lift_joint = all_joint_names[6]
    arm_flex_joint = all_joint_names[7]
    arm_roll_joint = all_joint_names[8]
    wrist_flex_joint = all_joint_names[9]
    wrist_roll_joint = all_joint_names[10]
    hand_l_proximal_joint = all_joint_names[11]
    hand_l_distal_joint = all_joint_names[12]
    hand_r_proximal_joint = all_joint_names[13]
    hand_r_distal_joint = all_joint_names[14]


def get_joint_value(robot_body_unique_id, joint_name: JointName):
    if joint_name.value not in all_joint_names:
        raise ValueError(f"Joint name {str(joint_name)} is invalid")
    
    joint_values = robot_body_unique_id.get_states().joint_position

    return joint_values[all_joint_names.index(joint_name.value)]

def get_all_joint_values(robot_body_unique_id, verbose=False):
    joint_values = robot_body_unique_id.get_states().joint_position

    if verbose:
        print("All joint values:")
        for name, value in zip(all_joint_names, joint_values):
            print(f"\t{name}: {value}")

    return joint_values

def get_joint_limits(robot_body_unique_id, verbose=False):
    joint_infos = robot_body_unique_id.get_joint_infos()
    joint_lower_limits = joint_infos["joint_lower_limit"]
    joint_upper_limits = joint_infos["joint_upper_limit"]

    if verbose:
        print("Joint limits:")
        for name, lower, upper in zip(all_joint_names, joint_lower_limits, joint_upper_limits):
            print(f"\t{name}: [{lower}, {upper}]")

    return joint_lower_limits, joint_upper_limits


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

def get_view_matrix(pybullet_client, camera_config: dict):
    # OpenGL camera settings.
    lookdir = np.float32([0, 0, 1]).reshape(3, 1)
    updir = np.float32([0, -1, 0]).reshape(3, 1)

    rotm = pybullet_client.getMatrixFromQuaternion(camera_config['orientation'])
    rotm = np.float32(rotm).reshape(3, 3)
    lookdir = (rotm @ lookdir).reshape(-1)
    lookat = camera_config['position'] + lookdir
    updir = (rotm @ updir).reshape(-1)

    return pybullet_client.computeViewMatrix(camera_config['position'], lookat, updir)

def get_projection_matrix(pybullet_client, camera_config: dict):
    focal_len = camera_config['intrinsics'][0]
    fovh = (camera_config['image_size'][0] / 2) / focal_len
    fovh = 180 * np.arctan(fovh) * 2 / np.pi

    # Notes: 1) FOV is vertical FOV 2) aspect must be float
    znear, zfar = camera_config['zrange']
    aspect_ratio = camera_config['image_size'][1] / camera_config['image_size'][0]

    return pybullet_client.computeProjectionMatrixFOV(fovh, aspect_ratio, znear, zfar)

def main():
    urdf_file_path = "hsrb_description/robots/hsrb.urdf"
    use_fixed_base = True
    connection_mode = p.GUI

    camera_joint_frame = 'hand_camera_gazebo_frame_joint' if hand else 'head_rgbd_sensor_gazebo_frame_joint'
    camera_config = deepcopy(CAMERA_REALSENSE_CONFIG if hand else CAMERA_XTION_CONFIG)

    bullet_client = bc.BulletClient(connection_mode=connection_mode)
    px_client = px.Client(client_id=bullet_client._client)

    bullet_client.setAdditionalSearchPath(pybullet_data.getDataPath())
    bullet_client.loadURDF("plane.urdf")

    robot_body_unique_id = px.Robot(urdf_file_path, use_fixed_base=use_fixed_base, physics_client=px_client, base_position=[0.0, 0.0, 0.0], flags=p.URDF_USE_SELF_COLLISION)
    robot_body_unique_id.torque_control = True

    n_joints = robot_body_unique_id.num_joints
    n_dofs = robot_body_unique_id.num_dofs
    print(f"Number of joints: {n_joints}, number of dofs: {n_dofs}")
    joint_lower_limits, joint_upper_limits = get_joint_limits(robot_body_unique_id)

    model_name = "007_tuna_fish_can"
    mesh_path = 'assets/ycb/{}/google_16k/nontextured.stl'.format(model_name)
    collision_path = 'assets/ycb/{}/google_16k/collision.obj'.format(model_name)
    mesh = trimesh.load(mesh_path, force='mesh', process=False)
    mesh.density = 150
    scale = 1

    centroid = mesh.centroid
    scale = 1
    scale = [scale, scale, scale]

    visual_shape_id = bullet_client.createVisualShape(
        shapeType=bullet_client.GEOM_MESH,
        rgbaColor=[1, 0, 0, 1],
        specularColor=[0.4, .4, 0],
        fileName=mesh_path, meshScale=scale,
        visualFramePosition=-centroid,
    )
    collision_shape_id = bullet_client.createCollisionShape(
        shapeType=bullet_client.GEOM_MESH,
        fileName=collision_path, meshScale=scale,
        collisionFramePosition=-centroid,
    )

    obj_id = bullet_client.createMultiBody(
        baseMass=mesh.mass,
        basePosition=(1.0, 0, 0),
        baseOrientation=(0, 0, 0, 1),
        baseCollisionShapeIndex=collision_shape_id,
        baseVisualShapeIndex=visual_shape_id,
        baseInertialFramePosition=np.array(mesh.center_mass - centroid),
    )

    bullet_client.changeDynamics(obj_id, -1, lateralFriction=0.25)
    bullet_client.setGravity(0, 0, -9.8)

    renderer = bullet_client.ER_BULLET_HARDWARE_OPENGL   # or pybullet_client.ER_TINY_RENDERER

    first_iteration = True

    while True:
        camera_link_state = robot_body_unique_id.get_link_state_by_name(camera_joint_frame)
        camera_link_position = camera_link_state.world_link_frame_position
        camera_link_orientation = camera_link_state.world_link_frame_orientation

        camera_config["position"] = list(camera_link_position)
        camera_config["orientation"] = list(camera_link_orientation)

        camera_config["orientation"] = (R.from_quat(camera_config["orientation"]) * R.from_euler('YZ', [0.5 * np.pi, -0.5 * np.pi])).as_quat()

        if first_iteration:
            first_iteration = False

            print("Camera config:")
            print(camera_config)

        width, height, rgb_img, depth_img, seg_img = bullet_client.getCameraImage(
            camera_config["image_size"][1],
            camera_config['image_size'][0], 
            viewMatrix=get_view_matrix(bullet_client, deepcopy(camera_config)),
            projectionMatrix=get_projection_matrix(bullet_client, deepcopy(camera_config)),
            shadow=0,
            flags=bullet_client.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
            renderer=renderer
            )

        time.sleep(0.01)
        bullet_client.stepSimulation()

if __name__ == "__main__":
    main()