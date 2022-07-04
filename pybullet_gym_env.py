"""
Ref:
- https://towardsdatascience.com/creating-a-custom-openai-gym-environment-for-stock-trading-be532be3910e
- https://towardsdatascience.com/beginners-guide-to-custom-environments-in-openai-s-gym-989371673952
- https://medium.com/cloudcraftz/build-a-custom-environment-using-openai-gym-for-reinforcement-learning-56d7a5aa827b
- https://github.com/openai/gym/blob/master/gym/envs/box2d/bipedal_walker.py
"""

import gym
from gym import spaces

import numpy as np
from enum import Enum
from copy import deepcopy
from scipy.spatial.transform import Rotation as R
import time

import trimesh

from stable_baselines3.common.env_checker import check_env
import pybullet as p
import pybullet_data
import pybullet_utils.bullet_client as bc
import pybulletX as px

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

class HsrPybulletEnv(gym.Env):
    def __init__(self) -> None:
        super().__init__()

        urdf_file_path = "hsrb_description/robots/hsrb.urdf"
        use_fixed_base = True
        torque_control = True
        self.connection_mode = p.GUI
        hand = False

        self.camera_joint_frame = 'hand_camera_gazebo_frame_joint' if hand else 'head_rgbd_sensor_gazebo_frame_joint'
        self.camera_config = deepcopy(CAMERA_REALSENSE_CONFIG if hand else CAMERA_XTION_CONFIG)

        self.bullet_client = bc.BulletClient(connection_mode=self.connection_mode)
        self.px_client = px.Client(client_id=self.bullet_client._client)

        self.bullet_client.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.bullet_client.loadURDF("plane.urdf")

        self.bullet_client.setGravity(0, 0, -9.8)

        self.renderer = self.bullet_client.ER_BULLET_HARDWARE_OPENGL   # or pybullet_client.ER_TINY_RENDERER

        self.robot_body_unique_id = px.Robot(
            urdf_file_path,
            use_fixed_base=use_fixed_base,
            physics_client=self.px_client,
            base_position=[0.0, 0.0, 0.0],
            flags=self.bullet_client.URDF_USE_SELF_COLLISION
            )
        self.robot_body_unique_id.torque_control = torque_control

        self.num_joints = self.robot_body_unique_id.num_joints
        self.num_dofs = self.robot_body_unique_id.num_dofs

        self.joint_limits_lower, self.joint_limits_upper = self.get_joint_limits()

        self.action_space = spaces.Box(
            np.array(self.joint_limits_lower).astype(np.float32),
            np.array(self.joint_limits_upper).astype(np.float32),
        )
        self.observation_space = spaces.Box(
            np.array(self.joint_limits_lower).astype(np.float32),
            np.array(self.joint_limits_upper).astype(np.float32)
        )

        self.add_object_to_scene(model_name="007_tuna_fish_can")

        self.spin()
    
    def step(self, action):
        ...
    
    def reset(self):
        self.robot_body_unique_id.reset()

    def render(self, mode="human", close=False):
        ...

    def spin(self):
        while True:
            self.get_joint_values(verbose=True)
            width, height, rgb_img, depth_img, seg_img = self.get_camera_image()
            self.bullet_client.stepSimulation()
            time.sleep(0.01)

    def get_camera_image(self):
        camera_link_state = self.robot_body_unique_id.get_link_state_by_name(self.camera_joint_frame)
        camera_link_position = camera_link_state.world_link_frame_position
        camera_link_orientation = camera_link_state.world_link_frame_orientation

        self.camera_config["position"] = list(camera_link_position)
        self.camera_config["orientation"] = list(camera_link_orientation)

        self.camera_config["orientation"] = (R.from_quat(self.camera_config["orientation"]) * R.from_euler('YZ', [0.5 * np.pi, -0.5 * np.pi])).as_quat()

        width, height, rgb_img, depth_img, seg_img = self.bullet_client.getCameraImage(
            self.camera_config["image_size"][1],
            self.camera_config['image_size'][0],
            viewMatrix=self.get_view_matrix(deepcopy(self.camera_config)),
            projectionMatrix=self.get_projection_matrix(deepcopy(self.camera_config)),
            shadow=0,
            flags=self.bullet_client.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
            renderer=self.renderer
            )

        return width, height, rgb_img, depth_img, seg_img

    def get_view_matrix(self, camera_config: dict):
        # OpenGL camera settings.
        lookdir = np.float32([0, 0, 1]).reshape(3, 1)
        updir = np.float32([0, -1, 0]).reshape(3, 1)

        rotm = self.bullet_client.getMatrixFromQuaternion(camera_config['orientation'])
        rotm = np.float32(rotm).reshape(3, 3)
        lookdir = (rotm @ lookdir).reshape(-1)
        lookat = camera_config['position'] + lookdir
        updir = (rotm @ updir).reshape(-1)

        return self.bullet_client.computeViewMatrix(camera_config['position'], lookat, updir)

    def get_projection_matrix(self, camera_config: dict):
        focal_len = camera_config['intrinsics'][0]
        fovh = (camera_config['image_size'][0] / 2) / focal_len
        fovh = 180 * np.arctan(fovh) * 2 / np.pi

        # Notes: 1) FOV is vertical FOV 2) aspect must be float
        znear, zfar = camera_config['zrange']
        aspect_ratio = camera_config['image_size'][1] / camera_config['image_size'][0]

        return self.bullet_client.computeProjectionMatrixFOV(fovh, aspect_ratio, znear, zfar)

    def get_joint_limits(self, verbose=False):
        joint_infos = self.robot_body_unique_id.get_joint_infos()
        joint_lower_limits = joint_infos["joint_lower_limit"]
        joint_upper_limits = joint_infos["joint_upper_limit"]

        if verbose:
            print("Joint limits:")
            for name, lower, upper in zip(all_joint_names, joint_lower_limits, joint_upper_limits):
                print(f"\t{name}: [{lower}, {upper}]")

        return joint_lower_limits, joint_upper_limits

    def get_joint_values(self, verbose=False):
        joint_values = self.robot_body_unique_id.get_states().joint_position

        if verbose:
            print("All joint values:")
            for name, value in zip(all_joint_names, joint_values):
                print(f"\t{name}: {value}")

        return joint_values

    def add_object_to_scene(self, model_name: str):
        mesh_path = 'assets/ycb/{}/google_16k/nontextured.stl'.format(model_name)
        collision_path = 'assets/ycb/{}/google_16k/collision.obj'.format(model_name)
        mesh = trimesh.load(mesh_path, force='mesh', process=False)
        mesh.density = 150
        scale = 1

        centroid = mesh.centroid
        scale = 1
        scale = [scale, scale, scale]

        visual_shape_id = self.bullet_client.createVisualShape(
            shapeType=self.bullet_client.GEOM_MESH,
            rgbaColor=[1, 0, 0, 1],
            specularColor=[0.4, .4, 0],
            fileName=mesh_path,
            meshScale=scale,
            visualFramePosition=-centroid,
        )
        collision_shape_id = self.bullet_client.createCollisionShape(
            shapeType=self.bullet_client.GEOM_MESH,
            fileName=collision_path,
            meshScale=scale,
            collisionFramePosition=-centroid,
        )

        obj_id = self.bullet_client.createMultiBody(
            baseMass=mesh.mass,
            basePosition=(1.0, 0, 0),
            baseOrientation=(0, 0, 0, 1),
            baseCollisionShapeIndex=collision_shape_id,
            baseVisualShapeIndex=visual_shape_id,
            baseInertialFramePosition=np.array(mesh.center_mass - centroid),
        )

        self.bullet_client.changeDynamics(obj_id, -1, lateralFriction=0.25)


if __name__ == "__main__":
    env = HsrPybulletEnv()
    check_env(env)