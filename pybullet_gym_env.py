"""
Ref:
- https://towardsdatascience.com/creating-a-custom-openai-gym-environment-for-stock-trading-be532be3910e
- https://towardsdatascience.com/beginners-guide-to-custom-environments-in-openai-s-gym-989371673952
- https://medium.com/cloudcraftz/build-a-custom-environment-using-openai-gym-for-reinforcement-learning-56d7a5aa827b
- https://github.com/openai/gym/blob/master/gym/envs/box2d/bipedal_walker.py
- Creating OpenAI Gym Environments with PyBullet;
    - https://gerardmaggiolino.medium.com/creating-openai-gym-environments-with-pybullet-part-1-13895a622b24
    - https://gerardmaggiolino.medium.com/creating-openai-gym-environments-with-pybullet-part-2-a1441b9a4d8e
"""

import gym
from gym import spaces

import numpy as np
from enum import Enum
from copy import deepcopy
from scipy.spatial.transform import Rotation as R
import time
import random
import os
import sys
import threading

import trimesh

from stable_baselines3.common.env_checker import check_env
import pybullet as p
import pybullet_data
import pybullet_utils.bullet_client as bc
import pybulletX as px

np.set_printoptions(suppress=True)

GRIPPER_JOINT_VALUES = {
    "open": {
        "distal": 0.0, # -np.pi * 0.25,
        "proximal": 0.7 # 1
    },
    "close": {
        "distal": 0.0,
        "proximal": 0.0 # -0.1
    }
}

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

observation_template = {
    "joint_values": None,
    "base_velocity": None,
    "object_pose": {
        "position_xyz": None,
        "quaternion_xyzw": None
    }
}

reward_values = {
    "object_not_grasped":
    {
        "base_is_mobile": 5,
        "getting_closer_to_object": 3,
        "penalty": -2
    },
    "grasping_in_progress":   # gripper_close_enough_to_grasp_object()
    {
        "base_is_mobile": 20,
    },
    "object_grasped":
    {
        "grasp_success": 20,
        "base_is_mobile": 10,
    }
}

"""
Joint information:
  Joint 0: b'joint_x', type: 1, limits: lower -10.0, upper: 10.0, link-name: b'link_x'
  Joint 1: b'joint_y', type: 1, limits: lower -10.0, upper: 10.0, link-name: b'link_y'
  Joint 2: b'joint_rz', type: 0, limits: lower -10.0, upper: 10.0, link-name: b'link_rz'
  Joint 3: b'base_footprint_joint', type: 4, limits: lower 0.0, upper: -1.0, link-name: b'base_link'
  Joint 4: b'base_roll_joint', type: 4, limits: lower 0.0, upper: -1.0, link-name: b'base_roll_link'
  Joint 5: b'base_r_drive_wheel_joint', type: 4, limits: lower 0.0, upper: -1.0, link-name: b'base_r_drive_wheel_link'
  Joint 6: b'base_l_drive_wheel_joint', type: 4, limits: lower 0.0, upper: -1.0, link-name: b'base_l_drive_wheel_link'
  Joint 7: b'base_r_passive_wheel_x_frame_joint', type: 4, limits: lower 0.0, upper: -1.0, link-name: b'base_r_passive_wheel_x_frame'
  Joint 8: b'base_r_passive_wheel_y_frame_joint', type: 4, limits: lower 0.0, upper: -1.0, link-name: b'base_r_passive_wheel_y_frame'
  Joint 9: b'base_r_passive_wheel_z_joint', type: 4, limits: lower 0.0, upper: -1.0, link-name: b'base_r_passive_wheel_z_link'
  Joint 10: b'base_l_passive_wheel_x_frame_joint', type: 4, limits: lower 0.0, upper: -1.0, link-name: b'base_l_passive_wheel_x_frame'
  Joint 11: b'base_l_passive_wheel_y_frame_joint', type: 4, limits: lower 0.0, upper: -1.0, link-name: b'base_l_passive_wheel_y_frame'
  Joint 12: b'base_l_passive_wheel_z_joint', type: 4, limits: lower 0.0, upper: -1.0, link-name: b'base_l_passive_wheel_z_link'
  Joint 13: b'base_range_sensor_joint', type: 4, limits: lower 0.0, upper: -1.0, link-name: b'base_range_sensor_link'
  Joint 14: b'base_imu_frame_joint', type: 4, limits: lower 0.0, upper: -1.0, link-name: b'base_imu_frame'
  Joint 15: b'base_f_bumper_joint', type: 4, limits: lower 0.0, upper: 0.0, link-name: b'base_f_bumper_link'
  Joint 16: b'base_b_bumper_joint', type: 4, limits: lower 0.0, upper: 0.0, link-name: b'base_b_bumper_link'
  Joint 17: b'torso_lift_joint', type: 1, limits: lower 0.0, upper: 0.345, link-name: b'torso_lift_link'
  Joint 18: b'head_pan_joint', type: 0, limits: lower -3.84, upper: 1.75, link-name: b'head_pan_link'
  Joint 19: b'head_tilt_joint', type: 0, limits: lower -1.57, upper: 0.52, link-name: b'head_tilt_link'
  Joint 20: b'head_l_stereo_camera_joint', type: 4, limits: lower 0.0, upper: -1.0, link-name: b'head_l_stereo_camera_link'
  Joint 21: b'head_l_stereo_camera_gazebo_frame_joint', type: 4, limits: lower 0.0, upper: -1.0, link-name: b'head_l_stereo_camera_gazebo_frame'
  Joint 22: b'head_r_stereo_camera_joint', type: 4, limits: lower 0.0, upper: -1.0, link-name: b'head_r_stereo_camera_link'
  Joint 23: b'head_r_stereo_camera_gazebo_frame_joint', type: 4, limits: lower 0.0, upper: -1.0, link-name: b'head_r_stereo_camera_gazebo_frame'
  Joint 24: b'head_center_camera_frame_joint', type: 4, limits: lower 0.0, upper: -1.0, link-name: b'head_center_camera_frame'
  Joint 25: b'head_center_camera_gazebo_frame_joint', type: 4, limits: lower 0.0, upper: -1.0, link-name: b'head_center_camera_gazebo_frame'
  Joint 26: b'head_rgbd_sensor_joint', type: 4, limits: lower 0.0, upper: -1.0, link-name: b'head_rgbd_sensor_link'
  Joint 27: b'head_rgbd_sensor_gazebo_frame_joint', type: 4, limits: lower 0.0, upper: -1.0, link-name: b'head_rgbd_sensor_gazebo_frame'
  Joint 28: b'arm_lift_joint', type: 1, limits: lower 0.0, upper: 0.69, link-name: b'arm_lift_link'
  Joint 29: b'arm_flex_joint', type: 0, limits: lower -2.62, upper: 0.0, link-name: b'arm_flex_link'
  Joint 30: b'arm_roll_joint', type: 0, limits: lower -2.09, upper: 3.84, link-name: b'arm_roll_link'
  Joint 31: b'wrist_flex_joint', type: 0, limits: lower -1.92, upper: 1.22, link-name: b'wrist_flex_link'
  Joint 32: b'wrist_roll_joint', type: 0, limits: lower -1.92, upper: 3.67, link-name: b'wrist_ft_sensor_mount_link'
  Joint 33: b'wrist_ft_sensor_frame_joint', type: 4, limits: lower 0.0, upper: 0.0, link-name: b'wrist_ft_sensor_frame'
  Joint 34: b'wrist_ft_sensor_frame_inverse_joint', type: 4, limits: lower 0.0, upper: -1.0, link-name: b'wrist_roll_link'
  Joint 35: b'hand_palm_joint', type: 4, limits: lower 0.0, upper: -1.0, link-name: b'hand_palm_link'
  Joint 36: b'hand_motor_joint', type: 4, limits: lower -0.798, upper: 1.24, link-name: b'hand_motor_dummy_link'
  Joint 37: b'hand_l_proximal_joint', type: 0, limits: lower -0.798, upper: 1.24, link-name: b'hand_l_proximal_link'
  Joint 38: b'hand_l_spring_proximal_joint', type: 4, limits: lower 0.0, upper: 0.698, link-name: b'hand_l_spring_proximal_link'
  Joint 39: b'hand_l_mimic_distal_joint', type: 4, limits: lower -0.698, upper: -0.0, link-name: b'hand_l_mimic_distal_link'
  Joint 40: b'hand_l_distal_joint', type: 0, limits: lower -1.24, upper: 0.798, link-name: b'hand_l_distal_link'
  Joint 41: b'hand_l_finger_tip_frame_joint', type: 4, limits: lower 0.0, upper: -1.0, link-name: b'hand_l_finger_tip_frame'
  Joint 42: b'hand_l_finger_vacuum_frame_joint', type: 4, limits: lower 0.0, upper: -1.0, link-name: b'hand_l_finger_vacuum_frame'
  Joint 43: b'hand_r_proximal_joint', type: 0, limits: lower -0.798, upper: 1.24, link-name: b'hand_r_proximal_link'
  Joint 44: b'hand_r_spring_proximal_joint', type: 4, limits: lower 0.0, upper: 0.698, link-name: b'hand_r_spring_proximal_link'
  Joint 45: b'hand_r_mimic_distal_joint', type: 4, limits: lower -0.698, upper: -0.0, link-name: b'hand_r_mimic_distal_link'
  Joint 46: b'hand_r_distal_joint', type: 0, limits: lower -1.24, upper: 0.798, link-name: b'hand_r_distal_link'
  Joint 47: b'hand_r_finger_tip_frame_joint', type: 4, limits: lower 0.0, upper: -1.0, link-name: b'hand_r_finger_tip_frame'
  Joint 48: b'hand_camera_frame_joint', type: 4, limits: lower 0.0, upper: -1.0, link-name: b'hand_camera_frame'
  Joint 49: b'hand_camera_gazebo_frame_joint', type: 4, limits: lower 0.0, upper: -1.0, link-name: b'hand_camera_gazebo_frame'
"""
all_joint_names = [  # order is important
    'joint_x',                 # 0   (actual joint index: 0)
    'joint_y',                 # 1   (actual joint index: 1)
    'joint_rz',                # 2   (actual joint index: 2)
    'torso_lift_joint',        # 3   (actual joint index: 17)
    'head_pan_joint',          # 4   (actual joint index: 18)
    'head_tilt_joint',         # 5   (actual joint index: 19)
    'arm_lift_joint',          # 6   (actual joint index: 28)
    'arm_flex_joint',          # 7   (actual joint index: 29)
    'arm_roll_joint',          # 8   (actual joint index: 30)
    'wrist_flex_joint',        # 9   (actual joint index: 31)
    'wrist_roll_joint',        # 10  (actual joint index: 32)
    'hand_l_proximal_joint',   # 11  (actual joint index: 37)
    'hand_l_distal_joint',     # 12  (actual joint index: 40)
    'hand_r_proximal_joint',   # 13  (actual joint index: 43)
    'hand_r_distal_joint'      # 14  (actual joint index: 46)
    ]


def file_exists(filepath: str) -> bool:
    return os.path.exists(filepath)


class Color(Enum):
    Red = "\033[0;31m"   
    Green = "\033[0;32m"
    Yellow = "\033[0;33m"
    Blue = "\033[0;34m"
    Purple = "\033[0;35m"
    Cyan = "\033[0;36m"
    White = "\033[0;37m"

    Color_Off = "\033[0m"
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

        self.urdf_file_path = "hsrb_description/robots/hsrb.urdf"
        if not file_exists(self.urdf_file_path):
            print(f"{Color.Red.value}ERROR! Urdf filepath does not exist: {self.urdf_file_path}{Color.Color_Off.value}")

        self.object_model_name = "002_master_chef_can"
        torque_control = True
        gui = True
        hand = False

        self.done = False
        self.episode_start_time = None
        self.episode_max_duration = 60   # seconds

        self.previous_proximity_to_object = float('inf')
        self.proximity_change_threshold = 1.0  # meter

        # https://stable-baselines.readthedocs.io/en/master/guide/rl_tips.html#tips-and-tricks-when-creating-a-custom-environment
        self.normalized_action_space = True
        self.normalized_observation_space = True

        self.connection_mode = p.GUI if gui else p.DIRECT
        print(f"{Color.Green.value}GUI mode is %s{Color.Color_Off.value}" % ("enabled" if gui else "disabled"))

        self.gravity_enabled = True
        self.use_fixed_base = True

        self.error_tolerance = 1.0
        self.is_base_moving_threshold = 1.0
        self.max_time_to_reach_goal = 20  # seconds allowed to reach goal

        self.camera_joint_frame = 'hand_camera_gazebo_frame_joint' if hand else 'head_rgbd_sensor_gazebo_frame_joint'
        self.camera_config = deepcopy(CAMERA_REALSENSE_CONFIG if hand else CAMERA_XTION_CONFIG)
        
        self.end_effector_joint_frame = 'hand_palm_joint'
        self.end_effector_link_idx = 35  # hand_palm_joint, hand_palm_link

        self.bullet_client = bc.BulletClient(connection_mode=self.connection_mode)
        self.px_client = px.Client(client_id=self.bullet_client._client)

        self.bullet_client.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.bullet_client.loadURDF("plane.urdf")

        if self.gravity_enabled:
            self.bullet_client.setGravity(0, 0, -9.8)

        self.renderer = self.bullet_client.ER_BULLET_HARDWARE_OPENGL   # or pybullet_client.ER_TINY_RENDERER

        self.robot_body_unique_id = px.Robot(
            self.urdf_file_path,
            use_fixed_base=self.use_fixed_base,
            physics_client=self.px_client,
            base_position=[0.0, 0.0, 0.0],
            flags=self.bullet_client.URDF_USE_SELF_COLLISION
            )
        print()
        self.robot_body_unique_id.torque_control = torque_control

        self.num_joints = self.robot_body_unique_id.num_joints   # 50
        self.num_dofs = self.robot_body_unique_id.num_dofs       # 15

        self.free_joint_indices = self.robot_body_unique_id.free_joint_indices
        self.joint_limits_lower, self.joint_limits_upper = self.get_joint_limits()
        self.joint_max_velocities, self.joint_max_forces = self.get_joint_max_velocities_and_forces()

        self.observation_space_length = 24  # 15 joint values, 2 base velocities, 3 object position, 4 object orientation (quaternion)
        self.observation_space = self.construct_observation_space()
        self.action_space = self.construct_action_space()

        self.added_obj_id = self.spawn_object_at_random_location(model_name=self.object_model_name)

        self.camera_thread = threading.Thread(target=self.spin)
        self.camera_thread.start()

    def close(self):
        self.camera_thread.join()
        self.px_client.release()

    def spawn_object_at_random_location(self, model_name: str, verbose=False):
        min_xy = 2.0
        max_xy = 7.0   # robot xy location is limited to (±10, ±10)

        # NOTE: if min_xy or max_xy changes, also update construct_observation_space()

        base_position = [random.uniform(min_xy, max_xy) for i in range(2)]
        base_position.append(0.0)

        base_position = tuple(base_position)

        return self.add_object_to_scene(model_name, base_position, verbose)

    def step(self, action: list, verbose=False):
        """
        action: expected to be normalized if action space is normalized
        """
        info = {}

        # before applying action, scale it back up if action space is normalized
        if self.normalized_action_space:
            if verbose:
                print("Scaling normalized joint value from [-1, 1] to joint limits:")
            for i in range(self.num_dofs):   # TODO: vectorize solution from https://stackoverflow.com/a/36000844/6010333
                normalized_value = deepcopy(action[i])
                action[i] = np.interp(action[i], (-1.0, 1.0), (self.joint_limits_lower[i], self.joint_limits_upper[i]))
                if verbose:
                    print(f"\t{all_joint_names[i]}: {normalized_value:.4f} --> {action[i]:.4f}")
        self.set_joint_position(action, verbose=verbose)

        obs = self.get_observation(verbose=verbose, numpify=True)
        reward = self.calculate_reward()

        if not self.observation_space.contains(obs):
            print(f"{Color.Yellow.value}WARNING: observation does not fall within observation space{Color.Color_Off.value}")
            print("\tApplied action is %s" % "normalized" if self.normalized_action_space else "not normalized")
            print(f"\tAction: {action}")
            print("\tLimit check:")
            for i in range(self.observation_space_length):
                color = Color.Color_Off.value if self.observation_space.low[i] <= obs[i] <= self.observation_space.high[i] else Color.Red.value
                print(f"\t\t{color}{obs[i]:.5f}: lower limit {self.observation_space.low[i]}, upper limit {self.observation_space.high[i]}{Color.Color_Off.value}")

        self.done = False # self.terminate_episode()
        
        return obs, reward, self.done, info

    def terminate_episode(self) -> bool:
        """
        Episode termination criteria;
            - episode duration has exceeded max episode duration
            - robot went outside square area of ±10m
            - robot is moving away from target without grabbing object
            - robot link that that does not have "hand" or "wrist" in its name has collided with object (will likely send the object flying away)
        """
        if self.done:
            print(f"{Color.Red.value}terminate_episode(): no episode is active{Color.Color_Off.value}")
            return False
        
        # episode duration has exceeded max episode duration
        if (time.time() - self.episode_start_time) >= self.episode_max_duration:
            print(f"{Color.Red.value}terminate_episode(): max episode duration {self.episode_max_duration}s exceeded{Color.Color_Off.value}")
            return True
        
        # robot went outside square area of ±10m
        current_base_position_xy = self.get_base_position_xy()  # [x, y]
        if (current_base_position_xy[0] > self.joint_limits_upper[0]) or (current_base_position_xy[1] > self.joint_limits_upper[1]):
            print(f"{Color.Red.value}terminate_episode(): robot went outside square of ±10m{Color.Color_Off.value}")
            return True
        
        # robot is moving away from target without grabbing object
        current_proximity_to_object = self.calculate_base_proximity_to_object()
        if not self.object_grasped():
            if current_proximity_to_object - self.previous_proximity_to_object > self.proximity_change_threshold:
                print(f"""{Color.Red.value}terminate_episode(): proximity is not decreasing, 
                currently {current_proximity_to_object}m from object, 
                previously {self.previous_proximity_to_object}m from object, 
                threshold {self.proximity_change_threshold}m{Color.Color_Off.value}""")
                return True
        self.previous_proximity_to_object = deepcopy(current_proximity_to_object)

        # robot link that is not a "hand" or "wrist" link has collided with object
        if self.collided_with_object(exclude_hand_and_wrist=True):
            print(f"{Color.Red.value}terminate_episode(): collision with object{Color.Color_Off.value}")
            return True

        return False

    def start_episode(self):
        print(f"{Color.Blue.value}Starting episode..{Color.Color_Off.value}")
        self.episode_start_time = time.time()

        self.done = False
        self.previous_proximity_to_object = float('inf')
    
    def get_observation(self, verbose=False, numpify=True):
        """
        Returns information about environment as a dict based on observation template

        numpify: if True, then converts observation_template dict to numpy array
                 (default value should be True else stable_baselines3.common.env_checker.check_env() fails)
        """
        obs = deepcopy(observation_template)
        obs["joint_values"] = self.get_joint_values()
        obs["base_velocity"] = self.get_base_velocity(xy_components=True)
        obj_position_xyz, obj_quaternion_xyzw = self.get_object_pose()
        obs["object_pose"] = {"position_xyz": obj_position_xyz, "quaternion_xyzw": obj_quaternion_xyzw}

        # if observation space is normalized, joint values have to normalized to [-1, 1]
        if self.normalized_observation_space:
            if verbose:
                print("Normalizing joint values to range [-1, 1]:")
            for i in range(self.num_dofs):   # TODO: vectorize solution from https://stackoverflow.com/a/36000844/6010333
                joint_value = deepcopy(obs["joint_values"][i])
                obs["joint_values"][i] = np.interp(obs["joint_values"][i], (self.joint_limits_lower[i], self.joint_limits_upper[i]), (-1.0, 1.0))
                if verbose:
                    print(f"\t{all_joint_names[i]}: {joint_value:.4f} --> %.4f" % obs["joint_values"][i])

        if verbose:
            print("OBSERVATION:")
            self.print_joint_values(obs["joint_values"], title="Joint values:", tab_indent=1)
            print("\tBase velocity: X = %.5f, Y = %.5f, L2 norm = %.5f" % (obs["base_velocity"][0], obs["base_velocity"][1], np.linalg.norm(obs["base_velocity"])))
            self.print_pose(obs["object_pose"]["position_xyz"], obs["object_pose"]["quaternion_xyzw"], title="Object pose:", tab_indent=1)

        return self.convert_observation_dict_to_numpy(obs, verbose) if numpify else obs

    def convert_observation_dict_to_numpy(self, obs: dict, verbose=False) -> np.array:
        """
        Converts observation dictionary (see observation_template) into a 1D numpy array.
        If modifying this method, also update convert_observation_numpy_to_dict()
        """

        l = []
        l.extend(obs["joint_values"])
        l.extend(obs["base_velocity"])
        l.extend(obs["object_pose"]["position_xyz"])
        l.extend(obs["object_pose"]["quaternion_xyzw"])

        obs_numpified = np.array(l, dtype=np.float32)

        if verbose:
            print("Conversion from observation dictionary to numpy array:")
            print("\tDictionary:")
            self.print_joint_values(obs["joint_values"], title="Joint values:", tab_indent=2)
            print("\t\tBase velocity: X = %.5f, Y = %.5f" % (obs["base_velocity"][0], obs["base_velocity"][1]))
            self.print_pose(obs["object_pose"]["position_xyz"], obs["object_pose"]["quaternion_xyzw"], title="Object pose:", tab_indent=2)
            print(f"\tNumpy array (shape: {obs_numpified.shape}, type: {type(obs_numpified)}, type of every element: {type(obs_numpified[0])}):\n{obs_numpified}")

        return obs_numpified

    def convert_observation_numpy_to_dict(self, obs: np.array, verbose=False) -> dict:
        """
        Converts observation numpy array to a dictionary (see observation_template).
        The parsing in this function is VERY STRONGLY tied to convert_observation_dict_to_numpy()
        """
        assert obs.shape[0] == self.observation_space_length, f"convert_observation_numpy_to_dict() expects an array of shape ({self.observation_space_length},)"

        d = deepcopy(observation_template)
        d["joint_values"] = obs[:15]   # 15 comes from number of joint values and is equal to self.num_dofs
        d["base_velocity"] = obs[15:17]
        d["object_pose"] = {"position_xyz": obs[17:20], "quaternion_xyzw": obs[20:]}

        if verbose:
            print("Conversion from observation numpy array to dictionary:")
            print(f"\tNumpy array (shape: {obs.shape}):\n{obs}")
            print("\tDictionary:")
            self.print_joint_values(d["joint_values"], title="Joint values:", tab_indent=2)
            print("\t\tBase velocity: X = %.5f, Y = %.5f" % (d["base_velocity"][0], d["base_velocity"][1]))
            self.print_pose(d["object_pose"]["position_xyz"], d["object_pose"]["quaternion_xyzw"], title="Object pose:", tab_indent=2)

        return d

    def construct_action_space(self, verbose=False):
        """
        Action space will have a total of 15 elements (same as the number of degrees of freedom i.e. self.num_dofs)

        If normalized_action_space is True, joint values are normalized to [-1, 1]
        """
        if self.normalized_action_space:
            action_space = spaces.Box(
                np.ones(self.num_dofs, dtype=np.float32) * -1.0,
                np.ones(self.num_dofs, dtype=np.float32)
            )
        else:
            action_space = spaces.Box(
                np.array(self.joint_limits_lower).astype(np.float32),
                np.array(self.joint_limits_upper).astype(np.float32),
            )

        if verbose:
            print(f"{Color.Cyan.value}Action space{Color.Color_Off.value} (type: {type(action_space)})\n{action_space}")

        return action_space

    def construct_observation_space(self, verbose=False):
        """
        Observation space will have a total of 24 elements (see observation_template):
        - 15 joint values
        - 2 base velocity X and Y components
        - 3 object position XYZ
        - 4 object orientation (quaternion XYZW)

        If normalized_observation_space is True, joint values are normalized to [-1, 1] (rest are not normalized) 
        """
        lower = []
        upper = []

        # joint values (normalized between [-1.0, 1.0])
        if self.normalized_observation_space:
            lower.extend(np.ones(self.num_dofs, dtype=np.float32) * -1.0)
            upper.extend(np.ones(self.num_dofs, dtype=np.float32))
        else:
            lower.extend(self.joint_limits_lower)
            upper.extend(self.joint_limits_upper)

        # Base velocity cannot be normalized because we don't know how high base velocity can go.
        # L2-norm velocity was observed to go up to 150:
        #     X component was found to fall in the range [-100, 100]
        #     Y component was found to fall in the range [-100, 100]
        #     so setting limits of X and Y components to [-120, 120] ¯\_(ツ)_/¯
        lower.extend(np.ones(2, dtype=np.float32) * -120.0)
        upper.extend(np.ones(2, dtype=np.float32) * 120.0)

        # robot xy location is limited to (±10, ±10) and object is spawned in bounds [2.0, 7.0] (see spawn_object_at_random_location())
        lower.extend(np.zeros(3, dtype=np.float32))
        upper.extend(np.ones(3, dtype=np.float32) * 10.0)   # this assumes object's Z position will always be within [0, 10] meters

        # robot orientation should already be a normalized quaternion (xyzw)
        lower.extend(np.ones(4, dtype=np.float32) * -1.0)
        upper.extend(np.ones(4, dtype=np.float32))

        early_exit = False
        if len(lower) != self.observation_space_length:
            print(f"{Color.Red.value}construct_observation_space(): length of 'lower' is {len(lower)} but should be equal to {self.observation_space_length}{Color.Color_Off.value}")
            early_exit = True
        if len(upper) != self.observation_space_length:
            print(f"{Color.Red.value}construct_observation_space(): length of 'upper' is {len(lower)} but should be equal to {self.observation_space_length}{Color.Color_Off.value}")
            early_exit = True
        if early_exit:
            sys.exit()

        observation_space = spaces.Box(
            np.array(lower).astype(np.float32),
            np.array(upper).astype(np.float32),
            dtype=np.float32
        )

        if verbose:
            print(f"{Color.Cyan.value}Observation space{Color.Color_Off.value} (type: {type(observation_space)})\n{observation_space}")

        return observation_space

    def calculate_reward(self):
        return 0.0

    def object_grasped(self) -> bool:
        print(f"{Color.Yellow.value}object_grasped() was called but it's not implemented{Color.Color_Off.value}")
        return False

    def calculate_base_proximity_to_object(self, verbose=False):
        robot_base_position = self.get_base_position_xy()  # [x, y]
        object_position_xyz, _ = self.get_object_pose()    # (x, y, z)
        object_position_xyz = list(object_position_xyz[:2])

        proximity = np.linalg.norm(np.array(robot_base_position) - np.array(object_position_xyz))

        if verbose:
            print("calculate_base_proximity_to_object():")
            print(f"\trobot base position: {robot_base_position}")
            print(f"\tobject position: {object_position_xyz}")
            print(f"\tproximity: {proximity}m")

        return proximity

    def base_is_moving(self) -> bool:
        return self.get_base_velocity(xy_components=False) > self.is_base_moving_threshold

    def gripper_close_enough_to_grasp_object(self) -> bool:
        ...

    def sample_action_space(self):
        return self.action_space.sample()
    
    def set_joint_position(self, q, verbose=False):
        if len(q) != self.num_dofs:
            raise ValueError(f"set_joint_positions(): q has {len(q)} values but robot has {self.num_dofs} DOF")

        current_base_position_xy = self.get_base_position_xy()
        line_from_xyz = [current_base_position_xy[0], current_base_position_xy[1], 0.0]
        line_to_xyz = [q[0], q[1], 0.0]

        line_id = self.bullet_client.addUserDebugLine(line_from_xyz, line_to_xyz, lineColorRGB=[1, 0, 0], lineWidth=10.0)
        
        self.bullet_client.setJointMotorControlArray(
            bodyUniqueId=self.robot_body_unique_id.id,
            jointIndices=self.free_joint_indices,
            controlMode=p.POSITION_CONTROL,
            targetPositions=q,
            targetVelocities=np.zeros_like(q)
        )

        # sluggish (code from Yuki's version)
        # current_joint_values = self.get_joint_values()
        # time_for_base = np.max([abs(q[i] - current_joint_values[i]) / self.joint_max_velocities[i] for i in [0, 1, 2]])
        # max_velocities = deepcopy(self.joint_max_velocities)
        # for i in range(3):   # first 3 joints that correspond to base XY position and yaw orientation
        #     max_velocities[i] = abs(q[i] - current_joint_values[i]) / time_for_base
        # for i in range(len(all_joint_names)):
        #     target_velocity = max_velocities[i] if (time_for_base >  0 and i <= 2) else 0.0
        #     self.bullet_client.setJointMotorControl2(
        #         bodyUniqueId=self.robot_body_unique_id.id,
        #         jointIndex=self.free_joint_indices[i],
        #         controlMode=p.POSITION_CONTROL,
        #         targetPosition=q[i],
        #         targetVelocity=target_velocity,
        #         force=self.joint_max_forces[i],
        #         maxVelocity=max_velocities[i]
        #     )

        # pybulletX's robot.set_joint_position() causes base to overshoot, not sure why
        # self.robot_body_unique_id.set_joint_position(
        #     q,
        #     max_forces=self.joint_max_forces,
        #     use_joint_effort_limits=True
        # )

        start_time = time.time()
        error = self.get_error(q, self.get_joint_values())
        base_velocity = self.get_base_velocity(xy_components=False)
        max_base_velocity = 0
        max_base_velocity_xy = [0, 0]
        robot_has_started_moving = False
        base_log_start, base_log_freq = start_time, 10.0
        while True:
            if not robot_has_started_moving and self.base_is_moving():
                robot_has_started_moving = True

            if base_velocity > max_base_velocity:
                max_base_velocity = base_velocity
                max_base_velocity_xy = self.get_base_velocity(xy_components=True)

            if time.time() - start_time > self.max_time_to_reach_goal:
                if verbose:
                    print(f"{Color.Yellow.value}TIMED OUT on its way to goal{Color.Color_Off.value}, L2 norm of error across all joints: {error:.4f}, base_velocity: {base_velocity:.4f}")
                break
            
            if error < self.error_tolerance and robot_has_started_moving and base_velocity <= 0.001:
                if verbose:
                    print(f"{Color.Green.value}REACHED GOAL in {time.time() - start_time:.4f} seconds{Color.Color_Off.value}, all joints error: {error:.4f}, base error: {self.get_base_error(q, self.get_joint_values()):.4f}, base velocity: {base_velocity:.4f}, max base velocity: {max_base_velocity:.4f} (components: {max_base_velocity_xy})")
                    self.print_joint_values(self.get_joint_values(), title="Joint values at goal:", tab_indent=2)
                break
            
            if verbose and time.time() - base_log_start > base_log_freq:
                base_log_start = time.time()
                print("Currently:")
                print(f"\tBase pose: {self.get_base_position_xy()}, velocity: {self.get_base_velocity()}, base error: {self.get_base_error(q, self.get_joint_values()):.4f}, all joint error: {error}")
                current_joint_values = self.get_joint_values()
                print(f"\tJoint values:")
                for i in range(len(all_joint_names)):
                    print(f"\t\t{all_joint_names[i]}: {current_joint_values[i]:.4f} (target: {q[i]:.4f}, error: {q[i] - current_joint_values[i]:.4f})")

            time.sleep(0.01)
            # self.bullet_client.stepSimulation()   # already running in a parallel thread that's executing self.spin()
            error = self.get_error(q, self.get_joint_values())
            base_velocity = self.get_base_velocity()

        self.bullet_client.removeUserDebugItem(line_id)

    def reset(self, verbose=False):
        self.remove_object_from_scene(self.added_obj_id)

        if verbose:
            print("================== RESET ========================")
        self.robot_body_unique_id.reset()
        self.added_obj_id = self.spawn_object_at_random_location(model_name=self.object_model_name, verbose=verbose)

        return self.get_observation(verbose=verbose, numpify=True)

    def collided_with_object(self, exclude_hand_and_wrist: bool, verbose=False) -> bool:
        """
        Contacts are NOT computed in real-time when this method is called.
        This method works with contact points computed during the most recent call to to stepSimulation() or performCollisionDetection()

        exclude_hand_and_wrist: if True, collision with robot link names that have "wrist" or "hand" in them will be ignored
        verbose: if True, names of robot links that collide and corresponding penetration depths will be logged
        """
        contacts = self.bullet_client.getContactPoints(
            bodyA=self.robot_body_unique_id.id,
            bodyB=self.added_obj_id,
            physicsClientId=self.bullet_client._client
        )

        collided = False
        collisions = {}
        if len(contacts) > 0:
            for i in range(len(contacts)):
                # each contact is a tuple of length 14
                contactFlag = contacts[i][0]         # reserved (always found to be 0)
                bodyUniqueIdA = contacts[i][1]       # body unique id of body A (always found to be 1)
                bodyUniqueIdB = contacts[i][2]       # body unique id of body B (always found to be 2)
                linkIndexA = contacts[i][3]          # link index of body A, -1 for base
                linkIndexB = contacts[i][4]          # link index of body B, -1 for base (always found to be -1, which represents "base")
                positionOnA = contacts[i][5]         # contact position on A, in Cartesian world coordinates
                positionOnB = contacts[i][6]         # contact position on B, in Cartesian world coordinates
                contactNormalOnB = contacts[i][7]    # contact normal on B, pointing towards A
                contactDistance = contacts[i][8]     # contact distance, positive for separation, negative for penetration
                normalForce = contacts[i][9]         # normal force applied during the last 'stepSimulation'
                # 10-13 are lateral friction

                if contactDistance < 0:
                    robot_link_name = self.get_link_name_from_index(linkIndexA)

                    if exclude_hand_and_wrist and ("hand" in robot_link_name or "wrist" in robot_link_name):
                        continue

                    collided = True
                    if verbose:
                        if (robot_link_name in collisions.keys() and contactDistance > collisions[robot_link_name]) or (robot_link_name not in collisions.keys()):
                            collisions[robot_link_name] = deepcopy(contactDistance)
                    else:
                        break   # if not verbose, no point iterating over all contacts

        if verbose and collided:
            print(f"{Color.Red.value}collided_with_object():{Color.Color_Off.value}")
            for link_name in collisions.keys():
                print(f"\t{link_name}: {collisions[link_name]:.6f}")

        return collided

    def get_link_name_from_index(self, link_index: int) -> str:
        """
        Returns link name for an input link_index (should be >= -1)
        """
        if link_index == -1:
            return "base"  # see documentation for getContactPoints()
        else:
            joint_info = self.bullet_client.getJointInfo(
                bodyUniqueId=self.robot_body_unique_id.id,
                jointIndex=link_index,  # in pybullet documentation, link index == joint index
                physicsClientId=self.bullet_client._client
            )
            return joint_info.link_name.decode("utf-8")

    def render(self, mode="human", close=False):
        ...

    def get_error(self, q1, q2, ignore_hand_joints=True):
        """
        These last 4 joints which are on the gripper never quite seem to reach target positions:
        - hand_l_proximal_joint
        - hand_l_distal_joint
        - hand_r_proximal_joint
        - hand_r_distal_joint
        (run "roslaunch hsrb_description hsrb_display.launch" to inspect)

        ignore_hand_joints: if True, the hand joints are ignored in the error calculation
        """

        if ignore_hand_joints:
            return np.linalg.norm(q1[:-4] - q2[:-4])   # order of q is the same as all_joint_names
        else:
            return np.linalg.norm(q1 - q2)   # order of q is the same as all_joint_names
    
    def get_base_error(self, q1, q2):
        return np.linalg.norm(q1[:3] - q2[:3])   # order of q is the same as all_joint_names
    
    def spin(self):
        while True:
            width, height, rgb_img, depth_img, seg_img = self.get_camera_image()
            self.bullet_client.stepSimulation()
            time.sleep(0.001)

    def print_joint_values(self, q, title=None, tab_indent=0):
        if len(q) != self.num_dofs:
            raise ValueError(f"print_joint_values(): q has {len(q)} values but robot has {self.num_dofs} DOF")
        
        if title is None:
            title = "All joint values:"

        tab_indent_str = ""
        for _ in range(tab_indent):
            tab_indent_str += "\t"        
        
        title = tab_indent_str + title
        print(title)
        tab_indent_str += "\t"
        for name, value in zip(all_joint_names, q):
            print(f"{tab_indent_str}{name}: {value:.4f}")
    
    def print_pose(self, position_xyz, quaternion_xyzw, title=None, tab_indent=0):
        if len(position_xyz) != 3:
            raise ValueError(f"print_pose(): position_xyz has {len(position_xyz)} elements but should have 3 elements")
        if len(quaternion_xyzw) != 4:
            raise ValueError(f"print_pose(): quaternion_xyzw has {len(quaternion_xyzw)} elements but should have 4 elements")
        
        if title is None:
            title = "Position and orientation:"

        tab_indent_str = ""
        for _ in range(tab_indent):
            tab_indent_str += "\t"

        title = tab_indent_str + title
        print(title)
        tab_indent_str += "\t"
        print(f"{tab_indent_str}Position:")
        for coor in position_xyz:
            print(f"{tab_indent_str}\t{coor:.4f}")
        print(f"{tab_indent_str}Quaternion (xyzw):")
        for coor in quaternion_xyzw:
            print(f"{tab_indent_str}\t{coor:.4f}")
    
    def get_joint_frame_pose(self, joint_frame: str):
        link_state = self.robot_body_unique_id.get_link_state_by_name(joint_frame)
        return link_state.world_link_frame_position, link_state.world_link_frame_orientation
    
    def get_camera_image(self):
        camera_link_position, camera_link_orientation = self.get_joint_frame_pose(self.camera_joint_frame)

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
            self.print_joint_values(joint_values, title="All joint values:")

        return joint_values
    
    def get_joint_velocities(self, verbose=False):
        return list(self.robot_body_unique_id.get_states().joint_velocity)
    
    def get_joint_max_velocities_and_forces(self, verbose=False):
        joint_infos = self.robot_body_unique_id.get_joint_infos()
        return joint_infos["joint_max_velocity"], joint_infos["joint_max_force"]
    
    def get_base_position_xy(self):
        return self.get_joint_values()[:2]   # first two elements are x and y
    
    def get_base_velocity(self, xy_components=False):
        """
        Returns X and Y components of base velocity if xy_components is true, else L2-norm of X and Y components
        """
        base_vel = self.get_joint_velocities()[:2]  # X and Y components
        return base_vel if xy_components else np.linalg.norm(base_vel)

    def calculate_inverse_kinematics(self, position_xyz: tuple, quaternion_xyzw: tuple, verbose=False):
        if verbose:
            self.print_pose(position_xyz, quaternion_xyzw, title="Calculating IK for")

        q = self.bullet_client.calculateInverseKinematics(
            bodyUniqueId=self.robot_body_unique_id.id,
            endEffectorLinkIndex=self.end_effector_link_idx,
            targetPosition=position_xyz,
            targetOrientation=quaternion_xyzw,
            lowerLimits=self.joint_limits_lower,
            upperLimits=self.joint_limits_upper,
            restPoses=self.get_joint_values(),
            maxNumIterations=1000,
            residualThreshold=1e-4,
            physicsClientId=self.bullet_client._client
        )

        if verbose:
            self.print_joint_values(q, title="IK solution:", tab_indent=1)

        return list(q)
        
    def calculate_forward_kinematics(self, q, verbose=False):
        """
        Ref: https://github.com/bulletphysics/bullet3/issues/2603
                "create a second DIRECT pybullet connection, load the arm there, reset it to the angles,
                and run the forward kinematics in that second 'dummy' robot."
        
        For creating a second connection, refer to https://github.com/bulletphysics/bullet3/issues/1925#issuecomment-428355937
        """

        if len(q) != self.num_dofs:
            raise ValueError(f"calculate_forward_kinematics(): q has {len(q)} values but robot has {self.num_dofs} DOF")
        
        if verbose:
            self.print_joint_values(q, title="Calculating FK for:")
        
        throwaway_client = bc.BulletClient(connection_mode=p.DIRECT)

        throwaway_client.setAdditionalSearchPath(pybullet_data.getDataPath())
        throwaway_client.loadURDF("plane.urdf")

        if self.gravity_enabled:
            throwaway_client.setGravity(0, 0, -9.8)

        throwaway_robot_body_unique_id = throwaway_client.loadURDF(
            self.urdf_file_path,
            useFixedBase=self.use_fixed_base,
            flags=p.URDF_USE_SELF_COLLISION
        )

        throwaway_client.setJointMotorControlArray(
            bodyIndex=throwaway_robot_body_unique_id,
            jointIndices=self.free_joint_indices,
            controlMode=p.POSITION_CONTROL,
            forces=np.zeros(self.num_dofs),
        )

        gripper_state = throwaway_client.getLinkState(
            bodyUniqueId=throwaway_robot_body_unique_id,
            linkIndex=self.end_effector_link_idx,
            computeForwardKinematics=1
            )
        
        throwaway_client.disconnect()
        
        if verbose:
            self.print_pose(gripper_state.world_link_frame_position, gripper_state.world_link_frame_orientation, title="FK results:")

        return gripper_state.world_link_frame_position, gripper_state.world_link_frame_orientation
    
    def gripper_open(self):
        q = self.get_joint_values()

        q[-1] = GRIPPER_JOINT_VALUES["open"]["distal"]        # hand_r_distal_joint
        q[-2] = GRIPPER_JOINT_VALUES["open"]["proximal"]      # hand_r_proximal_joint
        q[-3] = deepcopy(q[-1])                               # hand_l_distal_joint
        q[-4] = deepcopy(q[-2])                               # hand_l_proximal_joint

        self.set_joint_position(q)

    def gripper_close(self):
        q = self.get_joint_values()

        q[-1] = GRIPPER_JOINT_VALUES["close"]["distal"]        # hand_r_distal_joint
        q[-2] = GRIPPER_JOINT_VALUES["close"]["proximal"]      # hand_r_proximal_joint
        q[-3] = deepcopy(q[-1])                                # hand_l_distal_joint
        q[-4] = deepcopy(q[-2])                                # hand_l_proximal_joint

        self.set_joint_position(q)

    def get_object_pose(self):
        position_xyz, quaternion_xyzw = self.bullet_client.getBasePositionAndOrientation(
            bodyUniqueId=self.added_obj_id,
            physicsClientId=self.bullet_client._client
            )
        return position_xyz, quaternion_xyzw
    
    def add_object_to_scene(self, model_name: str, base_position: tuple, verbose=False):
        mesh_path = 'assets/ycb/{}/google_16k/nontextured.stl'.format(model_name)
        collision_path = 'assets/ycb/{}/google_16k/collision.obj'.format(model_name)
        file_not_found = False
        if not file_exists(mesh_path):
            print(f"{Color.Red.value}ERROR! Mesh filepath for {model_name} does not exist: {mesh_path}{Color.Color_Off.value}")
            file_not_found = True
        if not file_exists(collision_path):
            print(f"{Color.Red.value}ERROR! Collision filepath for {model_name} does not exist: {collision_path}{Color.Color_Off.value}")
            file_not_found = True
        if file_not_found:
            sys.exit()

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
            basePosition=base_position,
            baseOrientation=(0, 0, 0, 1),
            baseCollisionShapeIndex=collision_shape_id,
            baseVisualShapeIndex=visual_shape_id,
            baseInertialFramePosition=np.array(mesh.center_mass - centroid),
        )

        self.bullet_client.changeDynamics(obj_id, -1, lateralFriction=0.25)

        if verbose:
            self.print_pose(list(base_position), [0.0, 0.0, 0.0, 1.0], title="Object \"" + model_name + "\" pose")

        return obj_id

    def remove_object_from_scene(self, object_id):
        """
        https://github.com/bulletphysics/bullet3/issues/1389
        """
        self.bullet_client.removeBody(object_id)


if __name__ == "__main__":
    env = HsrPybulletEnv()
    # check_env(env)

    # "DRY RUN"
    # obs = env.reset()
    # home_joint_values = obs[:15]
    # n_steps = 10
    # for i in range(n_steps):
    #     print(f"Random action {i+1}/{n_steps}")
    #     # Random action
    #     action = env.action_space.sample()
    #     action[2:] = home_joint_values[2:]
    #     obs, reward, done, info = env.step(action)
    #     if done:
    #         obs = env.reset()
