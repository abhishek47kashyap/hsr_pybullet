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

observation_template = {
    "joint_values": None,
    "base_velocity_normalized": None,
    "object_pose": None
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
        torque_control = True
        self.connection_mode = p.GUI
        hand = False

        self.gravity_enabled = True
        self.use_fixed_base = True

        self.error_tolerance = 1.0
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
        self.robot_body_unique_id.torque_control = torque_control

        self.num_joints = self.robot_body_unique_id.num_joints
        self.num_dofs = self.robot_body_unique_id.num_dofs

        self.free_joint_indices = self.robot_body_unique_id.free_joint_indices
        self.joint_limits_lower, self.joint_limits_upper = self.get_joint_limits()
        self.joint_max_velocities, self.joint_max_forces = self.get_joint_max_velocities_and_forces()

        self.action_space = spaces.Box(
            np.array(self.joint_limits_lower).astype(np.float32),
            np.array(self.joint_limits_upper).astype(np.float32),
        )
        self.observation_space = spaces.Box(
            np.array(self.joint_limits_lower).astype(np.float32),
            np.array(self.joint_limits_upper).astype(np.float32)
        )

        self.added_obj_id = self.add_object_to_scene(model_name="002_master_chef_can", base_position=(2.0, 2.0, 0))
        position_xyz, quaternion_xyzw = self.get_object_pose()
        self.print_pose(position_xyz, quaternion_xyzw, title="Object position:")

        # print(f"Checking for existence of addUserDebugLine: {self.bullet_client.addUserDebugLine}")
        # random_action = self.get_random_joint_config()
        # random_action = [-0.30553615, -3.0663216, -8.185578, 0.07727123, -0.56605875, -0.44894674, 0.06767046, -1.6455808, 0.16513418, -0.85248154, -1.7382416, 0.15633392, -0.7695309, 0.39886168, -0.03402488]
        # random_action = [-5.0677767, -9.812688,  -3.4465344, 0.34411412, -1.1459529, -0.9748333, 0.23610148, -2.1307828, -0.70618105, -0.8699218, 1.338886, -0.77597266, -0.3975988, -0.58280176, 0.685311]
        # random_action = self.get_joint_values()
        # random_action[0] = 5.0
        # random_action[1] = 5.0
        # print(f"Sample action: {random_action}")
        # time.sleep(2)
        # self.step(random_action)

        N = 3
        for i in range(N):
            random_action = self.get_random_joint_config()
            print(f"Sample action {i+1}/{N}: {random_action}")
            self.step(random_action)

        # position_xyz = list(position_xyz)
        # position_xyz[2] += 0.1
        # position_xyz = tuple(position_xyz)
        # quaternion_xyzw = list(quaternion_xyzw)
        # quaternion_xyzw[1] = 1.0
        # quaternion_xyzw[3] = 0.0
        # quaternion_xyzw = tuple(quaternion_xyzw)
        # ik_solution = self.calculate_inverse_kinematics(position_xyz, quaternion_xyzw, True)
        # fk_solution = self.calculate_forward_kinematics(ik_solution, True)
        # self.step(action=ik_solution)

        # self.spin()

    def close(self):
        self.px_client.release()

    def step(self, action: list):
        done = False
        info = {}

        self.set_joint_position(action)

        obs = self.get_observation(verbose=False)

    def get_observation(self, verbose=False):
        obs = deepcopy(observation_template)
        obs["joint_values"] = self.get_joint_values()
        obs["base_velocity_normalized"] = self.get_base_velocity(normalized=True)
        obj_position_xyz, obj_quaternion_xyzw = self.get_object_pose()
        obs["object_pose"] = {"position_xyz": obj_position_xyz, "quaternion_xyzw": obj_quaternion_xyzw}

        if verbose:
            print("OBSERVATION:")
            self.print_joint_values(obs["joint_values"], title="Joint values:", tab_indent=1)
            print("\tBase velocity: %.5f" % obs["base_velocity_normalized"])
            self.print_pose(obs["object_pose"]["position_xyz"], obs["object_pose"]["quaternion_xyzw"], title="Object pose:", tab_indent=1)

        return obs
    
    def get_random_joint_config(self):
        return self.action_space.sample()
    
    def set_joint_position(self, q):
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
        base_velocity = self.get_base_velocity(normalized=True)
        robot_has_started_moving = False
        base_log_start, base_log_freq = start_time, 5.0
        while True:
            if not robot_has_started_moving and base_velocity > 1.0:
                robot_has_started_moving = True

            if time.time() - start_time > self.max_time_to_reach_goal:
                print(f"TIMED OUT on its way to goal, L2 norm of error across all joints: {error:.4f}, base_velocity: {base_velocity:.4f}")
                break
            
            if error < self.error_tolerance and robot_has_started_moving and base_velocity <= 0.001:
                print(f"REACHED GOAL in {time.time() - start_time:.4f} seconds, all joints error: {error:.4f},  base error: {self.get_base_error(q, self.get_joint_values()):.4f}, base velocity: {base_velocity:.4f}")
                self.print_joint_values(self.get_joint_values(), title="Joint values at goal:")
                break
            
            if time.time() - base_log_start > base_log_freq:
                base_log_start = time.time()
                print("Currently:")
                print(f"\tBase pose: {self.get_base_position_xy()}, velocity: {self.get_base_velocity()}, base error: {self.get_base_error(q, self.get_joint_values())}, all joint error: {error}")
                current_joint_values = self.get_joint_values()
                print(f"\tJoint values:")
                for i in range(len(all_joint_names)):
                    print(f"\t\t{all_joint_names[i]}: {current_joint_values[i]:.4f} (target: {q[i]:.4f}, error: {q[i] - current_joint_values[i]:.4f})")

            time.sleep(0.01)
            self.bullet_client.stepSimulation()
            error = self.get_error(q, self.get_joint_values())
            base_velocity = np.linalg.norm(self.get_base_velocity())

        self.bullet_client.removeUserDebugItem(line_id)
    
    def reset(self):
        self.robot_body_unique_id.reset()

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
            # self.get_joint_values(verbose=True)
            # print(f"SPINNING: Base velocity: {self.get_base_velocity()}")
            width, height, rgb_img, depth_img, seg_img = self.get_camera_image()
            self.bullet_client.stepSimulation()
            time.sleep(0.01)

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
    
    def get_base_velocity(self, normalized=False):
        base_vel = self.get_joint_velocities()[:2]  # X and Y components
        return np.linalg.norm(base_vel) if normalized else base_vel

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
    
    def get_object_pose(self):
        position_xyz, quaternion_xyzw = self.bullet_client.getBasePositionAndOrientation(
            bodyUniqueId=self.added_obj_id,
            physicsClientId=self.bullet_client._client
            )
        return position_xyz, quaternion_xyzw
    
    def add_object_to_scene(self, model_name: str, base_position: tuple):
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
            basePosition=base_position,
            baseOrientation=(0, 0, 0, 1),
            baseCollisionShapeIndex=collision_shape_id,
            baseVisualShapeIndex=visual_shape_id,
            baseInertialFramePosition=np.array(mesh.center_mass - centroid),
        )

        self.bullet_client.changeDynamics(obj_id, -1, lateralFriction=0.25)

        return obj_id


if __name__ == "__main__":
    env = HsrPybulletEnv()
    # check_env(env)