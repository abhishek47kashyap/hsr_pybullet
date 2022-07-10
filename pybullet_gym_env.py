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

        self.error_tolerance = 1.0
        self.max_time_to_reach_goal = 20  # seconds allowed to reach goal

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

        self.add_object_to_scene(model_name="007_tuna_fish_can")

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

        # self.spin()

    def close(self):
        self.px_client.release()

    def step(self, action: list):
        self.set_joint_position(action)
    
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
        base_velocity = np.linalg.norm(self.get_base_velocity())
        robot_has_started_moving = False
        base_log_start, base_log_freq = start_time, 5.0
        while True:
            if not robot_has_started_moving and base_velocity > 1.0:
                robot_has_started_moving = True

            if time.time() - start_time > self.max_time_to_reach_goal:
                print(f"TIMED OUT on its way to goal, L2 norm of error across all joints: {error}")
                break
            
            if error < self.error_tolerance and robot_has_started_moving and base_velocity <= 0.001:
                print(f"REACHED GOAL in {time.time() - start_time} seconds, all joints error: {error},  base error: {self.get_base_error(q, self.get_joint_values())}, base velocity: {base_velocity}")
                print(f"Joint values at goal: {self.get_joint_values()}")
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
            print(f"All joint values: {joint_values}")
            # for name, value in zip(all_joint_names, joint_values):
            #     print(f"\t{name}: {value}")

        return joint_values
    
    def get_joint_velocities(self, verbose=False):
        return list(self.robot_body_unique_id.get_states().joint_velocity)
    
    def get_joint_max_velocities_and_forces(self, verbose=False):
        joint_infos = self.robot_body_unique_id.get_joint_infos()
        return joint_infos["joint_max_velocity"], joint_infos["joint_max_force"]
    
    def get_base_position_xy(self):
        return self.get_joint_values()[:2]   # first two elements are x and y
    
    def get_base_velocity(self):
        return self.get_joint_velocities()[:2]

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
    # check_env(env)