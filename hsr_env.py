import time
import pybullet as p
import pybullet_utils.bullet_client as bc
import pybullet_data
import pybulletX as px

from scipy.spatial.transform import Rotation as R
import numpy as np
import copy
import env_utils as eu
import ravens.utils.utils as ru
from gym.spaces import Box, Discrete

DISTAL_OPEN = -np.pi * 0.25
DISTAL_CLOSE = 0
PROXIMAL_OPEN = 1
PROXIMAL_CLOSE = -0.1
BOUNDS = 5

CAMERA_XTION_CONFIG = [{
    'image_size': (480, 640),
    'intrinsics': (537.4933389299223, 0.0, 319.9746375212718, 0.0, 536.5961755975517, 244.54846607953, 0.0, 0.0, 1.0),
    'position': None,
    'rotation': None,
    # 'lookat': (0, 0, 0),
    'zrange': (0.5, 10.),
    'noise': False
}]


# robot.set_actions({'joint_position': q}) exceeds max velocity, so use this fn
def set_joint_position(client, robot, joint_position, max_forces=None, use_joint_effort_limits=True):
    vels = robot.get_joint_infos()['joint_max_velocity']
    force = robot.get_joint_infos()['joint_max_force']
    robot.torque_control = False

    assert not np.all(np.array(max_forces) == 0), "max_forces can't be all zero"

    limits = robot.joint_effort_limits(robot.free_joint_indices)
    if max_forces is None and np.all(limits == 0):
        # warnings.warn(
        #     "Joint maximum efforts provided by URDF are zeros. "
        #     "Set use_joint_effort_limits to False"
        # )
        use_joint_effort_limits = False

    opts = {}
    if max_forces is None:
        if use_joint_effort_limits:
            # Case 1
            opts["forces"] = limits
        else:
            # Case 2: do nothing
            pass
    else:
        if use_joint_effort_limits:
            # Case 3
            opts["forces"] = np.minimum(max_forces, limits)
        else:
            # Case 4
            opts["forces"] = max_forces

    assert len(robot.free_joint_indices) == len(joint_position) or len(robot.free_joint_indices) - 4 == len(
        joint_position), (
        f"number of target positions ({len(joint_position)}) should match "
        f"the number of joint indices ({len(robot.free_joint_indices)})"
    )

    for i in range(len(joint_position)):
        client.setJointMotorControl2(robot.id, robot.free_joint_indices[i], p.POSITION_CONTROL,
                                     targetPosition=joint_position[i], maxVelocity=vels[i], force=force[i])


def pose2mat(pos, orn):
    m = np.eye(4)
    m[:3, -1] = pos
    m[:3, :3] = R.from_quat(orn).as_matrix()
    return m


class HSREnv:
    def __init__(self, connect=p.GUI):
        c_direct = bc.BulletClient(connection_mode=p.DIRECT)
        c_gui = bc.BulletClient(connection_mode=connect)

        c_direct.setGravity(0, 0, -9.8)
        c_gui.setGravity(0, 0, -9.8)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        planeId = c_gui.loadURDF('plane.urdf')
        c_gui.changeDynamics(planeId, -1, lateralFriction=0.5)

        px_gui = px.Client(client_id=c_gui._client)
        self.robot = px.Robot('hsrb_description/robots/hsrb.urdf', use_fixed_base=True, physics_client=px_gui)
        px_direct = px.Client(client_id=c_direct._client)
        self.robot_direct = px.Robot('hsrb_description/robots/hsrb.urdf', use_fixed_base=True, physics_client=px_direct)

        c_gui.changeVisualShape(self.robot.id, 15, rgbaColor=(1, 0.5, 0, 1))
        c_gui.changeVisualShape(self.robot.id, 34, rgbaColor=(0, 1, 0, 1))
        c_gui.changeVisualShape(self.robot.id, 35, rgbaColor=(1, 0, 0, 1))
        c_gui.changeVisualShape(self.robot.id, 38, rgbaColor=(1, 0, 1, 1))
        c_gui.changeVisualShape(self.robot.id, 40, rgbaColor=(0.5, 0, 0.5, 1))
        c_gui.changeVisualShape(self.robot.id, 44, rgbaColor=(0, 1, 1, 1))
        c_gui.changeVisualShape(self.robot.id, 46, rgbaColor=(0, 0.5, 0.5, 1))

        self.uppers, self.lowers, self.ranges, self.rest = self.get_robot_info()
        self.max_vels = self.robot.get_joint_infos()['joint_max_velocity']
        self.max_forces = self.robot.get_joint_infos()['joint_max_force']
        self.c_gui, self.c_direct = c_gui, c_direct

        vs_id = c_gui.createVisualShape(p.GEOM_SPHERE, radius=0.03, rgbaColor=[1, 0, 0, 1])
        self.marker_id = c_gui.createMultiBody(basePosition=[0, 0, 0], baseCollisionShapeIndex=-1,
                                               baseVisualShapeIndex=vs_id)

        self.break_criteria = lambda: False

        # print(self.robot.get_joint_infos())

    def get_robot_info(self):
        joints = self.robot.get_joint_infos()
        names = joints['joint_name']
        print(list(enumerate(names)))
        # print(self.robot.get_joint_infos(range(self.robot.num_joints)))
        # print(self.robot.get_joint_info_by_name('head_rgbd_sensor_gazebo_frame_joint'))

        uppers, lowers, ranges, rest = [], [], [], []

        for i, name in enumerate(names):
            ll = joints['joint_lower_limit'][i]
            ul = joints['joint_upper_limit'][i]

            lower = min(ll, ul)
            upper = max(ll, ul)

            lowers.append(lower)
            uppers.append(upper)
            ranges.append(upper - lower)
            rest.append(0)

        return uppers, lowers, ranges, rest

    def get_heightmap(self, only_render=False, bounds=np.array([[0, 3], [-1.5, 1.5], [0, 0.3]]), px_size=0.01,
                      **kwargs):
        m_pos, m_orn = self.c_gui.getBasePositionAndOrientation(self.marker_id)
        self.c_gui.resetBasePositionAndOrientation(self.marker_id, (0, 100, 0), (0, 0, 0, 1))

        head_state = self.robot.get_link_state_by_name('head_rgbd_sensor_gazebo_frame_joint')
        orn = list(head_state.world_link_frame_orientation)
        orn = (R.from_quat(orn) * R.from_euler('YZ', [0.5 * np.pi, -0.5 * np.pi])).as_quat()
        head_pose = list(head_state.world_link_frame_position), orn
        head_mat = pose2mat(*head_pose)

        base_state = self.robot.get_link_state_by_name('base_footprint_joint')
        base_pose = list(base_state.world_link_frame_position), list(base_state.world_link_frame_orientation)
        base_mat = pose2mat(*base_pose)

        camera_config = copy.deepcopy(CAMERA_XTION_CONFIG)
        camera_config[0]['position'] = list(head_state.world_link_frame_position)
        camera_config[0]['rotation'] = orn

        if only_render:
            rgb, depth, seg = eu.render_camera(self.c_gui, camera_config[0])

            head_rel_mat = np.matmul(np.linalg.inv(base_mat), head_mat)
            camera_config[0]['position'] = head_rel_mat[:3, -1]
            camera_config[0]['rotation'] = R.from_matrix(head_rel_mat[:3, :3]).as_quat()
            camera_config[0]['base_frame'] = base_mat

            out = rgb, depth, seg, camera_config[0]
        else:
            out = eu.get_heightmaps(self.c_gui, camera_config, bounds=bounds, px_size=px_size, **kwargs)

        self.c_gui.resetBasePositionAndOrientation(self.marker_id, m_pos, m_orn)

        return out

    def open_gripper(self):
        self.gripper_command(True)

    def close_gripper(self):
        self.gripper_command(False)

    def look_at(self, v, sim=False):
        orig_q = list(self.robot.get_states()['joint_position'])
        curr_q = np.array(orig_q)
        self.reset_joints(orig_q, False)
        cam_axis = [1, 0, 0]

        self.c_gui.changeVisualShape(self.marker_id, -1, rgbaColor=(0, 1, 0, 1))
        self.c_gui.resetBasePositionAndOrientation(self.marker_id, v + np.array([0, 0, 0.1]), [0, 0, 0, 1])

        for _ in range(1000):
            curr_state = self.robot_direct.get_link_state_by_name('head_rgbd_sensor_gazebo_frame_joint')
            head_pos = np.array(curr_state.world_link_frame_position)
            head_orn = np.array(curr_state.world_link_frame_orientation)
            head_mat = pose2mat(head_pos, head_orn)

            point_head = np.linalg.inv(head_mat).dot([v[0], v[1], v[2], 1.0])[:3]
            error_angle = np.arccos(np.dot(point_head, cam_axis) / np.linalg.norm(point_head))

            if np.abs(error_angle) > 0.5 * np.pi:
                curr_q[2] += np.pi
            else:
                if error_angle <= 0.01 * np.pi:
                    break

                normal_head = np.cross(cam_axis, point_head)
                normal_world = head_mat[:3, :3].dot(normal_head)
                normed_normal_world = normal_world / np.linalg.norm(normal_world)
                step_size = 0.5
                correction_euler = R.from_rotvec(normed_normal_world * step_size * error_angle).as_euler('xyz')

                link_id = self.robot_direct.get_joint_indices_by_names(['head_rgbd_sensor_gazebo_frame_joint'])[0]
                link_id = int(link_id)
                zeros = [0.0] * len(orig_q)
                j_t, j_r = self.c_direct.calculateJacobian(self.robot_direct.id, link_id, [0, 0, 0], orig_q,
                                                           zeros, zeros)

                j_r = np.array(j_r).T

                correction_conf = j_r.dot(correction_euler)
                mask = np.zeros_like(correction_conf)
                mask[2] = 1.0  # base rotation
                mask[5] = 1.0  # head tilt
                correction_conf *= mask

                curr_q = curr_q + correction_conf

            self.reset_joints(curr_q, False)

        if sim:
            self.set_joint_position(curr_q, True)
            self.steps()

            # print('target', v)
            # print('error;', error_angle)
            # input('ok?')
        else:
            self.reset_joints(curr_q, True)

    def set_joint_position(self, q, gui):
        if gui:
            client, robot = self.c_gui, self.robot
        else:
            client, robot = self.c_direct, self.robot_direct

        set_joint_position(client, robot, q)

    def move_joints(self, config, sim=True):
        for k, v in config.items():
            j = self.robot.get_joint_info_by_name(k)

            if not sim:
                self.c_gui.resetJointState(self.robot.id, j.joint_index, v)

            self.c_gui.setJointMotorControl2(self.robot.id, j.joint_index, p.POSITION_CONTROL,
                                             targetPosition=v, maxVelocity=j.joint_max_velocity,
                                             force=j.joint_max_force)

        if sim:
            self.steps()

    def gripper_command(self, open):
        q = self.robot.get_states()['joint_position']
        q[-1] = DISTAL_OPEN if open else DISTAL_CLOSE
        q[-2] = PROXIMAL_OPEN if open else PROXIMAL_CLOSE
        q[-3] = DISTAL_OPEN if open else DISTAL_CLOSE
        q[-4] = PROXIMAL_OPEN if open else PROXIMAL_CLOSE
        self.set_joint_position(q, True)

        self.steps()

    def reset_joints(self, q, gui):
        if gui:
            client, robot = self.c_gui, self.robot
        else:
            client, robot = self.c_direct, self.robot_direct

        for joint_index, joint_angle in zip(robot.free_joint_indices, q):
            client.resetJointState(robot.id, joint_index, joint_angle)

    def holding_config(self):
        q = [0 for _ in self.robot.get_states()['joint_position']]

        # arm joints
        q[5] = np.pi * -0.25
        q[8] = np.pi * 0.5
        q[9] = -np.pi * 0.5

        return q

    def reset_pose(self):
        neutral = self.holding_config()
        neutral[0] = np.random.uniform(-2, -1)# q[0]
        neutral[1] = np.random.uniform(-1, 1)# q[1]
        # neutral[2] = q[2]

        self.reset_joints(neutral, True)
        self.set_joint_position(neutral, True)

    def holding_pose(self):
        curr_q = list(self.robot.get_states()['joint_position'])
        q = self.holding_config()

        for i in [0, 1, 2]:
            q[i] = curr_q[i]

        self.set_joint_position(q, True)
        self.steps()

    def steps(self, steps=240 * 10, finger_steps=240, stop_at_stop=True, stop_at_contact=False):
        prev = self.robot.get_states()['joint_position']

        for t in range(steps):
            if self.break_criteria():
                break

            self.c_gui.stepSimulation()

            curr = self.robot.get_states()['joint_position']

            eq = np.abs(curr - prev) < 1e-4  # 1e-3 too large
            timeout = [t >= steps] * len(eq[:-4]) + [t >= finger_steps] * 4

            if stop_at_contact:
                is_in_contact = False
                points1 = self.c_gui.getContactPoints(bodyA=self.robot.id, linkIndexA=40)
                points2 = self.c_gui.getContactPoints(bodyA=self.robot.id, linkIndexA=46)
                if len(points1) > 0:
                    for p in points1:
                        if p[9] > 0:
                            is_in_contact = True
                            break
                if not is_in_contact and len(points2) > 0:
                    for p in points2:
                        if p[9] > 0:
                            is_in_contact = True
                            break
                if is_in_contact:
                    print('stopping at contact')
                    break

            if stop_at_stop and t > 10 and np.all(np.logical_or(eq, timeout)):
                # print('breaking at', t, 'steps')
                # print(eq, timeout)
                break

            prev = curr

        q = self.robot.get_states()['joint_position']
        q[2] %= (2*np.pi)
        self.reset_joints(q, True)

    def stepSimulation(self):
        self.c_gui.stepSimulation()

    def move_base(self, x, y, angle):
        q = list(self.robot.get_states()['joint_position'])
        q[0] = x
        q[1] = y
        q[2] = angle

        self.set_joint_position(q[:-4], True)
        self.steps()

    def move_ee(self, pos, orn, open=True, t=10, stop_at_contact=False):
        orig_q = list(self.robot.get_states()['joint_position'])

        self.c_gui.changeVisualShape(self.marker_id, -1, rgbaColor=(1, 0, 0, 1))
        self.c_gui.resetBasePositionAndOrientation(self.marker_id, pos, orn)
        self.reset_joints(orig_q, False)

        lowers, uppers, ranges = list(self.lowers), list(self.uppers), list(self.ranges)

        for i in [2, 4, 5]:
            lowers[i] = orig_q[i]
            uppers[i] = orig_q[i]
            # ranges[i] = 0.01

        success = False

        for i in range(10):
            q = self.c_direct.calculateInverseKinematics(self.robot_direct.id, 34, pos, orn, lowerLimits=lowers,
                                                         upperLimits=uppers,
                                                         jointRanges=ranges, restPoses=orig_q,
                                                         maxNumIterations=1000, residualThreshold=1e-4)
            q = list(q)
            self.reset_joints(q, False)

            gripper_state = self.c_direct.getLinkState(self.robot_direct.id, 34, computeForwardKinematics=1)

            pos_err = np.linalg.norm(np.array(gripper_state[4]) - pos)
            orn_err = 1 - np.dot(gripper_state[1], orn)**2.0

            # print(i, pos_err, orn_err)

            if pos_err < 0.01 and orn_err < 0.01:
                success = True
                break
            else:
                sample = list(np.array(self.robot.action_space.sample()['joint_position']))
                sample[0] %= 10.0
                sample[1] %= 10.0
                # sample[2] %= (2*np.pi)

                self.reset_joints(sample, False)

        if not success:
            print('FAILED TO FIND ACCURATE IK')

        # q[-1] = DISTAL_OPEN if open else DISTAL_CLOSE
        # q[-2] = PROXIMAL_OPEN if open else PROXIMAL_CLOSE
        # q[-3] = DISTAL_OPEN if open else DISTAL_CLOSE
        # q[-4] = PROXIMAL_OPEN if open else PROXIMAL_CLOSE

        if q[2] > orig_q[2]:
            diff = (q[2] - orig_q[2]) % (2 * np.pi)
            if diff > np.pi:
                q[2] = orig_q[2] - (np.pi * 2 - diff)
            else:
                q[2] = orig_q[2] + diff
        else:
            diff = (orig_q[2] - q[2]) % (2 * np.pi)
            if diff > np.pi:
                q[2] = orig_q[2] + (np.pi * 2 - diff)
            else:
                q[2] = orig_q[2] - diff

        # set_joint_position(robot, orig_q, sim=False)

        self.set_joint_position(q[:-4], True)

        steps = 240 * t
        return self.steps(steps, stop_at_contact=stop_at_contact)

    def grasp_primitive(self, pos, angle=0, frame=None, stop_at_contact=False):
        if frame is None:
            frame = np.eye(4)

        mat = np.eye(4)
        mat[:3, -1] = pos
        down = p.getQuaternionFromEuler([np.pi, 0, angle])
        mat[:3, :3] = R.from_quat(down).as_matrix()

        mat = frame.dot(mat)
        pos = mat[:3, -1]
        down = R.from_matrix(mat[:3, :3]).as_quat()

        if np.abs(pos[0]) < BOUNDS and np.abs(pos[1]) < BOUNDS:
            self.move_ee(pos + np.array([0, 0, 0.3]), down)

            if self.break_criteria():
                return True

            self.open_gripper()
            if self.break_criteria():
                return True

            self.move_ee(pos, down, stop_at_contact=stop_at_contact)
            if self.break_criteria():
                return True
            # print('close')

            self.close_gripper()
            # print('done')
            # input('ok?')

            return True

        return False


DEFAULT_CONFIG = {
    'depth_noise': False,
    'rot_noise': False,
    'action_grasp': True,
    'action_look': False,
    'spawn_mode': 'box',
}


class GraspEnv:
    def __init__(self, n_objects=70, config=DEFAULT_CONFIG, setup_room=True, **kwargs):
        self.env = HSREnv(**kwargs)
        self.obj_ids = eu.spawn_ycb(self.env.c_gui, ids=list(range(n_objects)))

        if setup_room:
            self.furn_ids = self.generate_room()
        else:
            self.furn_ids = []

        self.res = 224
        self.px_size = 3.0 / self.res
        self.num_rots = 16
        self.stats = {
            'object_collisions': 0,
            'furniture_collisions': 0,
            'episodes': 0,
            'grasp_success_collision': 0,
            'grasp_success_safe': 0,
            'grasp_failure_collision': 0,
            'grasp_failure_safe': 0,
            'grasp_attempts': 0,
            'oob_actions': 0,
        }

        self.observation_space = Box(-1, 1, (self.res, self.res))
        self.config = config
        self.seed = None

        n_actions = (int(config['action_grasp']) * self.num_rots + int(config['action_look'])) * self.res * self.res
        self.action_space = Discrete(n_actions)

        self.hmap, self.obs_config, self.segmap = None, None, None

        self.dummy = np.zeros((3, self.res, self.res), dtype=np.float32)
        self.hmap_bounds = np.array([[0, 3], [-1.5, 1.5], [-0.05, 1]])

        self.spawn_mode = 'box'# config['spawn_mode']
        self.spawn_box = [[-1.5, -1, 0.4], [-0.5, 1.5, 0.6]]# [[0.5, -1.5, 0.4], [3.0, 1.5, 0.6]]
        self.spawn_radius = 3

        self.steps = 0

        self.object_collision, self.furniture_collision = False, False

        def wrapper(fn):
            def wrapper():
                fn()

                if not self.object_collision:
                    for id in self.obj_ids:
                        if len(self.env.c_gui.getClosestPoints(self.env.robot.id, id, 0, 15, -1)) > 0:
                            self.object_collision = True
                            break

                if not self.furniture_collision:
                    for id in self.furn_ids:
                        if len(self.env.c_gui.getClosestPoints(self.env.robot.id, id, 0)) > 0:
                            self.furniture_collision = True
                            break

            return wrapper

        def break_criteria():
            return self.furniture_collision

        self.env.c_gui.stepSimulation = wrapper(self.env.c_gui.stepSimulation)
        self.env.break_criteria = break_criteria

    def generate_room(self):
        ids = []

        x = self.env.c_gui.loadSDF('tmc_wrs_gazebo/tmc_wrs_gazebo_worlds/models/wrc_frame/model.sdf')[0]
        self.env.c_gui.resetBasePositionAndOrientation(x, (0, 0, 0), (0, 0, 0, 1))
        ids.append(x)

        x = self.env.c_gui.loadSDF('tmc_wrs_gazebo/tmc_wrs_gazebo_worlds/models/wrc_bookshelf/model.sdf')[0]
        self.env.c_gui.resetBasePositionAndOrientation(x, (2.7, -1, 0), p.getQuaternionFromEuler([0, 0, -1.57]))
        ids.append(x)

        x = self.env.c_gui.loadSDF('tmc_wrs_gazebo/tmc_wrs_gazebo_worlds/models/wrc_bin_black/model.sdf')[0]
        self.env.c_gui.resetBasePositionAndOrientation(x, (-2.7, -1.7, 0), (0, 0, 0, 1))
        self.env.c_gui.changeVisualShape(x, -1, rgbaColor=(0.3, 0.3, 0.3, 1))
        ids.append(x)

        x = self.env.c_gui.loadSDF('tmc_wrs_gazebo/tmc_wrs_gazebo_worlds/models/wrc_bin_green/model.sdf')[0]
        self.env.c_gui.resetBasePositionAndOrientation(x, (-2.7, -1.2, 0), (0, 0, 0, 1))
        self.env.c_gui.changeVisualShape(x, -1, rgbaColor=(0, 0.7, 0, 1))
        ids.append(x)

        x = self.env.c_gui.loadSDF('tmc_wrs_gazebo/tmc_wrs_gazebo_worlds/models/wrc_stair_like_drawer/model.sdf')[0]
        self.env.c_gui.resetBasePositionAndOrientation(x, (-2.7, 1, 0), (0, 0, 0, 1))
        ids.append(x)

        x = self.env.c_gui.loadSDF('tmc_wrs_gazebo/tmc_wrs_gazebo_worlds/models/trofast_knob/model-1_4.sdf')[0]
        self.env.c_gui.resetBasePositionAndOrientation(x, (-2.7, 0.67, 0.1 + 0.115), (0, 0, 0, 1))
        self.env.c_gui.changeVisualShape(x, -1, rgbaColor=(1, 0.5, 0, 1))
        ids.append(x)

        x = self.env.c_gui.loadSDF('tmc_wrs_gazebo/tmc_wrs_gazebo_worlds/models/trofast_knob/model-1_4.sdf')[0]
        self.env.c_gui.resetBasePositionAndOrientation(x, (-2.7, 1, 0.1 + 0.115), (0, 0, 0, 1))
        self.env.c_gui.changeVisualShape(x, -1, rgbaColor=(1, 0.5, 0, 1))
        ids.append(x)

        x = self.env.c_gui.loadSDF('tmc_wrs_gazebo/tmc_wrs_gazebo_worlds/models/trofast_knob/model-1_4.sdf')[0]
        self.env.c_gui.resetBasePositionAndOrientation(x, (-2.7, 1, 0.36 + 0.115), (0, 0, 0, 1))
        self.env.c_gui.changeVisualShape(x, -1, rgbaColor=(1, 0.5, 0, 1))
        ids.append(x)

        x = self.env.c_gui.loadSDF('tmc_wrs_gazebo/tmc_wrs_gazebo_worlds/models/wrc_tall_table/model.sdf')[0]
        self.env.c_gui.resetBasePositionAndOrientation(x, (-0.3, 1.2, 0), (0, 0, 0, 1))
        ids.append(x)

        x = self.env.c_gui.loadSDF('tmc_wrs_gazebo/tmc_wrs_gazebo_worlds/models/wrc_long_table/model.sdf')[0]
        self.env.c_gui.resetBasePositionAndOrientation(x, (-0.3, 0.2, 0), p.getQuaternionFromEuler([0, 0, 1.57]))
        ids.append(x)

        x = self.env.c_gui.loadSDF('tmc_wrs_gazebo/tmc_wrs_gazebo_worlds/models/wrc_long_table/model.sdf')[0]
        self.env.c_gui.resetBasePositionAndOrientation(x, (-2.7, -0.3, 0), p.getQuaternionFromEuler([0, 0, 1.57]))
        ids.append(x)

        x = self.env.c_gui.loadSDF('tmc_wrs_gazebo/tmc_wrs_gazebo_worlds/models/wrc_tray/model.sdf')[0]
        self.env.c_gui.resetBasePositionAndOrientation(x, (-2.7, -0.75, 0.4), p.getQuaternionFromEuler([0, 0, 1.57]))
        self.env.c_gui.changeVisualShape(x, -1, rgbaColor=(0.3, 0.3, 0.3, 1))
        ids.append(x)

        x = self.env.c_gui.loadSDF('tmc_wrs_gazebo/tmc_wrs_gazebo_worlds/models/wrc_tray/model.sdf')[0]
        self.env.c_gui.resetBasePositionAndOrientation(x, (-2.7, -0.45, 0.4), p.getQuaternionFromEuler([0, 0, 1.57]))
        self.env.c_gui.changeVisualShape(x, -1, rgbaColor=(0.3, 0.3, 0.3, 1))
        ids.append(x)

        x = self.env.c_gui.loadSDF('tmc_wrs_gazebo/tmc_wrs_gazebo_worlds/models/wrc_container_a/model.sdf')[0]
        self.env.c_gui.resetBasePositionAndOrientation(x, (-2.7, -0.2, 0.4), (0, 0, 0, 1))
        self.env.c_gui.changeVisualShape(x, -1, rgbaColor=(0.8, 0.0, 0.0, 1))
        ids.append(x)

        x = self.env.c_gui.loadSDF('tmc_wrs_gazebo/tmc_wrs_gazebo_worlds/models/wrc_container_b/model.sdf')[0]
        self.env.c_gui.resetBasePositionAndOrientation(x, (-2.7, 0.1, 0.4), (0, 0, 0, 1))
        self.env.c_gui.changeVisualShape(x, -1, rgbaColor=(0.3, 0.3, 0.2, 1))
        ids.append(x)

        return ids

    def random_action_sample_fn(config, uniform=False):
        if uniform:
            def fn():
                return np.random.randint((config['rots'] + 1) * config['res'] * config['res'])
        else:
            def fn():
                primitive = np.random.randint(int(config['action_grasp']) + int(config['action_look']))
                if primitive == 0:
                    return np.random.randint(config['rots'] * config['res'] * config['res'])
                elif primitive == 1:
                    offset = config['rots'] * config['res'] * config['res']
                    return offset + np.random.randint(config['res'] * config['res'])

                raise Exception('invalid primitive')

        return fn

    def reset(self):
        self.ep_start_time = time.time()

        for id in self.obj_ids:
            self.env.c_gui.resetBasePositionAndOrientation(id, (-100, np.random.uniform(-100, 100), -100), (0, 0, 0, 1))
            self.env.c_gui.changeDynamics(id, -1, mass=0)

        num_objs = np.random.randint(1, 30)
        selected = np.random.permutation(self.obj_ids)[:num_objs]

        self.env.reset_pose()

        self.env.move_joints({
            'joint_rz': np.random.uniform(-np.pi, np.pi),
            'head_tilt_joint': np.random.uniform(-1.57, 0),
            # 'head_pan_joint': np.random.uniform(np.pi * -0.25, np.pi * 0.25),
        }, sim=False)

        for i, id in enumerate(selected):
            if self.spawn_mode == 'box':
                x = np.random.uniform(self.spawn_box[0][0], self.spawn_box[1][0])
                y = np.random.uniform(self.spawn_box[0][1], self.spawn_box[1][1])
                z = np.random.uniform(self.spawn_box[0][2], self.spawn_box[1][2])
            else:
                theta = np.random.uniform(0, 2.0 * np.pi)
                dist = np.random.uniform(0.5, self.spawn_radius)
                x, y = np.cos(theta) * dist, np.sin(theta) * dist
                z = np.random.uniform(0.4, 0.6)

            pos = (x, y, z)
            # pos = self.pos[i]

            for t in range(10):
                self.env.c_gui.resetBasePositionAndOrientation(id, pos, R.random().as_quat())
                valid = True

                if len(self.env.c_gui.getClosestPoints(id, self.env.robot.id, 0)) > 0:
                    valid = False
                else:
                    for prev in selected[:i]:
                        if len(self.env.c_gui.getClosestPoints(id, prev, 0)) > 0:
                            valid = False
                            break

                if valid:
                    break

            self.env.c_gui.changeDynamics(id, -1, mass=0.1)

        # print('settle')
        x = [self.env.c_gui.getBasePositionAndOrientation(i)[0] for i in selected]
        for t in range(240 * 10):
            self.env.c_gui.stepSimulation()
            y = [self.env.c_gui.getBasePositionAndOrientation(i)[0] for i in selected]

            diff = np.abs(np.array(x) - np.array(y))
            if t > 10 and np.all(diff < 1e-3):
                # print('breaking at', t)
                break

            x = y
        # print('settle done')

        hmap = self.update_obs()

        self.steps = 0

        # import time
        # for x in np.linspace(0, 2*np.pi, 100):
        #     self.env.look_at([2 + np.sin(x), np.cos(x), 0])
        #     time.sleep(0.01)

        self.object_collision, self.furniture_collision = False, False

        return hmap

    def update_obs(self):
        rgb, depth, seg, config = self.env.get_heightmap(only_render=True, return_seg=True, bounds=self.hmap_bounds,
                                                         px_size=self.px_size)

        hmap, cmap, segmap = to_maps(rgb, depth, seg, config, self.hmap_bounds, self.px_size,
                                     depth_noise=self.config['depth_noise'], rot_noise=self.config['rot_noise'])

        assert hmap.shape[0] == self.res and hmap.shape[1] == self.res, 'resolutions do not match {} {}'.format(
            hmap.shape, self.res)

        self.rgb = rgb
        self.depth = depth
        self.seg = seg
        self.cmap = cmap

        self.hmap = hmap
        self.obs_config = config
        self.segmap = segmap

        return hmap[None]

    def set_seed(self, idx):
        self.seed = idx
        np.random.seed(idx)

    def step(self, action):
        action_type = None
        max_grasp_idx = self.num_rots * self.res * self.res

        if action < max_grasp_idx:
            rot_idx = int(action / (self.res * self.res))
            loc_idx = action % (self.res * self.res)
            grasp_py = int(loc_idx / self.res)
            grasp_px = int(loc_idx % self.res)
            action_type = 'grasp'

            # import matplotlib.pyplot as plt
            # plt.imshow(self.hmap)
            # plt.title('grasp')
            # plt.plot([grasp_px], [grasp_py], '*r')
            # plt.show()
        else:
            idx = action - max_grasp_idx
            idx %= (self.res * self.res)
            look_py = int(idx / self.res)
            look_px = int(idx % self.res)
            action_type = 'look'

            # import matplotlib.pyplot as plt
            # plt.imshow(self.hmap)
            # plt.title('look')
            # plt.plot([look_px], [look_py], '*r')
            # plt.show()

        reward = 0
        done = False

        # print(action_type)

        if action_type == 'grasp':
            grasp_x = [grasp_py, grasp_px]
            num_rots = 16
            angle = rot_idx * 2 * np.pi / num_rots

            # px in y, x axis in that order

            x = self.hmap_bounds[0, 0] + grasp_x[1] * self.px_size
            y = self.hmap_bounds[1, 0] + grasp_x[0] * self.px_size

            surface_height = 0
            self.hmap[self.hmap == 0] = surface_height - self.hmap_bounds[2, 0]
            z = self.hmap[grasp_x[0], grasp_x[1]] + self.hmap_bounds[2, 0]

            z += 0.24 - 0.07

            if self.env.grasp_primitive([x, y, z], angle, frame=self.obs_config['base_frame'], stop_at_contact=False):
                self.env.holding_pose()

                if self.env.break_criteria():
                    grasp_success = False
                else:
                    for _ in range(240):
                        self.env.stepSimulation()

                    grasp_success = False
                    obj = None

                    points = self.env.c_gui.getContactPoints(bodyA=self.env.robot.id, linkIndexA=40)
                    for c in points:
                        if c[2] in self.obj_ids:
                            grasp_success = True
                            obj = c[2]
                            break
                    if not grasp_success:
                        points = self.env.c_gui.getContactPoints(bodyA=self.env.robot.id, linkIndexA=46)
                        for c in points:
                            if c[2] in self.obj_ids:
                                grasp_success = True
                                obj = c[2]
                                break

                    if grasp_success:
                        self.env.c_gui.resetBasePositionAndOrientation(obj, (-100, np.random.uniform(-100, 100), -100),
                                                                       (0, 0, 0, 1))

                reward = 1 if grasp_success else -0.1

                if grasp_success:
                    if self.object_collision:
                        self.stats['grasp_success_collision'] += 1
                    else:
                        self.stats['grasp_success_safe'] += 1
                else:
                    if self.object_collision:
                        self.stats['grasp_failure_collision'] += 1
                    else:
                        self.stats['grasp_failure_safe'] += 1

                self.stats['grasp_attempts'] += 1
                # done |= grasp_success
            else:
                self.stats['oob_actions'] += 1
                reward = -0.25
        elif action_type == 'look':
            surface_height = 0
            self.hmap[self.hmap == 0] = surface_height - self.hmap_bounds[2, 0]
            look_z = self.hmap[look_py, look_px] + self.hmap_bounds[2, 0]
            look_x = self.hmap_bounds[0, 0] + look_px * self.px_size
            look_y = self.hmap_bounds[1, 0] + look_py * self.px_size

            look_v = self.obs_config['base_frame'].dot([look_x, look_y, look_z, 1])[:3]

            # if np.abs(look_x) < BOUNDS and np.abs(look_y) < BOUNDS:
            self.env.look_at(look_v, sim=True)

            # else:
            #    reward = -0.25
        else:
            raise Exception('no valid action')

        self.stats['furniture_collisions'] += int(self.furniture_collision)
        self.stats['object_collisions'] += int(self.object_collision)

        if self.furniture_collision:
            reward = -0.25
            done = True
        elif self.object_collision:
            reward = min(reward * 0.1, reward)
            # done = True

        if done:
            self.stats['episodes'] += 1

            print('seed:', self.seed, self.stats, time.time() - self.ep_start_time)

        hmap = self.update_obs()

        return hmap, reward, done, {}


def to_maps(rgb, depth, seg, config, bounds, px_size, depth_noise=False, pos_noise=False, rot_noise=False):
    config = copy.deepcopy(config)

    if depth_noise:
        #if np.random.uniform() < 0.5:
        depth = eu.distort(depth, noise=np.random.uniform(0, 1))

    if pos_noise:
        config['position'] = np.array(config['position']) + np.random.normal(0, 0.01, 3)

    if rot_noise:
        rvec = np.random.normal(0, 1, 3)
        rvec /= np.linalg.norm(rvec)
        mag = 1 / 180.0 * np.pi

        rot = R.from_quat(config['rotation'])
        config['rotation'] = (R.from_rotvec(mag * rvec) * rot).as_quat()

    hmaps, cmaps = ru.reconstruct_heightmaps(
        [rgb], [depth], [config], bounds, px_size)
    _, segmaps = ru.reconstruct_heightmaps(
        [seg[:, :, None]], [depth], [config], bounds, px_size)

    return hmaps[0], cmaps[0], segmaps[0]
