import time

import numpy as np
import pybullet as p
import pybullet_data
import trimesh

P_GAIN = 50

def main():
    c_gui = p.connect(p.GUI)

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf")

    robot_body_unique_id = robot_id = p.loadURDF("hsrb_description/robots/hsrb.urdf", useFixedBase=True, flags=p.URDF_USE_SELF_COLLISION)

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

    model_name = "007_tuna_fish_can"
    mesh_path = 'assets/ycb/{}/google_16k/nontextured.stl'.format(model_name)
    collision_path = 'assets/ycb/{}/google_16k/collision.obj'.format(model_name)
    mesh = trimesh.load(mesh_path, force='mesh', process=False)
    mesh.density = 150
    scale = 1

    centroid = mesh.centroid
    scale = 1
    scale = [scale, scale, scale]

    viz_shape_id = p.createVisualShape(
        shapeType=p.GEOM_MESH,
        rgbaColor=[1, 0, 0, 1],
        specularColor=[0.4, .4, 0],
        fileName=mesh_path, meshScale=scale,
        visualFramePosition=-centroid,
    )
    col_shape_id = p.createCollisionShape(
        shapeType=p.GEOM_MESH,
        fileName=collision_path, meshScale=scale,
        collisionFramePosition=-centroid,
    )

    obj_id = p.createMultiBody(
        baseMass=mesh.mass,
        basePosition=(1.0, 0, 0),
        baseCollisionShapeIndex=col_shape_id,
        baseVisualShapeIndex=viz_shape_id,
        baseOrientation=(0, 0, 0, 1),
        baseInertialFramePosition=np.array(mesh.center_mass - centroid),
    )

    p.changeDynamics(obj_id, -1, lateralFriction=0.25)

    p.setGravity(0, 0, -10)

    width = 128
    height = 128

    fov = 60
    aspect = width / height
    near = 0.02
    far = 1
    view_matrix = p.computeViewMatrix([0, 0, 0.5], [0, 0, 0], [1, 0, 0])
    projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)


    while True:
        images = p.getCameraImage(width, height, view_matrix, projection_matrix, renderer=p.ER_BULLET_HARDWARE_OPENGL) # Get depth values using the OpenGL renderer
        # images = p.getCameraImage(width, height, view_matrix, projection_matrix, renderer=p.ER_TINY_RENDERER)    # Get depth values using Tiny renderer
        depth_buffer_opengl = np.reshape(images[3], [width, height])
        depth_opengl = far * near / (far - (far - near) * depth_buffer_opengl)

        time.sleep(0.01)

        p.stepSimulation()

if __name__ == "__main__":
    main()