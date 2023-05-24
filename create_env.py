from isaacgym import gymapi
import random
import numpy as np
from isaacgym import gymtorch
import torch
from isaacgym.torch_utils import quat_apply, normalize


gym = gymapi.acquire_gym()

sim_params = gymapi.SimParams()

# set common parameters
sim_params.dt = 1 / 60
sim_params.substeps = 2
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)

# set PhysX-specific parameters
sim_params.physx.use_gpu = True
sim_params.physx.solver_type = 1
sim_params.physx.num_position_iterations = 6
sim_params.physx.num_velocity_iterations = 1
sim_params.physx.contact_offset = 0.01
sim_params.physx.rest_offset = 0.0

# set Flex-specific parameters
sim_params.flex.solver_type = 5
sim_params.flex.num_outer_iterations = 4
sim_params.flex.num_inner_iterations = 20
sim_params.flex.relaxation = 0.8
sim_params.flex.warm_start = 0.5

sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)

# set GPU pipeline parameters for tensor API
sim_params.use_gpu_pipeline = True
sim_params.physx.use_gpu = True

# create sim with these parameters
sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)

# configure the ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1) # z-up!
plane_params.distance = 0 # distance from origin
plane_params.static_friction = 1
plane_params.dynamic_friction = 1
plane_params.restitution = 0 # bounce

# create the ground plane
gym.add_ground(sim, plane_params)

asset_root = "./assets"
# asset_file = "aselsan/urdf/comar_generated.urdf"    

# asset_file = "go1/urdf/go1.urdf"
# asset_file = "anymal_c/urdf/anymal_c.urdf"
# asset_file = "anymal_b/urdf/anymal_b.urdf"
# asset_file = "aliengo/urdf/aliengo.urdf"
# asset_file = "comar/urdf/comar_a1.urdf"    
asset_file = "a1/urdf/a1.urdf"    




asset_options = gymapi.AssetOptions()
asset_options.armature = 0.0
asset_options.flip_visual_attachments = True

asset_options.fix_base_link = True
asset_options.disable_gravity = True

asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

# create environment

# set up the env grid
num_envs = 64
envs_per_row = 8
env_spacing = 2.0
env_lower = gymapi.Vec3(-env_spacing, 0.0, -env_spacing)
env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)

# cache some common handles for later use
envs = []
actor_handles = []

# create and populate the environments
for i in range(num_envs):
    env = gym.create_env(sim, env_lower, env_upper, envs_per_row)
    envs.append(env)

    height = 1.0

    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, 0.0, height)
    # pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)

    actor_handle = gym.create_actor(env, asset, pose, "MyActor", i, 1)
    actor_handles.append(actor_handle)

    props = gym.get_actor_dof_properties(env, actor_handle)
    props["driveMode"].fill(gymapi.DOF_MODE_EFFORT)
    props["stiffness"].fill(0.0)
    props["damping"].fill(0.0)
    gym.set_actor_dof_properties(env, actor_handle, props)

    num_dofs = gym.get_actor_dof_count(env, actor_handle)


# initialize the internal data structures used by the tensor API
gym.prepare_sim(sim)

_root_states = gym.acquire_actor_root_state_tensor(sim)
root_states = gymtorch.wrap_tensor(_root_states)

root_positions = root_states[:, 0:3]
root_orientations = root_states[:, 3:7]
root_linvels = root_states[:, 7:10]
root_angvels = root_states[:, 10:13]


_dof_states = gym.acquire_dof_state_tensor(sim)
dof_states = gymtorch.wrap_tensor(_dof_states)


cam_props = gymapi.CameraProperties()
viewer = gym.create_viewer(sim, cam_props)


while True:
    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)
    gym.sync_frame_time(sim)

    gym.refresh_actor_root_state_tensor(sim)
    gym.refresh_dof_state_tensor(sim)
    print(dof_states.view(num_envs, 12, 2)[..., 0])
    torque = torch.zeros((num_envs, 12), device="cuda")
    torque[..., ...] = 5
    # print(torque.shape)
    # gym.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(torque))
    
    # 0: FL_hip 
    # 1: FL_thigh
    # 2: FL_calf
    # 3: FR_hip
    # 4: FR_thigh
    # 5: FR_calf
    # 6: RL_hip
    # 7: RL_thigh
    # 8: RL_calf
    # 9: RR_hip
    # 10: RR_thigh
    # 11: RR_calf
    # measured_heights = _get_heights(None)
    # heights = torch.clip(root_states[:, 2].unsqueeze(1) - 0.5 - measured_heights, -1, 1.) * obs_scales.height_measurements


    # torque = np.zeros(shape=(num_envs, num_dofs))
    # torque[:, 2] = 100.0
    # actions = torch.tensor(torque, dtype=torch.float32, device="cuda:0")

    # gym.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(actions))

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)

