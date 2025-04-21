import sys
from legged_gym import LEGGED_GYM_ROOT_DIR
import os
from datetime import datetime

import isaacgym
from isaacgym import gymapi
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, export_policy_as_onnx, task_registry, Logger

import numpy as np
import torch
from termcolor import colored
import hydra
from omegaconf import DictConfig, OmegaConf
from easydict import EasyDict
from legged_gym.utils.helpers import class_to_dict
import cv2
from phc.utils import torch_utils

NOROSPY = False
RENDER = False

try:
    import rospy
except:
    NOROSPY = True
# from std_msgs.msg import String, Header, Float64MultiArray

command_state = {
    'vel_forward': 0.0,
    'vel_side': 0.0,
    'orientation': 0.0,
}

override = False

EXPORT_ONNX = False


def dict_compare(d1, d2):
    d1_keys = set(d1.keys())
    d2_keys = set(d2.keys())
    shared_keys = d1_keys.intersection(d2_keys)
    added = d1_keys - d2_keys
    removed = d2_keys - d1_keys
    modified = {o : (d1[o], d2[o]) for o in shared_keys if d1[o] != d2[o]}
    same = set(o for o in shared_keys if d1[o] == d2[o])
    return added, removed, modified, same

@hydra.main(
    version_base=None,
    config_path="../cfg",
    config_name="config_teleop",
)
def play(cfg_hydra: DictConfig) -> None:
    cfg_hydra = EasyDict(OmegaConf.to_container(cfg_hydra, resolve=True))
    cfg_hydra.physics_engine = gymapi.SIM_PHYSX
     
    env_cfg, train_cfg = cfg_hydra, cfg_hydra.train
    

    ##### Compare two configs. 
    # env_cfg_, train_cfg_prev = task_registry.get_cfgs(name=cfg_hydra.task)
    # env_cfg_, train_cfg_prev = class_to_dict(env_cfg_), class_to_dict(train_cfg_prev)
    # for k, v in env_cfg_.items():
    #     if isinstance(v, dict):
    #         for kk, vv in v.items():
    #             if not vv == env_cfg[k][kk]:
    #                 print(k, kk)
    #                 import ipdb; ipdb.set_trace()
    #                 print('...')
        
    #     elif not v == env_cfg[k]:
    #         import ipdb; ipdb.set_trace()
    #         print('...')

    # override some parameters for testing
    # env_cfg.env.num_envs = min(env_cfg.env.num_envs, 100)

    # if not env_cfg.train_velocity_estimation:
    env_cfg.env.num_envs = 1
    env_cfg.viewer.debug_viz = True
    env_cfg.motion.visualize = False
    # env_cfg.terrain.num_rows = 5
    # env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.mesh_type = 'plane'
    # env_cfg.terrain.mesh_type = 'plane'
    # if env_cfg.terrain.mesh_type == 'trimesh':
    #     env_cfg.terrain.terrain_types = ['flat', 'rough', 'low_obst']  # do not duplicate!
    #     env_cfg.terrain.terrain_proportions = [1.0, 0.0, 0.0]
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.env.episode_length_s = 20
    env_cfg.domain_rand.randomize_rfi_lim = False
    env_cfg.domain_rand.randomize_pd_gain = False
    env_cfg.domain_rand.randomize_link_mass = False
    env_cfg.domain_rand.randomize_base_com = False
    env_cfg.domain_rand.randomize_ctrl_delay = False
    env_cfg.domain_rand.ctrl_delay_step_range = [1, 3]



    # env_cfg.asset.termination_scales.max_ref_motion_distance = 1



    env_cfg.env.test = True

    if env_cfg.motion.realtime_vr_keypoints:
        env_cfg.asset.terminate_by_1time_motion = False
        env_cfg.asset.terminate_by_ref_motion_distance = False
        rospy.init_node("avppose_subscriber")
        from avp_pose_subscriber import AVPPoseInfo
        avpposeinfo = AVPPoseInfo()
        rospy.Subscriber("avp_pose", Float64MultiArray, avpposeinfo.avp_callback, queue_size=1)
    if cfg_hydra.joystick:
        env_cfg.commands.ranges.lin_vel_x = [0.0, 0.0]
        env_cfg.commands.ranges.lin_vel_y = [0.0, 0.0]
        env_cfg.commands.ranges.ang_vel_yaw = [0.0, 0.0]
        from pynput import keyboard
        from legged_gym.utils import key_response_fn

    # prepare environment
    
    env, _ = task_registry.make_env_hydra(name=cfg_hydra.task, hydra_cfg=cfg_hydra, env_cfg=env_cfg)

    logger = Logger(env.dt)
    robot_index = 0 # which robot is used for logging

    # 15是右肩pitch 18右手肘（较好） 14左手肘 11左肩picth(较好)
    joint_index = 18 # which joint is used for logging (从0开始计算)
    stop_state_log = 198 # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards


    obs = env.get_observations()
    
    if env_cfg.motion.realtime_vr_keypoints:
        init_root_pos = env._rigid_body_pos[..., 0, :].clone()
        init_avp_pos = avpposeinfo.avp_pose.copy()
        init_root_offset = init_root_pos[0, :2] - init_avp_pos[2, :2]
    # import ipdb; ipdb.set_trace()
    # obs[:, 9:12] = torch.Tensor([0.5, 0, 0])
    # load policy
    train_cfg.runner.resume = True
    
    
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=cfg_hydra.task, args=cfg_hydra, train_cfg=train_cfg)
    
    policy = ppo_runner.get_inference_policy(device=env.device)
    exported_policy_name = str(task_registry.loaded_policy_path.split('/')[-2]) + str(task_registry.loaded_policy_path.split('/')[-1])
    print('Loaded policy from: ', task_registry.loaded_policy_path)

    # export policy as a jit module (used to run it from C++)
    # if EXPORT_POLICY:
    #     path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
    #     export_policy_as_jit(ppo_runner.alg.actor_critic, path, exported_policy_name)
    #     print('Exported policy as jit script to: ', os.path.join(path, exported_policy_name))
    # if EXPORT_ONNX:
    #     exported_onnx_name = exported_policy_name.replace('.pt', '.onnx')
    #     path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
    #     export_policy_as_onnx(ppo_runner.alg.actor_critic, path, exported_onnx_name, onnx_num_observations=env_cfg.env.num_observations)
    #     print('Exported policy as onnx to: ', os.path.join(path, exported_onnx_name))
        
    
    if cfg_hydra.joystick:
        print(colored("joystick on", "green"))
        key_response = key_response_fn(mode='vel')
        def on_press(key):
            global command_state
            try:
                # print(key.char)
                key_response(key, command_state, env)
            except AttributeError:
                pass

        listener = keyboard.Listener(on_press=on_press)
        listener.start()

    i = 0
    

    '''
    视频录制
    '''
    if RENDER:
        camera_properties = gymapi.CameraProperties()
        camera_properties.width = 480
        camera_properties.height = 640
        h1 = env.gym.create_camera_sensor(env.envs[0], camera_properties)
        # camera_offset = gymapi.Vec3(1.0, -0.8, 0.3)
        camera_offset = gymapi.Vec3(1.2, -1.0, 0.3)
        camera_rotation = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1),
                                                    np.deg2rad(135))
        actor_handle = env.gym.get_actor_handle(env.envs[0], 0)
        body_handle = env.gym.get_actor_rigid_body_handle(env.envs[0], actor_handle, 0)
        env.gym.attach_camera_to_body(
            h1, env.envs[0], body_handle,
            gymapi.Transform(camera_offset, camera_rotation),
            gymapi.FOLLOW_POSITION)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'videos')
        experiment_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'videos',datetime.now().strftime('%b%d_%H-%M-%S'))
        dir = os.path.join(experiment_dir, 'h1.mp4')
        if not os.path.exists(video_dir):
            os.mkdir(video_dir)
        if not os.path.exists(experiment_dir):
            os.mkdir(experiment_dir)
        # 帧率为50
        video = cv2.VideoWriter(dir, fourcc, 50.0, (480, 640))

    # 计算mpjpe/time
    total_mpjpe = 0

    # while (not NOROSPY and not rospy.is_shutdown()) or (NOROSPY and i<=226):
    while (not NOROSPY and not rospy.is_shutdown()) or (NOROSPY):
        # for i in range(1000*int(env.max_episode_length)):

        # obs[:, -19:] = 0 # will destroy the performance


        # 计算MPJPE
        # 计算root relative的MPJPE（先合并关节再减去root）
        body_pos = env._rigid_body_pos
        body_rot = env._rigid_body_rot

        # 获取参考姿势
        offset = env.env_origins + env.env_origins_init_3Doffset
        motion_times = (env.episode_length_buf) * env.dt + env.motion_start_times
        motion_res = env._get_state_from_motionlib_cache_trimesh(env.motion_ids, motion_times, offset=offset)
        ref_body_pos_extend = motion_res['rg_pos_t']  # 参考姿势（已包含扩展关节）

        # 计算当前姿势的扩展关节位置
        extend_curr_pos = torch_utils.my_quat_rotate(
        body_rot[:, env.extend_body_parent_ids].reshape(-1, 4), 
        env.extend_body_pos.reshape(-1, 3)
        ).view(env.num_envs, -1, 3) + body_pos[:, env.extend_body_parent_ids]

        # 合并原始关节和扩展关节（当前姿势）
        body_pos_extend = torch.cat([body_pos, extend_curr_pos], dim=1)

        # 获取根关节位置（假设索引0是根关节）
        root_pos = body_pos_extend[:, [0]]  # 当前姿势的根关节
        # print(root_pos.shape)
        ref_root_pos = ref_body_pos_extend[:, [0]]  # 参考姿势的根关节

        # 转换为root relative坐标（所有关节统一处理）
        body_pos_relative = body_pos_extend - root_pos
        ref_body_pos_relative = ref_body_pos_extend - ref_root_pos

        # 计算MPJPE
        mpjpe = torch.norm(ref_body_pos_relative - body_pos_relative, p=2, dim=-1)  # L2距离
        mean_mpjpe = mpjpe.mean()  # 平均关节位置误差
        
        # 计算平均MPJPE
        # if(i<=226):
        #     total_mpjpe += mean_mpjpe.cpu().numpy()
        #     if(i==226):
        #         print(total_mpjpe/226)

        actions = policy(obs.detach())
        
        # print(torch.sum(torch.square(env.projected_gravity[:, :2]), dim=1))
        obs, _, rews, dones, infos = env.step(actions.detach())

        if env_cfg.motion.teleop_obs_version == 'v-teleop-dof-nolinvel-history-max':
            target_joint_index = -63 + joint_index 
            target_joint = obs[0][target_joint_index].cpu().numpy()
            # print(target_joint)
        else:
            target_joint = actions[robot_index, joint_index].item() * env.cfg.control.action_scale + env.default_dof_pos[robot_index, joint_index].item()
            

        # 保存图像
        if RENDER:
            env.gym.fetch_results(env.sim, True)
            env.gym.step_graphics(env.sim)
            env.gym.render_all_camera_sensors(env.sim)
            img = env.gym.get_camera_image(env.sim, env.envs[0], h1, gymapi.IMAGE_COLOR)
            height, width = camera_properties.height, camera_properties.width
            img = np.reshape(img, (height, width, 4))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            frame_path = os.path.join(experiment_dir, f"frame_{i}.png")
            cv2.imwrite(frame_path, img)
            # tiff_path = os.path.join(experiment_dir, f"frame_{i}.tiff")
            # cv2.imwrite(tiff_path, img, [cv2.IMWRITE_TIFF_COMPRESSION, 1])  
            video.write(img[..., :3])

        if env_cfg.motion.realtime_vr_keypoints:
            avpposeinfo.check()
            keypoints_pos = avpposeinfo.avp_pose.copy()
            keypoints_pos[:, 0] += init_root_offset[0].item()
            keypoints_pos[:, 1] += init_root_offset[1].item()
            # import ipdb; ipdb.set_trace()
            keypoints_vel = avpposeinfo.avp_vel.copy()
            print(keypoints_pos)
            env._update_realtime_vr_keypoints(keypoints_pos, keypoints_vel)
        # print("obs = ", obs)
        # print("actions = ", actions)
        # print()
        # exit()
        if override: 
            obs[:,9] = 0.5
            obs[:,10] = 0.0
            obs[:,11] = 0.0
        
        # overwrite linear velocity - z and angular velocity - xy
        # obs[:, 40] = 0.
        # obs[: 41:43] = 0.
        

        if i < stop_state_log:
            logger.log_states(
                {
                    'dof_pos_target': target_joint,
                    # 'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale + env.default_dof_pos[robot_index, joint_index].item(),
                    # 'dof_pos_target': env.actions[robot_index, joint_index].item() * env.cfg.control.action_scale + env.default_dof_pos[robot_index, joint_index].item(),
                    'dof_pos': env.dof_pos[robot_index, joint_index].item(),
                    'dof_vel': env.dof_vel[robot_index, joint_index].item(),
                    'dof_torque': env.torques[robot_index, joint_index].item(),
                    'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                    'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                    'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                    # 'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                    # 'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy()
                    'MPJPE': mean_mpjpe.cpu().numpy(),
                    "x_pos": root_pos[0][0][0].item(),
                    "x_pos_ref": ref_root_pos[0][0][0].item(),
                    "y_pos": root_pos[0][0][1].item(),
                    "y_pos_ref": ref_root_pos[0][0][1].item(),
                    "z_pos": root_pos[0][0][2].item(),
                    "z_pos_ref": ref_root_pos[0][0][2].item()
                }
            )
        elif i==stop_state_log:
            logger.plot_states()
        if  0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes>0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i==stop_rew_log:
            pass
            # logger.print_rewards()
        i += 1

    if RENDER:
        video.release()    


if __name__ == '__main__':
    EXPORT_POLICY = False
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    # args = get_args()
    play()