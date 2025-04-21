# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Value

class Logger:
    def __init__(self, dt):
        self.state_log = defaultdict(list)
        self.rew_log = defaultdict(list)
        self.dt = dt
        self.num_episodes = 0
        self.plot_process = None

    def log_state(self, key, value):
        self.state_log[key].append(value)

    def log_states(self, dict):
        for key, value in dict.items():
            self.log_state(key, value)

    def log_rewards(self, dict, num_episodes):
        for key, value in dict.items():
            if 'rew' in key:
                try:
                    self.rew_log[key].append(value.item() * num_episodes)
                except:
                    ...
                    # self.rew_log[key].append(value * num_episodes)
        self.num_episodes += num_episodes

    def reset(self):
        self.state_log.clear()
        self.rew_log.clear()

    def plot_states(self):
        self.plot_process = Process(target=self._plot)
        self.plot_process.start()

    def _plot(self):
        nb_rows = 3
        nb_cols = 3
        fig, axs = plt.subplots(nb_rows, nb_cols)
        for key, value in self.state_log.items():
            time = np.linspace(0, len(value)*self.dt, len(value))
            break
        log= self.state_log

        # 新建一个画布，保存结果
        joint_pos_fig, joint_pos_ax = plt.subplots(figsize=(8, 4))

        # plot joint targets and measured positions
        a = axs[0, 0]
        if log["dof_pos"]: 
            a.plot(time, log["dof_pos"], label='measured')
            joint_pos_ax.plot(time, log["dof_pos"], label='measured')  # 同步绘制到SVG画布
        if log["dof_pos_target"]: 
            a.plot(time, log["dof_pos_target"], label='target')
            joint_pos_ax.plot(time, log["dof_pos_target"], label='target')  # 同步绘制到SVG画布
        a.set(xlabel='time [s]', ylabel='Position [rad]', title='DOF Position')
        a.legend()

        # SVG画布设置
        joint_pos_ax.set(
            xlabel='time(s)', 
            ylabel='position', 
            title='dof command',
        )
        joint_pos_ax.legend()
        # 保存SVG
        joint_pos_fig.savefig('joint.svg', 
                            format='svg', 
                            bbox_inches='tight',  # 防止标签被裁剪
                            dpi=300)
        plt.close(joint_pos_fig)  # 关闭临时画布


        # 画MPJPE
        # 新建一个画布，保存结果
        mpjpe_fig, mpjpe_ax = plt.subplots(figsize=(8, 4))

        a = axs[0, 1]
        if log["MPJPE"]!=[]: 
            a.plot(time, log["MPJPE"], label='measured')
            mpjpe_ax.plot(time, log["MPJPE"], label='measured')
        a.set(xlabel='time [s]', ylabel='MPJPE [m]', title='MPJPE')
        a.legend() 

        # SVG画布设置
        mpjpe_ax.set(
            xlabel='time(s)', 
            ylabel='MPJPE [m]', 
            title='MPJPE',
        )
        mpjpe_ax.legend()
        # 保存SVG
        mpjpe_fig.savefig('mpjpe.svg', 
                            format='svg', 
                            bbox_inches='tight',  # 防止标签被裁剪
                            dpi=300)
        plt.close(mpjpe_fig)  # 关闭临时画布


        # 画x位置
        # 新建一个画布，保存结果
        x_fig, x_ax = plt.subplots(figsize=(8, 4))

        a = axs[0, 2]
        if log["x_pos"]!=[]: 
            a.plot(time, log["x_pos"], label='x_position')
            x_ax.plot(time, log["x_pos"], label='x_position')
        if log["x_pos_ref"]!=[]: 
            a.plot(time, log["x_pos_ref"], label='x_position_ref')
            x_ax.plot(time, log["x_pos_ref"], label='x_position_ref')
        a.set(xlabel='time [s]', ylabel='x_position[m]', title='x_position')
        a.legend() 

        # SVG画布设置
        x_ax.set(
            xlabel='time(s)', 
            ylabel='x_position[m]',  
            title='x_position',
        )
        x_ax.legend()
        # 保存SVG
        x_fig.savefig('x.svg', 
                            format='svg', 
                            bbox_inches='tight',  # 防止标签被裁剪
                            dpi=300)
        plt.close(x_fig)  # 关闭临时画布


        # 画y位置
        # 新建一个画布，保存结果
        y_fig, y_ax = plt.subplots(figsize=(8, 4))

        a = axs[1, 0]
        if log["y_pos"]!=[]: 
            a.plot(time, log["y_pos"], label='y_position')
            y_ax.plot(time, log["y_pos"], label='y_position')
        if log["y_pos_ref"]!=[]: 
            a.plot(time, log["y_pos_ref"], label='y_position_ref')
            y_ax.plot(time, log["y_pos_ref"], label='y_position_ref')
        a.set(xlabel='time [s]', ylabel='y_position[m]', title='y_position')
        a.legend() 

        # SVG画布设置
        y_ax.set(
            xlabel='time(s)', 
            ylabel='y_position[m]',  
            title='y_position',
        )
        y_ax.legend()
        # 保存SVG
        y_fig.savefig('y.svg', 
                            format='svg', 
                            bbox_inches='tight',  # 防止标签被裁剪
                            dpi=300)
        plt.close(y_fig)  # 关闭临时画布



        # 画z位置
        # 新建一个画布，保存结果
        z_fig, z_ax = plt.subplots(figsize=(8, 4))

        a = axs[1, 1]
        if log["z_pos"]!=[]: 
            a.plot(time, log["z_pos"], label='z_position')
            z_ax.plot(time, log["z_pos"], label='z_position')
        if log["z_pos_ref"]!=[]: 
            a.plot(time, log["z_pos_ref"], label='z_position_ref')
            z_ax.plot(time, log["z_pos_ref"], label='z_position_ref')
        a.set(xlabel='time [s]', ylabel='z_position[m]', title='z_position')
        a.legend() 

        # SVG画布设置
        z_ax.set(
            xlabel='time(s)', 
            ylabel='z_position[m]',  
            title='z_position',
        )
        z_ax.legend()
        # 保存SVG
        z_fig.savefig('z.svg', 
                            format='svg', 
                            bbox_inches='tight',  # 防止标签被裁剪
                            dpi=300)
        plt.close(z_fig)  # 关闭临时画布

        # # plot joint velocity
        # a = axs[1, 1]
        # if log["dof_vel"]: a.plot(time, log["dof_vel"], label='measured')
        # if log["dof_vel_target"]: a.plot(time, log["dof_vel_target"], label='target')
        # a.set(xlabel='time [s]', ylabel='Velocity [rad/s]', title='Joint Velocity')
        # a.legend()

        # # plot base vel x
        # a = axs[0, 0]
        # if log["base_vel_x"]: a.plot(time, log["base_vel_x"], label='measured')
        # if log["command_x"]: a.plot(time, log["command_x"], label='commanded')
        # a.set(xlabel='time [s]', ylabel='base lin vel [m/s]', title='Base velocity x')
        # a.legend()

        # # plot base vel y
        # a = axs[0, 1]
        # if log["base_vel_y"]: a.plot(time, log["base_vel_y"], label='measured')
        # if log["command_y"]: a.plot(time, log["command_y"], label='commanded')
        # a.set(xlabel='time [s]', ylabel='base lin vel [m/s]', title='Base velocity y')
        # a.legend()

        # # plot base vel yaw
        # a = axs[0, 2]
        # if log["base_vel_yaw"]: a.plot(time, log["base_vel_yaw"], label='measured')
        # if log["command_yaw"]: a.plot(time, log["command_yaw"], label='commanded')
        # a.set(xlabel='time [s]', ylabel='base ang vel [rad/s]', title='Base velocity yaw')
        # a.legend()

        # # plot base vel z
        # a = axs[1, 2]
        # if log["base_vel_z"]: a.plot(time, log["base_vel_z"], label='measured')
        # a.set(xlabel='time [s]', ylabel='base lin vel [m/s]', title='Base velocity z')
        # a.legend()

        # # plot contact forces
        # a = axs[2, 0]
        # if log["contact_forces_z"]:
        #     forces = np.array(log["contact_forces_z"])
        #     for i in range(forces.shape[1]):
        #         a.plot(time, forces[:, i], label=f'force {i}')
        # a.set(xlabel='time [s]', ylabel='Forces z [N]', title='Vertical Contact forces')
        # a.legend()

        # # plot torque/vel curves
        # a = axs[2, 1]
        # if log["dof_vel"]!=[] and log["dof_torque"]!=[]: a.plot(log["dof_vel"], log["dof_torque"], 'x', label='measured')
        # a.set(xlabel='Joint vel [rad/s]', ylabel='Joint Torque [Nm]', title='Torque/velocity curves')
        # a.legend()

        # plot torques
        # a = axs[2, 2]
        # if log["dof_torque"]!=[]: a.plot(time, log["dof_torque"], label='measured')
        # a.set(xlabel='time [s]', ylabel='Joint Torque [Nm]', title='Torque')
        # a.legend()

        plt.show()

    def print_rewards(self):
        print("Average rewards per second:")
        for key, values in self.rew_log.items():
            mean = np.sum(np.array(values)) / self.num_episodes
            print(f" - {key}: {mean}")
        print(f"Total number of episodes: {self.num_episodes}")
    
    def __del__(self):
        if self.plot_process is not None:
            self.plot_process.kill()