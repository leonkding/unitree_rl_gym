# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2021 ETH Zurich, Nikita Rudin
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
# Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.

import os
import time
import torch
import wandb
import statistics
import cv2
from collections import deque
from datetime import datetime
from .ppo import PPO
from .actor_critic import ActorCritic, Teaching_ActorCritic, HumanPlus_ActorCritic, TActorCritic
from .actor_critic_recurrent import ActorCriticRecurrent
from legged_gym.algo.vec_env import VecEnv
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from legged_gym.envs import *
from legged_gym import LEGGED_GYM_ROOT_DIR
from isaacgym.torch_utils import *
from isaacgym import gymapi

class OnPolicyRunner:

    def __init__(self, env: VecEnv, train_cfg, log_dir=None, device="cpu"):

        self.cfg = train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.all_cfg = train_cfg
        self.wandb_run_name = (
            datetime.now().strftime("%b%d_%H-%M-%S")
            + "_"
            + train_cfg["runner"]["experiment_name"]
            + "_"
            + train_cfg["runner"]["run_name"]
        )
        self.device = device
        self.env = env
        if self.env.num_privileged_obs is not None:
            num_critic_obs = self.env.num_privileged_obs
        else:
            num_critic_obs = self.env.num_obs
        
        if self.policy_cfg["architecture"] == 'RNN':
            actor_critic_class = eval('ActorCriticRecurrent')  # ActorCritic
            actor_critic: ActorCriticRecurrent = actor_critic_class(
                self.env.num_obs, num_critic_obs, self.env.num_actions, **self.policy_cfg
            ).to(self.device)
        else:
            actor_critic_class = eval('ActorCritic')  # ActorCritic
            #print(self.env.c_frame_stack)
            actor_critic: ActorCritic = actor_critic_class(
                87, 90*3, self.env.num_actions, self.env.frame_stack, **self.policy_cfg
            ).to(self.device)

        if self.policy_cfg["architecture"] == 'Mix':
            self.teaching_actorcritic = ActorCritic(87, 90*3, self.env.num_actions, self.env.frame_stack, **self.policy_cfg).to(self.device)
            #TActorCritic(39*3, 42*3, self.env.num_actions,self.env.frame_stack, **self.policy_cfg)
            print('Loading Pretrained Teaching Model')
            self.teaching_actorcritic.load_state_dict(torch.load(self.policy_cfg["teaching_model_path"], map_location='cuda:0')["model_state_dict"])
            print('Pretrained Teaching Model Loaded')
        else:
            self.teaching_actorcritic = None

        if self.policy_cfg['policy_type'] == "standing":
            actor_critic_class = eval('ActorCritic') 
            self.moving_actorcritic: ActorCritic = actor_critic_class(
                    self.env.num_obs, num_critic_obs, self.env.num_actions, **self.policy_cfg
                ).to(self.device)
            self.moving_actorcritic.load_state_dict(torch.load(self.policy_cfg["moving_model_path"], map_location='cuda:0')["model_state_dict"])
        else:
            self.moving_actorcritic = None

        alg_class = eval(self.cfg["algorithm_class_name"])  # PPO
        self.alg: PPO = alg_class(actor_critic, self.teaching_actorcritic, device=self.device, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # init storage and model
        if self.policy_cfg['architecture'] == 'Mix' or self.policy_cfg['architecture'] == 'Trans':
            self.alg.init_storage(
                self.env.num_envs,
                self.num_steps_per_env,
                [int(self.env.num_obs)],
                [self.env.num_privileged_obs],
                [self.env.num_actions],
            )
        else: 
            self.alg.init_storage(
                self.env.num_envs,
                self.num_steps_per_env,
                [int(self.env.num_obs)],
                [self.env.num_privileged_obs],
                [self.env.num_actions],
            )
        self.num_steps_warmup = 25

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        _, _ = self.env.reset()

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # initialize writer

    #if RENDER:
        camera_properties = gymapi.CameraProperties()
        camera_properties.width = 1920
        camera_properties.height = 1080
        h1 = self.env.gym.create_camera_sensor(self.env.envs[0], camera_properties)
        camera_offset = gymapi.Vec3(1, -1, 0.5)
        camera_rotation = gymapi.Quat.from_axis_angle(gymapi.Vec3(-0.3, 0.2, 1),
                                                    np.deg2rad(135))
        actor_handle = self.env.gym.get_actor_handle(self.env.envs[0], 0)
        body_handle = self.env.gym.get_actor_rigid_body_handle(self.env.envs[0], actor_handle, 0)
        self.env.gym.attach_camera_to_body(
            h1, self.env.envs[0], body_handle,
            gymapi.Transform(camera_offset, camera_rotation),
            gymapi.FOLLOW_POSITION)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'videos')
        experiment_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'videos', self.cfg["experiment_name"])
        dir = os.path.join(experiment_dir, datetime.now().strftime('%b%d_%H-%M-%S')+ self.cfg["run_name"] + '.mp4')
        if not os.path.exists(video_dir):
            os.mkdir(video_dir)
        if not os.path.exists(experiment_dir):
            os.mkdir(experiment_dir)
        video = cv2.VideoWriter(dir, fourcc, 50.0, (1920, 1080))


        if self.log_dir is not None and self.writer is None:
            wandb.init(
                project="WBC_loc",
                sync_tensorboard=True,
                name=self.wandb_run_name,
                config=self.all_cfg,
            )
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        #critic_obs = obs
        self.alg.actor_critic.train()  # switch to train mode (for dropout for example)
        #self.teaching_actorcritic.test()

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(
            self.env.num_envs, dtype=torch.float, device=self.device
        )
        cur_episode_length = torch.zeros(
            self.env.num_envs, dtype=torch.float, device=self.device
        )

        tot_iter = self.current_learning_iteration + num_learning_iterations

        camera_properties = gymapi.CameraProperties()
        camera_properties.width = 1920
        camera_properties.height = 1080
        h1 = self.env.gym.create_camera_sensor(self.env.envs[0], camera_properties)
        camera_offset = gymapi.Vec3(1, -1, 0.5)
        camera_rotation = gymapi.Quat.from_axis_angle(gymapi.Vec3(-0.3, 0.2, 1),
                                                    np.deg2rad(135))
        actor_handle = self.env.gym.get_actor_handle(self.env.envs[0], 0)
        body_handle = self.env.gym.get_actor_rigid_body_handle(self.env.envs[0], actor_handle, 0)
        self.env.gym.attach_camera_to_body(
                    h1, self.env.envs[0], body_handle,
                    gymapi.Transform(camera_offset, camera_rotation),
                    gymapi.FOLLOW_POSITION)


        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()

            if self.cfg["render"] and it % int(self.save_interval/2) == 0:


                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'videos')
                experiment_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'videos', self.cfg["experiment_name"])
                dir = os.path.join(experiment_dir, datetime.now().strftime('%b%d_%H-%M-%S')+ self.cfg["run_name"] + '.mp4')
                if not os.path.exists(video_dir):
                    os.mkdir(video_dir)
                if not os.path.exists(experiment_dir):
                    os.mkdir(experiment_dir)
                video = cv2.VideoWriter(dir, fourcc, 50.0, (1920, 1080))


            # Rollout
            with torch.inference_mode():
                #print(self.policy_cfg["policy_type"])
                if self.policy_cfg['policy_type'] == "standing":
                    for i in range(self.num_steps_warmup):
                        actions = self.moving_actorcritic.act(obs).detach()
                        obs, privileged_obs, rewards, dones, infos = self.env.step(actions)#, "moving") 
                        critic_obs = privileged_obs if privileged_obs is not None else obs
                                 
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs)
                    obs, privileged_obs, rewards, dones, infos = self.env.step(actions)#, self.policy_cfg['policy_type'])
                    critic_obs = privileged_obs if privileged_obs is not None else obs

                    obs, critic_obs, rewards, dones = (
                        obs.to(self.device),
                        critic_obs.to(self.device),
                        rewards.to(self.device),
                        dones.to(self.device),
                    )
                    self.alg.process_env_step(rewards, dones, infos)

                    #print(self.cfg["render"])
                    #print(it % int(self.save_interval/2)==0)
                    # if self.cfg["render"] and it % int(self.save_interval/2) == 0:
                    #     print('eee1')
                    #     self.env.gym.fetch_results(self.env.sim, True)
                    #     print('eee2')
                    #     self.env.gym.step_graphics(self.env.sim)
                    #     print('eee3')
                    #     #self.env.gym.render_all_camera_sensors(self.env.sim)
                    #     img = self.env.gym.get_camera_image(self.env.sim, self.env.envs[0], h1, gymapi.IMAGE_COLOR)
                    #     img = np.reshape(img, (1080, 1920, 4))
                    #     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    #     print('eee')
                    #     print(i)
                    #     #cv2.imwrite('/home/ziluoding/unitree_rl_gym/'+str(i)+'.jpg', img)
                    #     video.write(img[..., :3])
                    #     print('ooo')

                    if self.log_dir is not None:
                        # Book keeping
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(
                            cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist()
                        )
                        lenbuffer.extend(
                            cur_episode_length[new_ids][:, 0].cpu().numpy().tolist()
                        )
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0


                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs)

            if self.cfg["render"] and it % int(self.save_interval/2) == 0:
                video.release()

            mean_value_loss, mean_surrogate_loss, mean_imitation_loss, mean_supervised_loss = self.alg.update()
            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, "model_{}.pt".format(it)))
            ep_infos.clear()

        self.current_learning_iteration += num_learning_iterations
        self.save(
            os.path.join(
                self.log_dir, "model_{}.pt".format(self.current_learning_iteration)
            )
        )


    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        ep_string = f""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                #print('uuuuuu')
                #print(key)
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    #print(ep_info)
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar("Episode/" + key, value, locs["it"])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(
            self.num_steps_per_env
            * self.env.num_envs
            / (locs["collection_time"] + locs["learn_time"])
        )

        self.writer.add_scalar(
            "Loss/value_function", locs["mean_value_loss"], locs["it"]
        )
        self.writer.add_scalar(
            "Loss/surrogate", locs["mean_surrogate_loss"], locs["it"]
        )
        self.writer.add_scalar(
            "Loss/imitation", locs["mean_imitation_loss"], locs["it"]
        )
        self.writer.add_scalar(
            "Loss/supervised", locs["mean_supervised_loss"], locs["it"]
        )
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar(
            "Perf/collection time", locs["collection_time"], locs["it"]
        )
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])
        if len(locs["rewbuffer"]) > 0:
            self.writer.add_scalar(
                "Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"]
            )
            self.writer.add_scalar(
                "Train/mean_episode_length",
                statistics.mean(locs["lenbuffer"]),
                locs["it"],
            )
            self.writer.add_scalar(
                "Train/mean_reward/time",
                statistics.mean(locs["rewbuffer"]),
                self.tot_time,
            )
            self.writer.add_scalar(
                "Train/mean_episode_length/time",
                statistics.mean(locs["lenbuffer"]),
                self.tot_time,
            )


        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'Imitation loss:':>{pad}} {locs['mean_imitation_loss']:.4f}\n"""
                f"""{'Supervised loss:':>{pad}} {locs['mean_supervised_loss']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
            )
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'Imitation loss:':>{pad}} {locs['mean_imitation_loss']:.4f}\n"""
                f"""{'Supervised loss:':>{pad}} {locs['mean_supervised_loss']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
            f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n"""
        )
        print(log_string)

    def save(self, path, infos=None):
        torch.save(
            {
                "model_state_dict": self.alg.actor_critic.state_dict(),
                "optimizer_state_dict": self.alg.optimizer.state_dict(),
                "iter": self.current_learning_iteration,
                "infos": infos,
            },
            path,
        )

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path, map_location='cuda:7')
        self.alg.actor_critic.load_state_dict(loaded_dict["model_state_dict"])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference

    def get_inference_critic(self, device=None):
        self.alg.actor_critic.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.evaluate
