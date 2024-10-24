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

import torch
import torch.nn as nn
from torch.distributions import Normal
from legged_gym.algo.models.decision_transformer import DecisionTransformer
from legged_gym.algo.models.model import PPOTransformerModel


# a BERT-style transformer block
class Transformer_Block(nn.Module):
    def __init__(self, latent_dim, num_head, dropout_rate) -> None:
        super().__init__()
        self.num_head = num_head
        self.latent_dim = latent_dim
        self.ln_1 = nn.LayerNorm(latent_dim)
        self.attn = nn.MultiheadAttention(latent_dim, num_head, dropout=dropout_rate, batch_first=True)
        self.ln_2 = nn.LayerNorm(latent_dim)
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, 4 * latent_dim),
            nn.GELU(),
            nn.Linear(4 * latent_dim, latent_dim),
            nn.Dropout(dropout_rate),
        )
    
    def forward(self, x):
        x = self.ln_1(x)
        x = x + self.attn(x, x, x, need_weights=False)[0]
        x = self.ln_2(x)
        x = x + self.mlp(x)
        
        return x

class Transformer(nn.Module):
    def __init__(self, input_dim, output_dim, context_len, latent_dim=128, num_head=4, num_layer=4, dropout_rate=0.1) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.context_len = context_len
        self.latent_dim = latent_dim
        self.num_head = num_head
        self.num_layer = num_layer
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.Dropout(dropout_rate),
        )
        self.weight_pos_embed = nn.Embedding(context_len, latent_dim)
        self.attention_blocks = nn.Sequential(
            *[Transformer_Block(latent_dim, num_head, dropout_rate) for _ in range(num_layer)],
        )
        self.output_layer = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, output_dim),
        )
    
    def forward(self, x):
        x = self.input_layer(x)
        x = x + self.weight_pos_embed(torch.arange(x.shape[1], device=x.device))
        x = self.attention_blocks(x)

        # take the last token
        x = x[:, -1, :]
        x = self.output_layer(x)

        return x

class ActorCritic(nn.Module):
    is_recurrent = False
    
    def __init__(self,  num_actor_obs,
                        num_critic_obs,
                        num_actions,
                        obs_context_len = None,
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        init_noise_std=1.0,
                        architecture='MLP',
                        activation = 'elu',
                        **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(ActorCritic, self).__init__()

        activation = get_activation(activation)
        mlp_input_dim_a = num_actor_obs
        mlp_input_dim_c = num_critic_obs
        self.obs_context_len = obs_context_len
        # Police
        if architecture == 'Trans' or architecture == 'Mix':
            config = {"transformer":{ 
                "num_blocks":2,
                "num_heads":6,
                "gru_bias":0.0,
                "hidden_size":128,
                "embed_dim": 384,
                "memory_length":15,
            }}
            self.actor = PPOTransformerModel(config, mlp_input_dim_a, num_actions)
            #self.actor = DecisionTransformer(mlp_input_dim_a, num_actions, hidden_size = 192)
        else:
            actor_layers = []
            actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
            actor_layers.append(activation)
            for l in range(len(actor_hidden_dims)):
                if l == len(actor_hidden_dims) - 1:
                    actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
                else:
                    actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                    actor_layers.append(activation)
            self.actor = nn.Sequential(*actor_layers)
        


        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        #print(critic_layers)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor T: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        self.teaching_distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False
        

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]


    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        mean = self.actor(observations, self.obs_context_len)
        mean = mean#[:,:10]
        std = self.std#[:10]
        return Normal(mean, mean*0. + std)

    def act(self, observations, **kwargs):
        self.distribution = self.update_distribution(observations)
        return self.distribution.sample()  

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        actions_mean = self.actor(observations, self.obs_context_len)
        actions_mean = actions_mean#[:,:10]
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        num_batch = critic_observations.shape[0]
        observations_tmp = critic_observations.view(num_batch, self.obs_context_len, -1)
        observations_tmp = observations_tmp[:,-3:,:]
        observations_tmp = observations_tmp.contiguous().view(num_batch, -1)
        value = self.critic(observations_tmp)
        #value = self.critic(critic_observations)
        return value

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.CReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None



class TActorCritic(nn.Module):
    is_recurrent = False
    def __init__(self,  num_actor_obs,
                        num_critic_obs,
                        num_actions,
                        obs_context_len = None,
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        activation='elu',
                        init_noise_std=1.0,
                        **kwargs):
        if kwargs:
            print("TActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(TActorCritic, self).__init__()

        activation = get_activation(activation)
        self.obs_context_len = obs_context_len

        mlp_input_dim_a = num_actor_obs
        mlp_input_dim_c = num_critic_obs

        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False
        
        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]


    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):

        num_batch = observations.shape[0]
        observations_tmp = observations.view(num_batch, self.obs_context_len, -1)
        observations_tmp = torch.concat([observations_tmp[:,:,:6],observations_tmp[:,:,8:21],observations_tmp[:,:,30:40],observations_tmp[:,:,49:59] ], dim=-1)
        observations_tmp = observations_tmp[:,-3:,:]
        observations_tmp = observations_tmp.contiguous().view(num_batch, -1)
        mean = self.actor(observations_tmp)
        self.distribution = Normal(mean, mean*0. + self.std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        actions_mean = self.actor(observations)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        num_batch = critic_observations.shape[0]
        observations_tmp = critic_observations.view(num_batch, self.obs_context_len, -1)
        observations_tmp = torch.concat([observations_tmp[:,:,:9],observations_tmp[:,:,11:24],observations_tmp[:,:,33:43],observations_tmp[:,:,52:62] ], dim=-1)
        observations_tmp = observations_tmp[:,-3:,:]
        observations_tmp = observations_tmp.contiguous().view(num_batch, -1)
        value = self.critic(observations_tmp)
        return value

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None




class HumanPlus_ActorCritic(nn.Module):
    is_recurrent = False
    
    def __init__(self,  num_actor_obs,
                        num_critic_obs,
                        num_actions,
                        obs_context_len,
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        init_noise_std=1.0,
                        architecture='MLP',
                        activation = 'elu',
                        **kwargs):
        if kwargs:
            print("HumanPlus_ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(HumanPlus_ActorCritic, self).__init__()

        activation = get_activation(activation)
        mlp_input_dim_a = num_actor_obs
        mlp_input_dim_c = num_actor_obs
        self.obs_context_len = obs_context_len
        # Police
        # Policy
        self.actor = Transformer(num_actor_obs, num_actions, obs_context_len)
        self.actor.output_layer[1].weight.data *= 0.01 # init last layer to be 100x smaller

        # Value function
        self.critic = Transformer(num_actor_obs, 1, obs_context_len)

        print(f"Actor T: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        self.teaching_distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False
        

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]


    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):

        batch_size, seq_length = observations.shape[0], self.obs_context_len
        observations = observations.view(batch_size, seq_length, -1)

        mean = self.actor(observations)
        #mean = mean[:,:10]
        return Normal(mean, mean*0. + self.std)

    def act(self, observations, **kwargs):

        self.distribution = self.update_distribution(observations)
        return self.distribution.sample()  

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):

        batch_size, seq_length = observations.shape[0], self.obs_context_len
        observations = observations.view(batch_size, seq_length, -1)
        actions_mean = self.actor(observations)
        #actions_mean = actions_mean[:,:10]
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        batch_size, seq_length = critic_observations.shape[0], self.obs_context_len
        critic_observations = critic_observations.view(batch_size, seq_length, -1)
        value = self.critic(critic_observations)
        return value

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.CReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None

class Teaching_ActorCritic(nn.Module):
    def __init__(self,  #num_actor_obs,
                        num_teaching_actor_obs,
                        num_critic_obs,
                        num_actions,
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        init_noise_std=1.0,
                        architecture='Mix',
                        activation = nn.ELU(),
                        **kwargs):
        if kwargs:
            print("Teaching_ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(Teaching_ActorCritic, self).__init__()
        #print(kwargs.keys())

        #self.num_actor_obs = num_actor_obs
        mlp_input_dim_a = num_teaching_actor_obs
        mlp_input_dim_c = num_critic_obs
        # Police

        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)
        
        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False
        

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]


    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        mean = self.actor(observations)
        return Normal(mean, mean*0. + self.std)

    def act(self, observations, **kwargs):
        # Input of teaching policy has one dim more than transformer-based policy, which is the phase info. 
        num_batch = observations.shape[0]
        observations_tmp = observations.view(num_batch, -1, self.num_actor_obs)
        observations_tmp = observations_tmp[:,:,1:]
        observations_tmp = observations_tmp.contiguous().view(num_batch, -1)
        self.distribution = self.update_distribution(observations_tmp)
        return self.distribution.sample()  

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        actions_mean = self.actor(observations)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value
