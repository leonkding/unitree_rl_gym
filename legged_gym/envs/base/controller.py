import dataclasses
import os
import re
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch


@dataclasses.dataclass
class Control:
    torque: torch.Tensor  # (num_envs, control_dim) the current torque
    buffer: (
        torch.Tensor
    )  # (num_envs, buffer_len, control_dim) the buffer of past targets

    def push(self, action: torch.Tensor):
        self.buffer = torch.cat((action[:, None, :], self.buffer[:, :-1]), dim=1)

    @property
    def prev_action(self):
        return self.buffer[:, 1]

    @property
    def action(self):
        return self.buffer[:, 0]

    @property
    def ctrl_dim(self) -> int:
        return self.buffer.shape[-1]


class PDController:
    def __init__(
        self,
        control_dim: int,
        device: str,
        torque_limit: torch.Tensor,
        kp: torch.Tensor,
        kd: torch.Tensor,
        num_envs: int,
        seed: int = 0,
        decimation_count: Union[int, Tuple[int, int]] = (3, 5),
        scale: Optional[torch.Tensor] = None,
        offset: Optional[torch.Tensor] = None,
    ):
        self.scale = (
            torch.ones((1, control_dim), device=device)
            if scale is None
            else scale.to(device)
        )
        self.offset = (
            torch.zeros((1, control_dim), device=device)
            if offset is None
            else offset.to(device)
        )
        self.np_random = np.random.RandomState(seed)
        self.torque_limit = torque_limit.to(device)
        self.device = device
        self.decimation_count_range = (
            decimation_count
            if not isinstance(decimation_count, int)
            else (decimation_count, decimation_count)
        )
        self.control_dim = control_dim
        self.kp = kp.to(self.device)[None, :].float()
        self.kd = kd.to(self.device)[None, :].float()
        self.num_envs = num_envs
        self.prev_normalized_target = torch.zeros((1, control_dim), device=self.device)

    @property
    def decimation_count(self) -> int:
        return int(
            self.np_random.randint(
                self.decimation_count_range[0],
                self.decimation_count_range[1] + 1,
            )
        )

    def __call__(
        self,
        action: torch.Tensor,
    ):
        """
        action: (num_envs, control_dim) from the network
        """
        return self.compute_torque(
            normalized_action=action * self.scale
            + self.offset.repeat(action.shape[0], 1),
        )

    def compute_torque(self, normalized_action: torch.Tensor):
        """
        normalized_action: (control_dim, ) after __call__
        """
        self.prev_normalized_target = normalized_action
        return torch.clip(
            normalized_action, min=-self.torque_limit, max=self.torque_limit
        )


class VelocityController(PDController):
    def __init__(self, **kwargs):
        kwargs["require_prev_state"] = True
        super().__init__(**kwargs)

    def compute_torque(
        self,
        normalized_action: torch.Tensor,
    ):
        dt = state.sim_dt

        curr_vel = state.dof_vel.clone()
        prev_vel = state.dof_vel.clone()
        torques = (
            self.kp * (normalized_action - curr_vel)
            - self.kd * (curr_vel - prev_vel) / dt
        )
        return super().compute_torque(torques, state)


class PositionController(PDController):
    def compute_torque(
        self,
        normalized_action: torch.Tensor,
    ):
        curr_pos = state.dof_pos.clone()
        curr_vel = state.dof_vel.clone()
        assert normalized_action.shape == curr_pos.shape
        assert curr_vel.shape == curr_pos.shape
        if normalized_action.shape[0] != self.kp.shape[0]:
            self.kp = self.kp.repeat(normalized_action.shape[0], 1)
            self.kd = self.kd.repeat(normalized_action.shape[0], 1)
        torques = self.kp * (normalized_action - curr_pos) - self.kd * curr_vel
        return super().compute_torque(torques, state)