from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from config import RL


class GaussianActor(nn.Module):
    def __init__(
        self, obs_dim: int, act_dim: int, hidden_dim: int = 256, log_std_init: float = -0.5
    ):
        super().__init__()
        self.mean = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, act_dim),
            nn.Tanh(),
        )
        self.log_std = nn.Parameter(torch.ones(act_dim) * log_std_init)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mean(obs)

    def dist(self, obs: torch.Tensor) -> Normal:
        mean = self.mean(obs)
        std = self.log_std.exp().expand_as(mean)
        return Normal(mean, std)


class CentralizedValueCritic(nn.Module):

    def __init__(self, total_obs_dim: int, num_agents: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(total_obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_agents),
        )

    def forward(self, all_obs: torch.Tensor) -> torch.Tensor:
        return self.net(all_obs)


class RolloutBuffer:
    def __init__(self, M: int, obs_dim: int, act_dim: int):
        self.M, self.obs_dim, self.act_dim = M, obs_dim, act_dim
        self.clear()

    def clear(self) -> None:
        self.obs, self.act, self.logp = [], [], []
        self.val, self.rew, self.done = [], [], []
        self.alive, self.next_alive = [], []

    def add(self, obs, act, logp, val, rew, done, alive=None, next_alive=None) -> None:
        self.obs.append(obs)
        self.act.append(act)
        self.logp.append(logp)
        self.val.append(val)
        self.rew.append(rew)
        self.done.append(done)
        self.alive.append(
            np.ones(self.M, dtype=np.float32)
            if alive is None
            else np.asarray(alive, dtype=np.float32)
        )
        self.next_alive.append(
            np.ones(self.M, dtype=np.float32)
            if next_alive is None
            else np.asarray(next_alive, dtype=np.float32)
        )

    def __len__(self) -> int:
        return len(self.obs)


class MAPPO:
    def __init__(
        self,
        num_agents: int,
        obs_dim: int,
        act_dim: int,
        device: str | torch.device = "cpu",
        clip: float = 0.2,
        ppo_epochs: int = 10,
        num_minibatches: int = 4,
        gae_lambda: float = 0.95,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        lr: float = 3e-4,
        log_std_init: float = -0.5,
        fixed_action_values: dict[int, float] | None = None,
        environment_action_dim: int | None = None,
        frequency_rule: str = "learned",
    ):
        self.M = num_agents
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.environment_action_dim = int(
            act_dim if environment_action_dim is None else environment_action_dim
        )
        self.frequency_rule = str(frequency_rule)
        self.device = torch.device(device)

        self.gamma = float(RL["gamma"])
        self.hidden_dim = int(RL["hidden_dim"])
        self.clip = clip
        self.ppo_epochs = ppo_epochs
        self.num_minibatches = num_minibatches
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm

        self.fixed_action_values = {
            int(index): float(value) for index, value in (fixed_action_values or {}).items()
        }
        invalid = [
            index
            for index, value in self.fixed_action_values.items()
            if index < 0 or index >= act_dim or not -1.0 <= value <= 1.0
        ]
        if invalid:
            raise ValueError(f"invalid fixed action dimensions: {invalid}")
        self.learned_action_dims = tuple(
            index for index in range(act_dim) if index not in self.fixed_action_values
        )
        if not self.learned_action_dims:
            raise ValueError("MAPPO requires at least one learned action dimension")

        self.actor = GaussianActor(obs_dim, act_dim, self.hidden_dim, log_std_init).to(self.device)
        self.critic = CentralizedValueCritic(num_agents * obs_dim, num_agents, self.hidden_dim).to(
            self.device
        )
        self.opt = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr
        )

        self.buffer = RolloutBuffer(num_agents, obs_dim, act_dim)

    def _apply_fixed_actions(self, actions: torch.Tensor) -> torch.Tensor:
        if not self.fixed_action_values:
            return actions
        actions = actions.clone()
        for index, value in self.fixed_action_values.items():
            actions[..., index] = value
        return actions

    def _learned_log_prob(self, dist: Normal, actions: torch.Tensor) -> torch.Tensor:
        return dist.log_prob(actions)[..., self.learned_action_dims].sum(-1)

    def _learned_entropy(self, dist: Normal) -> torch.Tensor:
        return dist.entropy()[..., self.learned_action_dims].sum(-1)

    def set_noise(self, progress: float) -> None:
        pass

    @torch.no_grad()
    def act(self, obs_list) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

        o = torch.as_tensor(np.asarray(obs_list, dtype=np.float32), device=self.device)
        dist = self.actor.dist(o)
        a = self._apply_fixed_actions(dist.sample())
        logp = self._learned_log_prob(dist, a)
        v = self.critic(o.reshape(1, -1)).squeeze(0)
        return (
            a.cpu().numpy().astype(np.float32),
            logp.cpu().numpy().astype(np.float32),
            v.cpu().numpy().astype(np.float32),
        )

    @torch.no_grad()
    def select_actions(self, obs_list, noise: bool = True) -> np.ndarray:

        o = torch.as_tensor(np.asarray(obs_list, dtype=np.float32), device=self.device)
        a = self.actor.dist(o).sample() if noise else self.actor(o)
        a = self._apply_fixed_actions(a)
        return np.clip(a.cpu().numpy(), -1.0, 1.0).astype(np.float32)

    def _last_value(self, obs_list) -> np.ndarray:
        with torch.no_grad():
            o = torch.as_tensor(np.asarray(obs_list, dtype=np.float32), device=self.device)
            return self.critic(o.reshape(1, -1)).squeeze(0).cpu().numpy().astype(np.float32)

    def _compute_gae(self, rew, val, done, last_val, next_alive=None):

        T = rew.shape[0]
        if next_alive is None:
            next_alive = np.ones_like(rew, dtype=np.float32)
        adv = np.zeros_like(rew)
        lastgae = np.zeros(self.M, dtype=np.float32)
        next_val = last_val
        for t in reversed(range(T)):
            nonterminal = (1.0 - done[t]) * next_alive[t]
            delta = rew[t] + self.gamma * next_val * nonterminal - val[t]
            lastgae = delta + self.gamma * self.gae_lambda * nonterminal * lastgae
            adv[t] = lastgae
            next_val = val[t]
        ret = adv + val
        return adv, ret

    def update(self, last_obs) -> dict | None:
        if len(self.buffer) == 0:
            return None
        b = self.buffer
        obs = np.asarray(b.obs, dtype=np.float32)
        act = np.asarray(b.act, dtype=np.float32)
        logp_old = np.asarray(b.logp, dtype=np.float32)
        val = np.asarray(b.val, dtype=np.float32)
        rew = np.asarray(b.rew, dtype=np.float32)
        done = np.asarray(b.done, dtype=np.float32)
        alive = np.asarray(b.alive, dtype=np.float32)
        next_alive = np.asarray(b.next_alive, dtype=np.float32)
        last_val = self._last_value(last_obs)

        adv, ret = self._compute_gae(
            rew,
            val,
            done,
            last_val,
            next_alive=next_alive,
        )
        T = obs.shape[0]

        all_obs = obs.reshape(T, -1)
        all_obs_rep = np.repeat(all_obs, self.M, axis=0)
        agent_idx = np.tile(np.arange(self.M), T)
        flat_obs = obs.reshape(T * self.M, self.obs_dim)
        flat_act = act.reshape(T * self.M, self.act_dim)
        flat_logp_old = logp_old.reshape(T * self.M)
        flat_adv = adv.reshape(T * self.M)
        flat_ret = ret.reshape(T * self.M)
        valid_idx = np.flatnonzero(alive.reshape(T * self.M) > 0.5)
        if valid_idx.size == 0:
            self.buffer.clear()
            return None

        all_obs_rep = all_obs_rep[valid_idx]
        agent_idx = agent_idx[valid_idx]
        flat_obs = flat_obs[valid_idx]
        flat_act = flat_act[valid_idx]
        flat_logp_old = flat_logp_old[valid_idx]
        flat_adv = flat_adv[valid_idx]
        flat_ret = flat_ret[valid_idx]

        flat_adv = (flat_adv - flat_adv.mean()) / (flat_adv.std() + 1e-8)

        dev = self.device
        t_obs = torch.as_tensor(flat_obs, device=dev)
        t_act = torch.as_tensor(flat_act, device=dev)
        t_logp_old = torch.as_tensor(flat_logp_old, device=dev)
        t_adv = torch.as_tensor(flat_adv, device=dev)
        t_ret = torch.as_tensor(flat_ret, device=dev)
        t_all_obs = torch.as_tensor(all_obs_rep, device=dev)
        t_aidx = torch.as_tensor(agent_idx, device=dev, dtype=torch.long)

        N = int(valid_idx.size)
        mb = max(1, N // self.num_minibatches)
        pi_losses, v_losses, ent_losses = [], [], []
        for _ in range(self.ppo_epochs):
            perm = torch.randperm(N, device=dev)
            for s in range(0, N, mb):
                idx = perm[s : s + mb]
                dist = self.actor.dist(t_obs[idx])
                logp = self._learned_log_prob(dist, t_act[idx])
                entropy = self._learned_entropy(dist).mean()

                ratio = torch.exp(logp - t_logp_old[idx])
                a_mb = t_adv[idx]
                surr1 = ratio * a_mb
                surr2 = torch.clamp(ratio, 1.0 - self.clip, 1.0 + self.clip) * a_mb
                pi_loss = -torch.min(surr1, surr2).mean()

                v_all = self.critic(t_all_obs[idx])
                v_pred = v_all.gather(1, t_aidx[idx].unsqueeze(1)).squeeze(1)
                v_loss = F.mse_loss(v_pred, t_ret[idx])

                loss = pi_loss + self.value_coef * v_loss - self.entropy_coef * entropy
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.max_grad_norm,
                )
                self.opt.step()

                pi_losses.append(pi_loss.item())
                v_losses.append(v_loss.item())
                ent_losses.append(entropy.item())

        self.buffer.clear()
        return {
            "actor_loss": float(np.mean(pi_losses)),
            "critic_loss": float(np.mean(v_losses)),
            "entropy": float(np.mean(ent_losses)),
            "noise_sigma": float(
                self.actor.log_std.exp()[list(self.learned_action_dims)].mean().item()
            ),
            "valid_agent_samples": N,
        }

    def save(self, path: str) -> None:
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "metadata": {
                    "policy_action_dim": int(self.act_dim),
                    "environment_action_dim": int(self.environment_action_dim),
                    "frequency_rule": self.frequency_rule,
                },
            },
            path,
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        metadata = ckpt.get("metadata")
        if metadata is not None:
            expected = {
                "policy_action_dim": int(self.act_dim),
                "environment_action_dim": int(self.environment_action_dim),
                "frequency_rule": self.frequency_rule,
            }
            mismatches = [
                f"{key}: checkpoint={metadata.get(key)!r}, expected={value!r}"
                for key, value in expected.items()
                if metadata.get(key) != value
            ]
            if mismatches:
                raise RuntimeError(
                    "MAPPO checkpoint control-interface mismatch:\n  - " + "\n  - ".join(mismatches)
                )
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
