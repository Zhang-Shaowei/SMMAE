import copy

import numpy as np
import torch
import torch as th
import torch.nn as nn
from components.episode_buffer import EpisodeBatch
from intrinsic.coop_pred import CoopPred
from modules.mixers.qmix import QMixer
from modules.mixers.vdn import VDNMixer
from torch.optim import Adam


class SMMAEQLearner:
    def __init__(self, mac, scheme, exp_scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.params = list(mac.agent_parameters())
        self.last_target_update_episode = 0

        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        self.optimiser = Adam(params=self.params, lr=args.lr)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.training_steps = 0
        self.last_target_update_step = 0
        self.log_stats_t = -self.args.learner_log_interval - 1


        self.intrinsic = None
        self.intrinsic_optimiser = None
        self.CELoss = None
        if self.args.use_adaptive_alpha:
            self.intrinsic = CoopPred(args, scheme)
            self.intrinsic_optimiser = Adam(params=self.intrinsic.parameters(), lr=args.pred_lr)
            self.CELoss = nn.CrossEntropyLoss(reduction='none')


        self.explore_optimiser = None
        self.explore_optimiser = Adam(params=mac.explore_parameters(), lr=args.lr)

        self.density_model = None

    def set_up_density(self, density_model):
        self.density_model = density_model

    def _get_filled_obs(self, batch, mask, batch_to_update_flag):
        # will dismiss the last obs(different from before)
        obs = batch['obs'][:, :-1]
        res = th.zeros((obs.shape[0] * obs.shape[1], obs.shape[2], obs.shape[3]))
        useful_num = 0
        for i in range(obs.shape[0]):
            if batch_to_update_flag[i]:
                cur_batch_useful_num = int(mask[i].sum())
                res[useful_num: useful_num + cur_batch_useful_num] = obs[i, 0: cur_batch_useful_num]
                useful_num += cur_batch_useful_num
        return res[:useful_num]

    def _get_filled_next_obs(self, batch, mask, batch_to_update_flag):  # use o_{t+1} to calculate density
        # will dismiss the last obs(different from before)
        next_obs = batch['obs'][:, 1:]   # use next obs
        res = th.zeros((next_obs.shape[0] * next_obs.shape[1], next_obs.shape[2], next_obs.shape[3]))
        useful_num = 0
        for i in range(next_obs.shape[0]):
            if batch_to_update_flag[i]:
                cur_batch_useful_num = int(mask[i].sum())
                res[useful_num: useful_num + cur_batch_useful_num] = next_obs[i, 0: cur_batch_useful_num]
                useful_num += cur_batch_useful_num
        return res[:useful_num]

    def _get_filled_state(self, batch, mask):
        # will dismiss the last state(different from before)
        state = batch['state'][:, :-1]
        res = th.zeros((state.shape[0] * state.shape[1], state.shape[2]))
        useful_num = 0
        for i in range(state.shape[0]):
            cur_batch_useful_num = int(mask[i].sum())
            res[useful_num: useful_num + cur_batch_useful_num] = state[i, 0: cur_batch_useful_num]
            useful_num += cur_batch_useful_num
        return res[:useful_num]

    def _get_filled_next_state(self, batch, mask):
        # will dismiss the last state(different from before)
        next_state = batch['state'][:, 1:] # use next state
        res = th.zeros((next_state.shape[0] * next_state.shape[1], next_state.shape[2]))
        useful_num = 0
        for i in range(next_state.shape[0]):
            cur_batch_useful_num = int(mask[i].sum())
            res[useful_num: useful_num + cur_batch_useful_num] = next_state[i, 0: cur_batch_useful_num]
            useful_num += cur_batch_useful_num
        return res[:useful_num]

    def train_exp(self, batch1, batch: EpisodeBatch, t_env: int, episode_num: int, batch_to_update_flag):
        rewards = batch["reward"][:, :-1]
        original_rewards = batch["original_reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        new_rewards = None
        if batch_to_update_flag.sum() > 0:
            agent_batch_obs = self._get_filled_next_obs(batch, mask, batch_to_update_flag)  # use next obs to calculate

            cur = 0
            while cur < agent_batch_obs.shape[0]:
                vae_update_batch_obs = agent_batch_obs[cur: cur + self.args.vae_density_update_batch_size]
                cur += self.args.vae_density_update_batch_size
                # use minimal batch size to update vae to keep stable
                if (vae_update_batch_obs.shape[0] <= 1
                        or vae_update_batch_obs.shape[0] < self.args.vae_density_update_batch_minsize):
                    continue
                for agent_id in range(self.args.n_agents):
                    self.density_model.update(vae_update_batch_obs[:, agent_id], None, agent_id)

            if agent_batch_obs.shape[0] > 1:
                new_rewards = torch.zeros((agent_batch_obs.shape[0], self.args.n_agents, 1))
                for agent_id in range(self.args.n_agents):
                    new_rewards[:, agent_id], _ = self.density_model.get_logprob_for(agent_batch_obs[:, agent_id], None, agent_id)


        # reward_cnt = 0
        changed_num = 0
        if new_rewards is not None:
            new_rewards = - new_rewards    # is negative!
            new_rewards = new_rewards.cuda()     # to add env value
            for i in range(mask.shape[0]):
                if batch_to_update_flag[i]:
                    cur_batch_change_num = int(mask[i].sum())

                    rewards[i, 0: cur_batch_change_num] = self.args.exp_policy_env_reward_weight * original_rewards[i,
                            0: cur_batch_change_num] + self.args.exp_reward_weight * new_rewards[changed_num: changed_num + cur_batch_change_num]

                    changed_num += cur_batch_change_num


        explore_out = []
        self.mac.explore_policy.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            q_outs = self.mac.explore_policy.forward(batch, t=t)
            explore_out.append(q_outs)
        explore_out = th.stack(explore_out, dim=1)

        chosen_action_qvals = th.gather(explore_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        target_explore_out = []
        self.target_mac.explore_policy.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_q_outs = self.target_mac.explore_policy.forward(batch, t=t)
            target_explore_out.append(target_q_outs)
        target_explore_out = th.stack(target_explore_out[1:], dim=1)  # Concat across time

        target_explore_out[avail_actions[:, 1:] == 0] = -9999999

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            explore_out_detach = explore_out.clone().detach()
            explore_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = explore_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_explore_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_explore_out.max(dim=3)[0]

        targets = rewards.squeeze(-1) + self.args.gamma * (1 - terminated).expand_as(
            target_max_qvals) * target_max_qvals
        # Td-error
        td_error = chosen_action_qvals - targets.detach()
        mask = mask.expand_as(td_error)
        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask
        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error ** 2).sum() / mask.sum()

        self.explore_optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.mac.explore_policy.parameters(), self.args.grad_norm_clip)
        self.explore_optimiser.step()

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("exp_loss", loss.item(), t_env)
            self.logger.log_stat("exp_grad_norm", grad_norm, t_env)
            self.log_stats_t = t_env

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]


        # Calculate estimated Q-Values
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time
        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time


        mac_out_detach = mac_out.clone().detach()
        mac_out_detach[avail_actions == 0] = -9999999
        cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]

        ce_losses_unmask = None
        ce_losses = None
        if self.args.use_adaptive_alpha:
            enc_output, enc_slf_attn = self.intrinsic.forward(batch)

            # reconstruction loss
            ce_losses_unmask = []
            ce_losses = []  # just for print
            ce_loss = th.tensor(0, dtype=th.float32, device=self.args.device)
            for agent_id in range(self.args.n_agents):
                ce_error_agent = self.CELoss(enc_output[:, agent_id],
                                             cur_max_actions.reshape(-1, self.args.n_agents)[:, agent_id])
                ce_error_agent = ce_error_agent.view(*mask.shape)

                # 0-out the targets that came from padded data
                masked_ce_error_agent = ce_error_agent * mask

                # Normal L2 loss, take mean over actual data
                ce_loss_agent = masked_ce_error_agent.sum() / mask.sum()
                ce_loss += ce_loss_agent
                ce_losses.append(ce_loss_agent.item())
                ce_losses_unmask.append(ce_error_agent.mean().item())
            ce_loss /= self.args.n_agents
            self.intrinsic_optimiser.zero_grad()
            ce_loss.backward()
            self.intrinsic_optimiser.step()


        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # Mix
        if self.mixer is not None:
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:])

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error ** 2).sum() / mask.sum()

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(
            self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        self.training_steps += 1
        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets_hard()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            if self.args.use_adaptive_alpha:
                self.logger.log_stat("pred_loss_unmask/pred_loss_mean", np.mean(ce_losses_unmask), t_env)
                self.logger.log_stat("pred_loss_unmask/pred_loss_std", np.std(ce_losses_unmask), t_env)
                self.logger.log_stat("pred_loss/pred_loss_mean", np.mean(ce_losses), t_env)
                self.logger.log_stat("pred_loss/pred_loss_std", np.std(ce_losses), t_env)
                for agent_id in range(self.args.n_agents):
                    self.logger.log_stat(f"p_loss_agent/pred_loss_{agent_id}", ce_losses[agent_id], t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item() / mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals *
                                                  mask).sum().item() / (mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean",
                                 (targets * mask).sum().item() / (mask_elems * self.args.n_agents), t_env)


    def _update_targets_hard(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())

    def _update_targets_soft(self, tau):
        # for target_param, param in zip(self.target_mac.parameters(), self.mac.parameters()):
        # update agent parameters
        for target_param, param in zip(self.target_mac.agent_parameters(), self.mac.agent_parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau)
        # update exploration parameters
        for target_param, param in zip(self.target_mac.explore_parameters(), self.mac.explore_parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau)
        if self.mixer is not None:
            for target_param, param in zip(self.target_mixer.parameters(), self.mixer.parameters()):
                target_param.data.copy_(
                    target_param.data * (1.0 - tau) + param.data * tau)

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.args.use_adaptive_alpha:
            self.intrinsic.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()
        self.density_model.cuda()


    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))


    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(
                th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(
            th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
