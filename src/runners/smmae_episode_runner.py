from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from components.episode_buffer import EpisodeBatch
from envs import REGISTRY as env_REGISTRY



class SMMAEEpisodeRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        self.n_agents = self.get_env_info()["n_agents"]
        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

    def setup(self, scheme, exp_scheme, groups, preprocess, mac, density_model):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.new_exp_batch = partial(EpisodeBatch, exp_scheme, groups, self.batch_size, self.episode_limit + 1,
                                     preprocess=preprocess, device=self.args.device)
        self.mac = mac
        self.density_model = density_model

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.exp_batch = self.new_exp_batch()
        self.env.reset()
        self.t = 0

    def run(self, alphas=None, test_mode=False):
        self.reset()

        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)

        while not terminated:
            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()]
            }
            self.batch.update(pre_transition_data, ts=self.t)

            pre_exp_transition_data = {
                # "agent_state": [self.env.get_agent_state(i) for i in range(self.n_agents)],
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_agent_actions(i) for i in range(self.n_agents)],
                "agent_id": [i for i in range(self.n_agents)],
                "obs": [self.env.get_obs()]
            }
            self.exp_batch.update(pre_exp_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            actions, agent_outs = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env,
                                                          alphas=alphas, test_mode=test_mode)

            reward, terminated, env_info = self.env.step(actions[0])
            episode_return += reward

            post_exp_transition_data = {
                    "actions": [actions[0, i] for i in range(self.n_agents)],
                    "reward": [(-99999,) for i in range(self.n_agents)],  # need to change
                    "original_reward": [(reward,) for i in range(self.n_agents)],  # need to change
                    "terminated": [(terminated != env_info.get("episode_limit", False),)],
                }
            self.exp_batch.update(post_exp_transition_data, ts=self.t)

            post_transition_data = {
                "actions": actions,
                "q_vals": agent_outs[0].detach(),
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }

            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1

        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()]
        }
        self.batch.update(last_data, ts=self.t)

        last_exp_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_agent_actions(i) for i in range(self.n_agents)],
            "agent_id": [i for i in range(self.n_agents)],
            "obs": [self.env.get_obs()]
        }
        self.exp_batch.update(last_exp_data, ts=self.t)

        # Select actions in the last stored state
        actions, agent_outs = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, alphas=alphas,
                                                      test_mode=test_mode)
        self.batch.update({"actions": actions, "q_vals": agent_outs[0].detach()}, ts=self.t)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_return)

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch, self.exp_batch

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean", v / stats["n_episodes"], self.t_env)
        stats.clear()
