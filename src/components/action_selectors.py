import numpy as np
import torch
import torch as th
from scipy.special import entr
from torch.distributions import Categorical

from .epsilon_schedules import DecayThenFlatSchedule

REGISTRY = {}


class MultinomialActionSelector():

    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)
        self.test_greedy = getattr(args, "test_greedy", True)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):
        masked_policies = agent_inputs.clone()
        masked_policies[avail_actions == 0.0] = 0.0

        self.epsilon = self.schedule.eval(t_env)

        if test_mode and self.test_greedy:
            picked_actions = masked_policies.max(dim=2)[1]
        else:
            picked_actions = Categorical(masked_policies).sample().long()

        return picked_actions


REGISTRY["multinomial"] = MultinomialActionSelector


class EpsilonGreedyActionSelector:

    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay=args.decay)
        self.epsilon = self.schedule.eval(0)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):

        # Assuming agent_inputs is a batch of Q-Values for each agent bav
        self.epsilon = self.schedule.eval(t_env)

        if test_mode:
            # Greedy action selection only
            self.epsilon = self.args.evaluation_epsilon

        # mask actions that are excluded from selection
        masked_q_values = agent_inputs.clone()
        masked_q_values[avail_actions == 0.0] = -float("inf")  # should never be selected!

        random_numbers = th.rand_like(agent_inputs[:, :, 0])
        pick_random = (random_numbers < self.epsilon).long()
        random_actions = Categorical(avail_actions.float()).sample().long()

        picked_actions = pick_random * random_actions + (1 - pick_random) * masked_q_values.max(dim=2)[1]
        return picked_actions


REGISTRY["epsilon_greedy"] = EpsilonGreedyActionSelector


class SMMAEAlphaAdaptiveActionSelector:

    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay=args.decay)
        self.epsilon = self.schedule.eval(0)
        self.epsilons = torch.Tensor([self.epsilon for _ in range(self.args.n_agents)]).to(self.args.device)

    def select_action(self, agent_inputs, explore_inputs, avail_actions, t_env, alphas=None, test_mode=False):
        # Assuming agent_inputs is a batch of Q-Values for each agent bav
        if t_env < self.args.epsilon_anneal_time or alphas is None:       # don't use alpha if alpha is none
            epsilon = self.schedule.eval(t_env)
            self.epsilons = torch.Tensor([epsilon for _ in range(self.args.n_agents)]).to(self.args.device)
        else:
            self.epsilons = alphas

        if test_mode:
            # Greedy action selection only
            self.epsilons = torch.Tensor([self.args.evaluation_epsilon for _ in range(self.args.n_agents)]).to(
                self.args.device)

        self.epsilon = self.epsilons.mean()

        # mask actions that are excluded from selection
        masked_q_values = agent_inputs.clone()
        masked_q_values[avail_actions == 0.0] = -float("inf")  # should never be selected!

        random_numbers = th.rand_like(agent_inputs[:, :, 0])
        self.epsilons = self.epsilons.expand_as(random_numbers)
        pick_random = (random_numbers < self.epsilons).long()


        if explore_inputs is not None:
            # replace random_actions with exploration policy
            masked_explore_q_values = explore_inputs.clone()
            masked_explore_q_values[avail_actions == 0.0] = -float("inf")  # should never be selected!

            explore_random_numbers = th.rand_like(agent_inputs[:, :, 0])
            explore_random = (explore_random_numbers < 0.2).long()  # 0.2 prob random, 0.8 prob use exploration policy
            random_actions = Categorical(avail_actions.float()).sample().long()
            # explore action
            explore_actions = explore_random * random_actions + (1 - explore_random) * masked_explore_q_values.max(dim=2)[1]
        else:
            explore_actions = Categorical(avail_actions.float()).sample().long()   # choose to use random

        picked_actions = pick_random * explore_actions + (1 - pick_random) * masked_q_values.max(dim=2)[1]
        return picked_actions


REGISTRY["smmae_alpha_adaptive"] = SMMAEAlphaAdaptiveActionSelector


class DirectedActionSelector:

    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay=self.args.decay)
        self.epsilon = self.schedule.eval(0)

    def select_action(self, agent_inputs, explore_inputs, avail_actions, t_env, test_mode=False):
        # Assuming agent_inputs is a batch of Q-Values for each agent bav
        self.epsilon = self.schedule.eval(t_env)

        if test_mode:
            # Greedy action selection only
            self.epsilon = self.args.evaluation_epsilon

        # mask actions that are excluded from selection
        masked_q_values = agent_inputs.clone()
        masked_q_values[avail_actions == 0.0] = -float("inf")  # should never be selected!

        random_numbers = th.rand_like(agent_inputs[:, :, 0])
        pick_random = (random_numbers < self.epsilon).long()

        # replace random_actions with exploration policy
        masked_explore_q_values = explore_inputs.clone()
        masked_explore_q_values[avail_actions == 0.0] = -float("inf")  # should never be selected!

        explore_random_numbers = th.rand_like(agent_inputs[:, :, 0])
        explore_random = (explore_random_numbers < 0.2).long()
        random_actions = Categorical(avail_actions.float()).sample().long()
        explore_actions = explore_random * random_actions + (1 - explore_random) * masked_explore_q_values.max(dim=2)[1]

        picked_actions = pick_random * explore_actions + (1 - pick_random) * masked_q_values.max(dim=2)[1]
        return picked_actions


REGISTRY["directed"] = DirectedActionSelector


class SoftPoliciesSelector:

    def __init__(self, args):
        self.args = args

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):
        m = Categorical(agent_inputs)
        picked_actions = m.sample().long()
        return picked_actions


REGISTRY["soft_policies"] = SoftPoliciesSelector
