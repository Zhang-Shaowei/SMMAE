import collections

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch

from torch import optim

class VAEDensity(object):
    def __init__(self, n_agents, obs_dim, args):
        # only for log
        self.counters = [collections.Counter() for i in range(n_agents)]

        self.n_agents = n_agents
        self.args = args
        self.lr = args.vae_density_lr
        self.beta = args.vae_density_beta
        self.enc = []
        self.enc_mu= []
        self.enc_logvar= []
        self.dec= []
        self.optimizers= []
        for i in range(self.n_agents):
            self.enc.append(nn.Sequential(
            nn.Linear(obs_dim, args.vae_density_encoder_hidden_size),
            nn.ReLU(),
            ))
            self.enc_mu.append(nn.Linear(args.vae_density_encoder_hidden_size, args.vae_density_latent_dim))
            self.enc_logvar.append(nn.Linear(args.vae_density_encoder_hidden_size, args.vae_density_latent_dim))
            self.dec.append(nn.Sequential(
            nn.Linear(args.vae_density_latent_dim, args.vae_density_decoder_hidden_size),
            nn.ReLU(),
            nn.Linear(args.vae_density_decoder_hidden_size, obs_dim),
            ))
            params = (list(self.enc[i].parameters()) +
                  list(self.enc_mu[i].parameters()) +
                  list(self.enc_logvar[i].parameters()) +
                  list(self.dec[i].parameters()))
            self.optimizers.append(optim.Adam(params, lr=self.lr))

    # only for log
    def _discretize_state(self, state):
        state = np.array(state).reshape(-1) / 1.0
        state = np.floor(state).astype(np.int)
        state = str(state.tolist())
        return state

    def cuda(self):
        for i in range(self.n_agents):
            self.enc[i].cuda()
            self.enc_mu[i].cuda()
            self.enc_logvar[i].cuda()
            self.dec[i].cuda()

    def get_logprob_for(self, agent_obs, agent_state, agent_id):
        res_log_prob = None  # only for compatible with old code
        entr_list = None  # only for compatible with old code
        if agent_obs is not None:  # only for compatible with old code
            try:
                obs = agent_obs.cuda() # in new code, agent_obs is tensor
            except:
                obs = torch.from_numpy(agent_obs.astype(np.float32)).cuda()

            with torch.no_grad():
                enc_features = self.enc[agent_id](obs)
                mu = self.enc_mu[agent_id](enc_features)
                logvar = self.enc_logvar[agent_id](enc_features)

                stds = (0.5 * logvar).exp()
                epsilon = torch.randn(mu.size(), device=self.args.device)
                latent_z = epsilon * stds + mu

                obs_distribution_params = self.dec[agent_id](latent_z)
                log_prob = -1. * F.mse_loss(obs, obs_distribution_params,
                                            reduction='none')
                log_prob = torch.sum(log_prob, -1, keepdim=True)
            res_log_prob = log_prob.detach()

        # ------------------------------only for log------------------------
        if agent_state is not None: # only for compatible with old code
            entr_list = []
            for i in range(agent_state.shape[0]):
                log_state = self._discretize_state(agent_state[i])
                count = self.counters[agent_id].get(log_state, 1)
                total = sum(self.counters[agent_id].values())
                p = count / total
                logprob = np.log(p)
                entr = -p * logprob
                entr_list.append(entr)
        # ------------------------------------------------------------------

        return res_log_prob, entr_list    # will return entropy list rather than entropy

    def update(self, agent_obs, agent_state, agent_id):
        # ------------------------only for log----------------------------------
        if agent_state is not None:                                    # only for compatible with old code
            for i in range(agent_state.shape[0]):
                log_state = self._discretize_state(agent_state[i])
                self.counters[agent_id].update({log_state: 1})
        # ----------------------------------------------------------------------
        if agent_obs is None:   # only for compatible with old code
            return

        try:
            obs = agent_obs.cuda()  # in new code agent_obs is tensor
        # obs = torch.from_numpy(state.copy().astype(np.float32)).cuda()
        except:
            obs = torch.from_numpy(agent_obs.astype(np.float32)).cuda() # use batch

        enc_features = self.enc[agent_id](obs)
        mu = self.enc_mu[agent_id](enc_features)
        logvar = self.enc_logvar[agent_id](enc_features)

        stds = (0.5 * logvar).exp()
        epsilon = torch.randn(mu.size(), device=self.args.device)
        latent_z = epsilon * stds + mu

        kle = -0.5 * torch.sum(
            1 + logvar - mu.pow(2) - logvar.exp(), dim=1
        ).mean()

        obs_distribution_params = self.dec[agent_id](latent_z)
        log_prob = -1. * F.mse_loss(obs, obs_distribution_params,
                                    reduction='mean')

        loss = self.beta * kle - log_prob

        self.optimizers[agent_id].zero_grad()
        loss.backward()
        self.optimizers[agent_id].step()

        return loss.cpu().item()

    def get_total_entropy(self, agent_id):
        counts = np.asarray(list(self.counters[agent_id].values()), dtype=np.float32)
        probs = counts / counts.sum()
        entropy = -(probs * np.log(probs)).sum()
        return entropy


class ExplorePolicy:
    def __init__(self, exp_scheme, args):
        self.args = args
        self.n_agents = args.n_agents
        input_shape = exp_scheme["state"]["vshape"] + self.n_agents
        self.network = ExploreNetwork(input_shape, args)
        self.hidden = None

    def parameters(self):
        return self.network.parameters()

    def init_hidden(self, batch_size):
        # (batch-size, n_agents, rnn_hidden_dim)
        self.hidden = self.network.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)

    def forward(self, ep_batch, t, test_mode=False):
        inputs = self._build_inputs(ep_batch, t)
        q_outs, self.hidden = self.network(inputs, self.hidden)
        return q_outs.view(ep_batch.batch_size, self.n_agents, -1)

    def _build_inputs(self, batch, t):
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["state"][:, t].repeat(1, 1, self.n_agents).view(bs * self.n_agents, -1))
        inputs.append(th.eye(self.n_agents, device=batch.device).expand(bs, -1, -1))
        inputs = th.cat([x.reshape(bs * self.n_agents, -1) for x in inputs], dim=1)
        return inputs

    def load_state(self, other_policy):
        self.network.load_state_dict(other_policy.network.state_dict())

    def state_dict(self):
        return self.network.state_dict()

    def cuda(self):
        self.network.cuda()


class ExploreNetwork(nn.Module):
    def __init__(self, input_shape, args):
        super(ExploreNetwork, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        # if self.args.use_rnn:
        #     self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        # else:
        #     self.rnn = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        # if self.args.use_rnn:
        #     h = self.rnn(x, h_in)
        # else:
        #     h = F.relu(self.rnn(x))
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h
