import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from intrinsic.attention import MultiHeadAttention, PositionwiseFeedForward


class RNNModel(nn.Module):
    def __init__(self, args, dropout=0.5):
        super(RNNModel, self).__init__()
        self.args = args

        self.state_dim = int(np.prod(args.state_shape))
        self.fc1 = nn.Linear(self.state_dim, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_agents * args.n_actions)

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        a = self.fc2(h)
        return a, h

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()


class CoopPredMLP:
    def __init__(self, args):
        self.args = args
        self.action_predictor = RNNModel(args).to(args.device)
        self.state_dim = int(np.prod(args.state_shape))
        self.hidden_states = None

    def init_hidden(self, batch_size):
        self.hidden_states = self.action_predictor.init_hidden().expand(batch_size, -1)  # bav

    def forward(self, ep_batch, t):
        state = ep_batch["state"][:, t].reshape(ep_batch.batch_size, self.state_dim)
        actions_pred, self.hidden_states = self.action_predictor(state, self.hidden_states)
        return actions_pred.reshape(ep_batch.batch_size, self.args.n_actions, self.args.n_agents)

    def parameters(self):
        return self.action_predictor.parameters()

    def cuda(self):
        self.action_predictor.cuda()


class CoopPred(nn.Module):
    def __init__(self, args, scheme):
        super(CoopPred, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        n_head = args.n_head
        dropout = 0.1
        d_model = self._get_input_shape(scheme)
        d_k = 32  # 64 / 2
        d_v = 32  # 64 / 2
        d_inner = 32  # 2048
        d_out = args.n_actions
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, d_out, dropout=dropout)

    def forward(self, ep_batch):
        obs = ep_batch["obs"][:, :-1].reshape(-1, *ep_batch["obs"].shape[-2:])
        agent_ids = th.eye(self.n_agents, device=self.args.device).expand(obs.shape[0], -1, -1)
        slf_attn_mask = (1 - th.eye(self.n_agents)).to(self.args.device)  # mask, cannot see self
        inputs = th.cat([obs, agent_ids], dim=-1)
        enc_output, enc_slf_attn = self.slf_attn(inputs, inputs, inputs, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn

    def init_hidden(self, batch_size):
        pass

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"] + self.args.n_agents
        return input_shape

