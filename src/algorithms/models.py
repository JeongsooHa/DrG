from typing import Optional, List
import torch
from torch import jit, nn
from torch.nn import functional as F
import torch.distributions
from torch.distributions.normal import Normal
from torch.distributions.transforms import Transform, TanhTransform
from torch.distributions.transformed_distribution import TransformedDistribution
import numpy as np


def weight_init(m):
	"""Custom weight init for Conv2D and Linear layers"""
	if isinstance(m, nn.Linear):
		nn.init.orthogonal_(m.weight.data)
		if hasattr(m.bias, 'data'):
			m.bias.data.fill_(0.0)
	elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
		# delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
		assert m.weight.size(2) == m.weight.size(3)
		m.weight.data.fill_(0.0)
		if hasattr(m.bias, 'data'):
			m.bias.data.fill_(0.0)
		mid = m.weight.size(2) // 2
		gain = nn.init.calculate_gain('relu')
		nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)

# Wraps the input tuple for a function to process a time x batch x features sequence in batch x features (assumes one output)
def bottle(f, x_tuple):
  x_sizes = tuple(map(lambda x: x.size(), x_tuple))
  y = f(*map(lambda x: x[0].view(x[1][0] * x[1][1], *x[1][2:]), zip(x_tuple, x_sizes)))
  y_size = y.size()
  output = y.view(x_sizes[0][0], x_sizes[0][1], *y_size[1:])
  return output

class TransitionModel(jit.ScriptModule):
  __constants__ = ['min_std_dev']

  def __init__(self, belief_size, state_size, action_size, hidden_size, embedding_size, activation_function='relu', min_std_dev=0.1):
    super().__init__()
    self.act_fn = getattr(F, activation_function)
    self.min_std_dev = min_std_dev
    self.fc_embed_state_action = nn.Linear(state_size + action_size, belief_size)
    self.rnn = nn.GRUCell(belief_size, belief_size)
    self.fc_embed_belief_prior = nn.Linear(belief_size, hidden_size)
    self.fc_state_prior = nn.Linear(hidden_size, 2 * state_size)
    self.fc_embed_belief_posterior = nn.Linear(belief_size + embedding_size, hidden_size)
    self.fc_state_posterior = nn.Linear(hidden_size, 2 * state_size)
    self.apply(weight_init)
    self.modules = [self.fc_embed_state_action, self.fc_embed_belief_prior, self.fc_state_prior, self.fc_embed_belief_posterior, self.fc_state_posterior]

  # Operates over (previous) state, (previous) actions, (previous) belief, (previous) nonterminals (mask), and (current) observations
  # Diagram of expected inputs and outputs for T = 5 (-x- signifying beginning of output belief/state that gets sliced off):
  # t :  0  1  2  3  4  5
  # o :    -X--X--X--X--X-
  # a : -X--X--X--X--X-
  # n : -X--X--X--X--X-
  # pb: -X-
  # ps: -X-
  # b : -x--X--X--X--X--X-
  # s : -x--X--X--X--X--X-
  # @jit.script_method
  def forward(self, prev_state:torch.Tensor, actions:torch.Tensor, prev_belief:torch.Tensor, observations:Optional[torch.Tensor]=None, nonterminals:Optional[torch.Tensor]=None) -> List[torch.Tensor]:
    '''
    Input: init_belief, init_state:  torch.Size([50, 200]) torch.Size([50, 30])
    Output: beliefs, prior_states, prior_means, prior_std_devs, posterior_states, posterior_means, posterior_std_devs
            torch.Size([49, 50, 200]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30])
    '''
    # Create lists for hidden states (cannot use single tensor as buffer because autograd won't work with inplace writes)
    T = actions.size(0) + 1
    beliefs, prior_states, prior_means, prior_std_devs, posterior_states, posterior_means, posterior_std_devs = [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T
    beliefs[0], prior_states[0], posterior_states[0] = prev_belief, prev_state, prev_state
    # Loop over time sequence
    for t in range(T - 1):
      _state = prior_states[t] if observations is None else posterior_states[t]  # Select appropriate previous state
      _state = _state if nonterminals is None else _state * nonterminals[t]  # Mask if previous transition was terminal
      # Compute belief (deterministic hidden state)
      hidden = self.act_fn(self.fc_embed_state_action(torch.cat([_state, actions[t]], dim=1)))
      beliefs[t + 1] = self.rnn(hidden, beliefs[t])
      # Compute state prior by applying transition dynamics
      hidden = self.act_fn(self.fc_embed_belief_prior(beliefs[t + 1]))
      prior_means[t + 1], _prior_std_dev = torch.chunk(self.fc_state_prior(hidden), 2, dim=1)
      prior_std_devs[t + 1] = F.softplus(_prior_std_dev) + self.min_std_dev
      prior_states[t + 1] = prior_means[t + 1] + prior_std_devs[t + 1] * torch.randn_like(prior_means[t + 1])     
      if observations is not None:
        # Compute state posterior by applying transition dynamics and using current observation
        t_ = t - 1  # Use t_ to deal with different time indexing for observations
        hidden = self.act_fn(self.fc_embed_belief_posterior(torch.cat([beliefs[t + 1], observations[t_ + 1]], dim=1)))
        posterior_means[t + 1], _posterior_std_dev = torch.chunk(self.fc_state_posterior(hidden), 2, dim=1)
        posterior_std_devs[t + 1] = F.softplus(_posterior_std_dev) + self.min_std_dev
        posterior_states[t + 1] = posterior_means[t + 1] + posterior_std_devs[t + 1] * torch.randn_like(posterior_means[t + 1])
    # Return new hidden states
    hidden = [torch.stack(beliefs[1:], dim=0), torch.stack(prior_states[1:], dim=0), torch.stack(prior_means[1:], dim=0), torch.stack(prior_std_devs[1:], dim=0)]
    if observations is not None:
      hidden += [torch.stack(posterior_states[1:], dim=0), torch.stack(posterior_means[1:], dim=0), torch.stack(posterior_std_devs[1:], dim=0)]
    return hidden

  def forward_rollout(self, prev_state:torch.Tensor, actions:torch.Tensor, prev_belief:torch.Tensor, observations:Optional[torch.Tensor]=None, nonterminals:Optional[torch.Tensor]=None) -> List[torch.Tensor]:
    '''
    Input: init_belief, init_state:  torch.Size([50, 200]) torch.Size([50, 30])
    Output: beliefs, prior_states, prior_means, prior_std_devs, posterior_states, posterior_means, posterior_std_devs
            torch.Size([49, 50, 200]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30])
    '''
    # Create lists for hidden states (cannot use single tensor as buffer because autograd won't work with inplace writes)
    T = actions.size(0) + 1
    beliefs, prior_states, prior_means, prior_std_devs, posterior_states, posterior_means, posterior_std_devs = [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T
    beliefs[0], prior_states[0], posterior_states[0] = prev_belief, prev_state, prev_state
    # Loop over time sequence
    for t in range(T - 1):
      _state = prior_states[t] if observations is None else posterior_states[t]  # Select appropriate previous state
      _state = _state if nonterminals is None else _state * nonterminals[t]  # Mask if previous transition was terminal
      # Compute belief (deterministic hidden state)
      hidden = self.act_fn(self.fc_embed_state_action(torch.cat([_state, actions[t]], dim=1)))
      beliefs[t + 1] = self.rnn(hidden, beliefs[t])
      # Compute state prior by applying transition dynamics
      hidden = self.act_fn(self.fc_embed_belief_prior(beliefs[t + 1]))
      prior_means[t + 1], _prior_std_dev = torch.chunk(self.fc_state_prior(hidden), 2, dim=1)
      prior_std_devs[t + 1] = F.softplus(_prior_std_dev) + self.min_std_dev
      prior_states[t + 1] = prior_means[t + 1] + prior_std_devs[t + 1] * torch.randn_like(prior_means[t + 1])     
      if observations is not None:
        # Compute state posterior by applying transition dynamics and using current observation
        t_ = t - 1  # Use t_ to deal with different time indexing for observations
        hidden = self.act_fn(self.fc_embed_belief_posterior(torch.cat([beliefs[t + 1], observations[t_ + 1]], dim=1)))
        posterior_means[t + 1], _posterior_std_dev = torch.chunk(self.fc_state_posterior(hidden), 2, dim=1)
        posterior_std_devs[t + 1] = F.softplus(_posterior_std_dev) + self.min_std_dev
        posterior_states[t + 1] = posterior_means[t + 1] + posterior_std_devs[t + 1] * torch.randn_like(posterior_means[t + 1])
    # Return new hidden states
    hidden = [torch.stack(beliefs[1:], dim=0), torch.stack(prior_states[1:], dim=0), torch.stack(prior_means[1:], dim=0), torch.stack(prior_std_devs[1:], dim=0)]
    if observations is not None:
      hidden += [torch.stack(posterior_states[1:], dim=0), torch.stack(posterior_means[1:], dim=0), torch.stack(posterior_std_devs[1:], dim=0)]
    return hidden

class ContrastiveObsModel(jit.ScriptModule):
  def __init__(self, state_size, embedding_size, activation_function='relu'):
    super().__init__()
    
    self.act_fn = getattr(F, activation_function)
    self.state_size = state_size 
    self.embedding_size = embedding_size
    
    self.obs_enc1 = nn.Linear(self.embedding_size, self.state_size)
    self.obs_enc2 = nn.Linear(self.state_size, self.state_size)
    self.W = nn.Parameter(torch.rand(self.state_size, self.state_size))
    # self.obs_enc2 = nn.Linear(self.embedding_size, self.state_size)
    
    # self.state_merge1 = nn.Linear(self.state_size, self.state_size)
    # self.state_merge2 = nn.Linear(self.state_size, self.state_size)
    self.cross_entropy_loss = nn.CrossEntropyLoss()
    self.apply(weight_init)
    self.modules = [self.obs_enc1, self.obs_enc2]

  def compute_logits(self, z, x):
    """
    Uses logits trick for CURL:
    - compute (B,B) matrix z (W x.T)
    - positives are all diagonal elements
    - negatives are all other elements
    - to compute loss use multiclass cross entropy with identity matrix for labels
    """
    Wz = torch.matmul(self.W, x.T)  # (z_dim,B)
    logits = torch.matmul(z, Wz)  # (B,B)
    logits = logits - torch.max(logits, 1)[0][:, None]
    
    return logits

  def forward(self, z, x):
    """
    Both inputs have the shape of [batch_sz, length, dim]. For each positive sample, we use the rest of batch_sz * length - 1 samples as negative samples
    Args:
        z (tensor): latent state
        x (tensor): encoded observation
    """

    x = torch.reshape(x, (-1, x.shape[-1]))
    z = torch.reshape(z, (-1, z.shape[-1]))

    # use mixed precision of float32 to avoid overflow
    x = self.act_fn(self.obs_enc1(x))
    z = self.act_fn(self.obs_enc2(z))
    
    logits = self.compute_logits(z, x)
    labels = torch.arange(logits.shape[0]).long().cuda()
    
    return self.cross_entropy_loss(logits, labels)

class ContrastiveObsModel_2(jit.ScriptModule):
  def __init__(self, state_size, embedding_size, activation_function='relu'):
    super().__init__()
    
    self.act_fn = getattr(F, activation_function)
    self.state_size = state_size 
    self.embedding_size = embedding_size
    
    self.obs_enc1 = nn.Linear(self.embedding_size, self.embedding_size)
    self.obs_enc2 = nn.Linear(self.embedding_size, self.state_size)
    
    self.state_merge1 = nn.Linear(self.state_size, self.state_size)
    self.state_merge2 = nn.Linear(self.state_size, self.state_size)
    self.apply(weight_init)
    self.modules = [self.obs_enc1, self.obs_enc2, self.state_merge1, self.state_merge2]

  def forward(self, z, x):
    """
    Both inputs have the shape of [batch_sz, length, dim]. For each positive sample, we use the rest of batch_sz * length - 1 samples as negative samples
    Args:
        z (tensor): latent state
        x (tensor): encoded observation
    """

    x = torch.reshape(x, (-1, x.shape[-1]))
    z = torch.reshape(z, (-1, z.shape[-1]))

    # use mixed precision of float32 to avoid overflow

    x = self.act_fn(self.obs_enc1(x))
    x = self.act_fn(self.obs_enc2(x))
    
    z = self.act_fn(self.state_merge1(z))
    z = self.act_fn(self.state_merge2(z))
    

    weight_mat = torch.matmul(z, x.T)

    positive = torch.diag(weight_mat)
    norm = torch.logsumexp(weight_mat, 1)

    # compute the infonce loss and change the predicion back to float16
    info_nce = positive - norm

    return info_nce

class BackTransitionModel(jit.ScriptModule):
  def __init__(self, belief_size, state_size, action_size, hidden_size, embedding_size, activation_function='relu'):
    # [--belief-size: 200, --hidden-size: 200, --state-size: 30]
    super().__init__()
    self.act_fn = getattr(F, activation_function)
    self.fc_belief_posterior = nn.Linear(belief_size + state_size, hidden_size)
    self.fc_adding_action = nn.Linear(hidden_size+action_size, hidden_size)
    self.fc_predict_embed = nn.Linear(hidden_size, embedding_size)
    self.apply(weight_init)
    self.modules = [self.fc_belief_posterior, self.fc_adding_action, self.fc_predict_embed]

  # @jit.script_method
  def forward(self, belief, posterior, action):
    x = torch.cat([belief, posterior],dim=1)
    hidden = self.act_fn(self.fc_belief_posterior(x))
    x = torch.cat([hidden, action],dim=1)
    hidden = self.act_fn(self.fc_adding_action(x))
    pred_before_state = self.fc_predict_embed(hidden)
    return pred_before_state

class InverseDynamicsModel(jit.ScriptModule):
      # [--belief-size: 200, --hidden-size: 200, --state-size: 30]
  def __init__(self, state_size, action_size, embedding_size, activation_function='relu'):
    super().__init__()
    self.act_fn = getattr(F, activation_function)
    self.fc1 = nn.Linear(2*state_size, embedding_size)
    self.fc2 = nn.Linear(embedding_size, embedding_size)
    self.fc3 = nn.Linear(embedding_size, action_size)
    self.apply(weight_init)
    self.modules = [self.fc1, self.fc2, self.fc3]

  # @jit.script_method
  def forward(self, current_state, next_state):
    hidden = self.act_fn(self.fc1(torch.cat([current_state, next_state], dim=1)))
    hidden = self.act_fn(self.fc2(hidden))
    pred_action = self.fc3(hidden)
    return pred_action

class InverseDynamicsModel_merge(jit.ScriptModule):
  # [--belief-size: 200, --hidden-size: 200, --state-size: 30]
  def __init__(self, state_size, action_size, embedding_size, activation_function='relu'):
    super().__init__()
    self.act_fn = getattr(F, activation_function)
    self.belief_post_1 = nn.Linear(state_size, state_size)
    self.belief_post_2 = nn.Linear(state_size, state_size)
    self.fc1 = nn.Linear(2*state_size, embedding_size)
    self.fc2 = nn.Linear(embedding_size, embedding_size)
    self.fc3 = nn.Linear(embedding_size, action_size)
    self.apply(weight_init)
    self.modules = [self.belief_post_1, self.belief_post_2, self.fc1, self.fc2, self.fc3]

  # @jit.script_method
  def forward(self, current_state, next_state):
    hidden_1 = self.act_fn(self.belief_post_1(current_state))
    hidden_2 = self.act_fn(self.belief_post_2(next_state))
    hidden = self.act_fn(self.fc1(torch.cat([hidden_1, hidden_2], dim=1)))
    hidden = self.act_fn(self.fc2(hidden))
    pred_action = self.fc3(hidden)
    return pred_action

class Discriminator(jit.ScriptModule):
  def __init__(self, state_size, hidden_size, activation_function='relu'):
    # [--belief-size: 200, --hidden-size: 200, --state-size: 30]
    super().__init__()
    self.act_fn = getattr(F, activation_function)
    self.fc1 = nn.Linear(state_size, hidden_size)
    # self.bn1 = nn.BatchNorm1d(hidden_size)
    self.fc2 = nn.Linear(hidden_size, hidden_size)
    # self.bn2 = nn.BatchNorm1d(hidden_size)
    self.fc3 = nn.Linear(hidden_size, 1)
    self.apply(weight_init)
    self.modules = [self.fc1, self.fc2, self.fc3]

  # @jit.script_method
  def forward(self, state):
    state = state.view(-1, state.shape[-1])
    hidden = self.act_fn(self.fc1(state))
    # hidden = self.bn1(hidden)
    hidden = self.act_fn(self.fc2(hidden))
    # hidden = self.bn2(hidden)
    validity = self.fc3(hidden)
    return validity

class SymbolicObservationModel(jit.ScriptModule):
  def __init__(self, observation_size, belief_size, state_size, embedding_size, activation_function='relu'):
    super().__init__()
    self.act_fn = getattr(F, activation_function)
    self.fc1 = nn.Linear(belief_size + state_size, embedding_size)
    self.fc2 = nn.Linear(embedding_size, embedding_size)
    self.fc3 = nn.Linear(embedding_size, observation_size)
    self.apply(weight_init)
    self.modules = [self.fc1, self.fc2, self.fc3]

  @jit.script_method
  def forward(self, belief, state):
    hidden = self.act_fn(self.fc1(torch.cat([belief, state], dim=1)))
    hidden = self.act_fn(self.fc2(hidden))
    observation = self.fc3(hidden)
    return observation


class VisualObservationModel(jit.ScriptModule):
  __constants__ = ['embedding_size']
  
  def __init__(self, belief_size, state_size, embedding_size, activation_function='relu'):
    super().__init__()
    self.act_fn = getattr(F, activation_function)
    self.embedding_size = embedding_size
    self.fc1 = nn.Linear(belief_size + state_size, embedding_size)
    self.conv1 = nn.ConvTranspose2d(embedding_size, 128, 5, stride=2)
    self.conv2 = nn.ConvTranspose2d(128, 64, 5, stride=2)
    self.conv3 = nn.ConvTranspose2d(64, 32, 6, stride=2)
    self.conv4 = nn.ConvTranspose2d(32, 3, 6, stride=2)
    self.apply(weight_init)
    self.modules = [self.fc1, self.conv1, self.conv2, self.conv3, self.conv4]

  @jit.script_method
  def forward(self, belief, state):
    hidden = self.fc1(torch.cat([belief, state], dim=1))  # No nonlinearity here
    hidden = hidden.view(-1, self.embedding_size, 1, 1)
    hidden = self.act_fn(self.conv1(hidden))
    hidden = self.act_fn(self.conv2(hidden))
    hidden = self.act_fn(self.conv3(hidden))
    observation = self.conv4(hidden)
    return observation

class RewardModel(jit.ScriptModule):
  def __init__(self, belief_size, state_size, hidden_size, activation_function='relu'):
    # [--belief-size: 200, --hidden-size: 200, --state-size: 30]
    super().__init__()
    self.act_fn = getattr(F, activation_function)
    self.fc1 = nn.Linear(belief_size + state_size, hidden_size)
    self.fc2 = nn.Linear(hidden_size, hidden_size)
    self.fc3 = nn.Linear(hidden_size, 1)
    self.apply(weight_init)
    self.modules = [self.fc1, self.fc2, self.fc3]

  # @jit.script_method
  def forward(self, belief, state):
    x = torch.cat([belief, state],dim=1)
    hidden = self.act_fn(self.fc1(x))
    hidden = self.act_fn(self.fc2(hidden))
    reward = self.fc3(hidden).squeeze(dim=1)
    return reward

class ValueModel(jit.ScriptModule):
  def __init__(self, belief_size, state_size, hidden_size, activation_function='relu'):
    super().__init__()
    self.act_fn = getattr(F, activation_function)
    self.fc1 = nn.Linear(belief_size + state_size, hidden_size)
    self.fc2 = nn.Linear(hidden_size, hidden_size)
    self.fc3 = nn.Linear(hidden_size, hidden_size)
    self.fc4 = nn.Linear(hidden_size, 1)
    self.apply(weight_init)
    self.modules = [self.fc1, self.fc2, self.fc3, self.fc4]

  @jit.script_method
  def forward(self, belief, state):
    x = torch.cat([belief, state],dim=1)
    hidden = self.act_fn(self.fc1(x))
    hidden = self.act_fn(self.fc2(hidden))
    hidden = self.act_fn(self.fc3(hidden))
    reward = self.fc4(hidden).squeeze(dim=1)
    return reward

class ActorModel(jit.ScriptModule):
  def __init__(self, belief_size, state_size, hidden_size, action_size, dist='tanh_normal',
                activation_function='elu', min_std=1e-4, init_std=5, mean_scale=5):
    super().__init__()
    self.act_fn = getattr(F, activation_function)
    self.fc1 = nn.Linear(belief_size + state_size, hidden_size)
    self.fc2 = nn.Linear(hidden_size, hidden_size)
    self.fc3 = nn.Linear(hidden_size, hidden_size)
    self.fc4 = nn.Linear(hidden_size, hidden_size)
    self.fc5 = nn.Linear(hidden_size, 2*action_size)
    self.apply(weight_init)
    self.modules = [self.fc1, self.fc2, self.fc3, self.fc4, self.fc5]

    self._dist = dist
    self._min_std = min_std
    self._init_std = init_std
    self._mean_scale = mean_scale

  @jit.script_method
  def forward(self, belief, state):
    raw_init_std = torch.log(torch.exp(self._init_std) - 1)
    x = torch.cat([belief, state],dim=1)
    hidden = self.act_fn(self.fc1(x))
    hidden = self.act_fn(self.fc2(hidden))
    hidden = self.act_fn(self.fc3(hidden))
    hidden = self.act_fn(self.fc4(hidden))
    action = self.fc5(hidden).squeeze(dim=1)

    action_mean, action_std_dev = torch.chunk(action, 2, dim=1)
    action_mean = self._mean_scale * torch.tanh(action_mean / self._mean_scale)
    action_std = F.softplus(action_std_dev + raw_init_std) + self._min_std
    return action_mean, action_std

  def get_action(self, belief, state, det=False):
    action_mean, action_std = self.forward(belief, state)
    dist = Normal(action_mean, action_std)
    dist = TransformedDistribution(dist, TanhBijector())
    dist = torch.distributions.Independent(dist,1)
    dist = SampleDist(dist)
    if det: return dist.mode()
    else: return dist.rsample()


class SymbolicEncoder(jit.ScriptModule):
  def __init__(self, observation_size, embedding_size, activation_function='relu'):
    super().__init__()
    self.act_fn = getattr(F, activation_function)
    self.fc1 = nn.Linear(observation_size, embedding_size)
    self.fc2 = nn.Linear(embedding_size, embedding_size)
    self.fc3 = nn.Linear(embedding_size, embedding_size)
    self.apply(weight_init)
    self.modules = [self.fc1, self.fc2, self.fc3]

  @jit.script_method
  def forward(self, observation):
    hidden = self.act_fn(self.fc1(observation))
    hidden = self.act_fn(self.fc2(hidden))
    hidden = self.fc3(hidden)
    return hidden

class VisualEncoder(jit.ScriptModule):
  __constants__ = ['embedding_size']
  
  def __init__(self, embedding_size, activation_function='relu'):
    super().__init__()
    self.act_fn = getattr(F, activation_function)
    self.embedding_size = embedding_size
    self.conv1 = nn.Conv2d(3, 32, 4, stride=2)
    self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
    self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
    self.conv4 = nn.Conv2d(128, 256, 4, stride=2)
    self.fc = nn.Identity() if embedding_size == 1024 else nn.Linear(1024, embedding_size)
    self.ln = nn.LayerNorm(embedding_size)
    self.apply(weight_init)
    self.modules = [self.conv1, self.conv2, self.conv3, self.conv4]

  @jit.script_method
  def forward(self, observation): #, keypoint=False):
    hidden = self.act_fn(self.conv1(observation))
    hidden = self.act_fn(self.conv2(hidden))
    hidden = self.act_fn(self.conv3(hidden))
    hidden = self.act_fn(self.conv4(hidden))
    # if keypoint:
    #     return hidden
    hidden = hidden.view(-1, 1024)
    hidden = self.fc(hidden)  # Identity if embedding size is 1024 else linear projection
    # hidden = self.ln(hidden)
    return hidden

# "atanh", "TanhBijector" and "SampleDist" are from the following repo
# https://github.com/juliusfrost/dreamer-pytorch
def atanh(x):
    return 0.5 * torch.log((1 + x) / (1 - x))

class TanhBijector(torch.distributions.Transform):
    def __init__(self):
        super().__init__()
        self.bijective = True

    @property
    def sign(self): return 1.

    def _call(self, x): return torch.tanh(x)

    def _inverse(self, y: torch.Tensor):
        y = torch.where(
            (torch.abs(y) <= 1.),
            torch.clamp(y, -0.99999997, 0.99999997),
            y)
        y = atanh(y)
        return y

    def log_abs_det_jacobian(self, x, y):
        return 2. * (np.log(2) - x - F.softplus(-2. * x))


class SampleDist:
  def __init__(self, dist, samples=100):
    self._dist = dist
    self._samples = samples

  @property
  def name(self):
    return 'SampleDist'

  def __getattr__(self, name):
    return getattr(self._dist, name)

  def mean(self):
    sample = dist.rsample()
    return torch.mean(sample, 0)

  def mode(self):
    dist = self._dist.expand((self._samples, *self._dist.batch_shape))
    sample = dist.rsample()
    logprob = dist.log_prob(sample)
    batch_size = sample.size(1)
    feature_size = sample.size(2)
    indices = torch.argmax(logprob, dim=0).reshape(1, batch_size, 1).expand(1, batch_size, feature_size)
    return torch.gather(sample, 0, indices).squeeze(0)

  def entropy(self):
    dist = self._dist.expand((self._samples, *self._dist.batch_shape))
    sample = dist.rsample()
    logprob = dist.log_prob(sample)
    return -torch.mean(logprob, 0)

  def sample(self):
    return self._dist.sample()
