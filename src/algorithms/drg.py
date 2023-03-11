import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Normal
from torch.distributions.kl import kl_divergence

from torch.autograd import Variable
from copy import deepcopy
import utils
import algorithms.modules as m
from algorithms.models import bottle, VisualEncoder, VisualObservationModel, \
    SymbolicObservationModel, RewardModel, TransitionModel, \
    ValueModel, ActorModel, InverseDynamicsModel, InverseDynamicsModel_merge, \
    BackTransitionModel, ContrastiveObsModel, Discriminator
import os
from tqdm import tqdm
from torchvision.utils import make_grid, save_image
import time
from tensorboardX import SummaryWriter

class DRG(object):
    def __init__(self, obs_shape, action_shape, args):
        print("######## DRG ########")

        self.args = args
        checking = False
        self.metrics = {
            'steps': [], 'episodes': [], 'train_rewards': [], \
            'test_episodes_de': [], 'test_rewards_de': [], \
            'test_episodes_ce': [], 'test_rewards_ce': [], \
            'test_episodes_ch': [], 'test_rewards_ch': [], \
            'test_episodes_ve': [], 'test_rewards_ve': [], \
            'test_episodes_vh': [], 'test_rewards_vh': [], \
            # 'test_episodes_vn': [], 'test_rewards_vn': [], \
            'reward_loss': [], 'kl_loss': [], 'actor_loss': [], 'value_loss': [], \
            'observation_loss':[], 'encoder_loss':[], 'back_transition_loss': [], \
            'inv_posterior_loss': [], 'inv_belief_loss': [], 'inv_belief_post_loss': [],\
            'contrast_post_loss': [], 'contrast_belief_loss': [], 'contrast_belief_post_loss': [],\
            'contrast_encoder_loss': []
            }
        
        self.action_size = action_shape[0]
        self.transition_model = TransitionModel(self.args.belief_size, self.args.state_size, self.action_size, self.args.hidden_size, self.args.embedding_size, self.args.dense_activation_function).cuda()
        if self.args.target_transition_loss:
            self.target_transition_model = TransitionModel(self.args.belief_size, self.args.state_size, self.action_size, self.args.hidden_size, self.args.embedding_size, self.args.dense_activation_function).cuda()
            # self.target_transition_model = deepcopy(self.transition_model)
            self.hard_update_params(self.transition_model, self.target_transition_model)
        
        if self.args.use_adv_loss:
            self.Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
            self.adversarial_loss = torch.nn.BCEWithLogitsLoss().cuda()
            self.discriminator = Discriminator(self.args.state_size, self.args.hidden_size).cuda()
        
        self.reward_model = RewardModel(self.args.belief_size, self.args.state_size, self.args.hidden_size, self.args.dense_activation_function).cuda()
        self.encoder = VisualEncoder(self.args.embedding_size, self.args.cnn_activation_function).cuda()
        self.target_encoder = deepcopy(self.encoder)
        
        if self.args.observation_loss:
            self.observation_model = VisualObservationModel(self.args.belief_size, self.args.state_size, self.args.embedding_size, self.args.cnn_activation_function).cuda()
        if self.args.back_transition_loss:
            self.back_transition_model = TransitionModel(self.args.belief_size, self.args.state_size, self.action_size, self.args.hidden_size, self.args.embedding_size, self.args.dense_activation_function).cuda()
            # self.back_transition_model = BackTransitionModel(self.args.belief_size, self.args.state_size, self.action_size, self.args.hidden_size, self.args.embedding_size).cuda()
        
        if self.args.inv_posterior_loss:
            self.inv_posterior_model = InverseDynamicsModel(self.args.state_size, self.action_size, self.args.embedding_size).cuda()
        if self.args.inv_belief_loss:
            self.inv_belief_model = InverseDynamicsModel(self.args.belief_size, self.action_size, self.args.embedding_size).cuda()
        if self.args.inv_belief_post_loss:
            self.inv_belief_post_model = InverseDynamicsModel_merge(self.args.belief_size+self.args.state_size, self.action_size, self.args.embedding_size).cuda()
        
        if self.args.contrast_belief_post_loss:
            self.contrast_belief_post_model = ContrastiveObsModel(self.args.belief_size+self.args.state_size, self.args.embedding_size, self.args.dense_activation_function).cuda()
        if self.args.contrast_post_loss:
            self.contrast_post_model = ContrastiveObsModel(self.args.state_size, self.args.embedding_size, self.args.dense_activation_function).cuda()
        if self.args.contrast_belief_loss:
            self.contrast_belief_model = ContrastiveObsModel(self.args.belief_size, self.args.embedding_size, self.args.dense_activation_function).cuda()
        if self.args.contrast_encoder_loss:
            self.contrast_encoder_model = ContrastiveObsModel(self.args.embedding_size, self.args.embedding_size, self.args.dense_activation_function).cuda()
        if self.args.target_transition_loss:
            self.contrast_transition_model = ContrastiveObsModel(self.args.belief_size+self.args.state_size, self.args.belief_size+self.args.state_size, self.args.dense_activation_function).cuda()

        self.actor_model = ActorModel(self.args.belief_size, self.args.state_size, self.args.hidden_size, self.action_size, self.args.dense_activation_function).cuda()
        self.value_model = ValueModel(self.args.belief_size, self.args.state_size, self.args.hidden_size, self.args.dense_activation_function).cuda()

        self.param_list = \
            list(self.transition_model.parameters()) + \
            list(self.reward_model.parameters()) + \
            list(self.encoder.parameters())
        
        if self.args.contrast_belief_post_loss:
            self.param_list += list(self.contrast_belief_post_model.parameters())
        
        if self.args.observation_loss:
            self.param_list += list(self.observation_model.parameters())
        if self.args.back_transition_loss:
            self.param_list += list(self.back_transition_model.parameters())
        
        if self.args.inv_posterior_loss:
            self.param_list += list(self.inv_posterior_model.parameters())
        if self.args.inv_belief_loss:
            self.param_list += list(self.inv_belief_model.parameters())
        if self.args.inv_belief_post_loss:
            self.param_list += list(self.inv_belief_post_model.parameters())
        
        if self.args.contrast_post_loss:
            self.param_list += list(self.contrast_post_model.parameters())
        if self.args.contrast_belief_loss:
            self.param_list += list(self.contrast_belief_model.parameters())
        if self.args.contrast_encoder_loss:
            self.param_list += list(self.contrast_encoder_model.parameters())
        if self.args.target_transition_loss:
            self.param_list += list(self.contrast_transition_model.parameters())


        if self.args.use_adv_loss:
            self.D_param_list = list(self.discriminator.parameters())
            self.D_optimizer = torch.optim.Adam(self.D_param_list, lr=2e-4, betas=(0.5, 0.999))

        self.value_actor_param_list = list(self.value_model.parameters()) + list(self.actor_model.parameters())
        self.params_list = self.param_list + self.value_actor_param_list

        self.model_optimizer = torch.optim.Adam(self.param_list, lr=0 if self.args.learning_rate_schedule != 0 else self.args.model_learning_rate, eps=self.args.adam_epsilon)
        self.actor_optimizer = torch.optim.Adam(self.actor_model.parameters(), lr=0 if self.args.learning_rate_schedule != 0 else self.args.actor_learning_rate, eps=self.args.adam_epsilon)
        self.value_optimizer = torch.optim.Adam(self.value_model.parameters(), lr=0 if self.args.learning_rate_schedule != 0 else self.args.value_learning_rate, eps=self.args.adam_epsilon)

        self.global_prior = Normal(torch.zeros(self.args.dreamer_batch_size, self.args.state_size).cuda(), torch.ones(self.args.dreamer_batch_size, self.args.state_size).cuda())  # Global prior N(0, I)
        self.args.free_nats = torch.full((1, ), self.args.free_nats).cuda()  # Allowed deviation in KL divergence
        self.train_steps = args.train_steps
        self.episodes = self.train_steps // (self.args.episode_length //self.args.action_repeat)
        print("Num of episode : ", self.episodes)
        
        if checking:
            print_info()
        self.planner = self.actor_model

        self.train()
        
        # work_dir = os.path.join('results', '{}_{}'.format(args.domain_name, args.task_name))

    def load_model(self, model_path):
         
        model_dicts = torch.load(model_path)
        self.transition_model.load_state_dict(model_dicts['transition_model'])
        self.reward_model.load_state_dict(model_dicts['reward_model'])
        self.encoder.load_state_dict(model_dicts['encoder'])
        self.actor_model.load_state_dict(model_dicts['actor_model'])
        self.value_model.load_state_dict(model_dicts['value_model'])
        self.model_optimizer.load_state_dict(model_dicts['model_optimizer'])
        if self.args.contrast_belief_post_loss:
            self.contrast_belief_post_model.load_state_dict(model_dicts['contrast_belief_post_model'])
        if self.args.observation_loss:
            self.observation_model.load_state_dict(model_dicts['observation_model'])
        if self.args.back_transition_loss:
            self.back_transition_model.load_state_dict(model_dicts['back_transition_model'])
        if self.args.inv_posterior_loss:
            self.inv_posterior_model.load_state_dict(model_dicts['inv_posterior_model'])
        if self.args.inv_belief_loss:
            self.inv_belief_model.load_state_dict(model_dicts['inv_belief_model'])
        if self.args.inv_belief_post_loss:
            self.inv_belief_post_model.load_state_dict(model_dicts['inv_belief_post_model'])
        if self.args.contrast_post_loss:
            self.contrast_post_model.load_state_dict(model_dicts['contrast_post_model'])
        if self.args.contrast_belief_loss:
            self.contrast_belief_model.load_state_dict(model_dicts['contrast_belief_model'])
        if self.args.contrast_encoder_loss:
            self.contrast_encoder_model.load_state_dict(model_dicts['contrast_encoder_model'])
        if self.args.target_transition_loss:
            self.contrast_transition_model.load_state_dict(model_dicts['contrast_transition_model'])

    def soft_update_params(self, net, target_net, tau):
      for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(
                tau * param.data + (1 - tau) * target_param.data
            )
    
    def hard_update_params(self, net, target_net):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(param.data)
        
    def train(self, training=True):
        self.transition_model.train()
        self.reward_model.train()
        self.encoder.train()
        self.actor_model.train()
        self.value_model.train()

        if self.args.contrast_belief_post_loss:
            self.contrast_belief_post_model.train()

        if self.args.observation_loss:
            self.observation_model.train()
        if self.args.back_transition_loss:
            self.back_transition_model.train()
        if self.args.inv_posterior_loss:
            self.inv_posterior_model.train()
        if self.args.inv_belief_loss:
            self.inv_belief_model.train()
        if self.args.inv_belief_post_loss:
            self.inv_belief_post_model.train()
        if self.args.contrast_post_loss:
            self.contrast_post_model.train()
        if self.args.contrast_belief_loss:
            self.contrast_belief_model.train()
        if self.args.contrast_encoder_loss:
            self.contrast_encoder_model.train()
        if self.args.target_transition_loss:
            self.contrast_transition_model.train()

    def eval(self):
        # Set models to eval mode
        self.transition_model.eval()
        self.reward_model.eval() 
        self.encoder.eval()
        self.actor_model.eval()
        self.value_model.eval()

        if self.args.contrast_belief_post_loss:
            self.contrast_belief_post_model.eval()

        if self.args.observation_loss:
            self.observation_model.eval()
        if self.args.back_transition_loss:
            self.back_transition_model.eval()
        if self.args.inv_posterior_loss:
            self.inv_posterior_model.eval()
        if self.args.inv_belief_loss:
            self.inv_belief_model.eval()
        if self.args.inv_belief_post_loss:
            self.inv_belief_post_model.eval()
        if self.args.contrast_post_loss:
            self.contrast_post_model.eval()
        if self.args.contrast_belief_loss:
            self.contrast_belief_model.eval()
        if self.args.contrast_encoder_loss:
            self.contrast_encoder_model.eval()
        if self.args.target_transition_loss:
            self.contrast_transition_model.eval()

    def only_encoder(self, observation):
        with torch.no_grad():
            observation = torch.as_tensor(np.array(observation).astype(np.float32)).cuda()
            utils.preprocess_observation_(observation, self.args.bit_depth)
            return self.encoder(observation.unsqueeze(dim=0)).cpu().numpy()

    def update_belief_and_act(self, env, belief, posterior_state, action, observation, explore=False):
        # Infer belief over current state q(s_t|o≤t,a<t) from the history
        # print("action size: ",action.size()) torch.Size([1, 6])
        if isinstance(env, utils.EnvBatcher):
            observation = torch.as_tensor(observation).cuda()
            utils.preprocess_observation_(observation, self.args.bit_depth)
            belief, _, _, _, posterior_state, _, _ = \
                self.transition_model(posterior_state, action.unsqueeze(dim=0), belief, self.encoder(observation).unsqueeze(dim=0))  # Action and observation need extra time dimension
        else:
            observation = torch.as_tensor(np.array(observation).astype(np.float32)).cuda()
            utils.preprocess_observation_(observation, self.args.bit_depth)
            belief, _, _, _, posterior_state, _, _ = \
                self.transition_model(posterior_state, action.unsqueeze(dim=0), belief, self.encoder(observation.unsqueeze(dim=0)).unsqueeze(dim=0))  # Action and observation need extra time dimension
        belief, posterior_state = belief.squeeze(dim=0), posterior_state.squeeze(dim=0)  # Remove time dimension from belief/state
        action = self.planner.get_action(belief, posterior_state, det=not(explore))
        if explore:
            action = torch.clamp(Normal(action, self.args.action_noise).rsample(), -1, 1) # Add gaussian exploration noise on top of the sampled action
            # action = action + self.args.action_noise * torch.randn_like(action)  # Add exploration noise ε ~ p(ε) to the action
        
        # Perform environment step (action repeats handled internally)
        next_observation, reward, done, _ = env.step(action.detach().cpu().numpy() if isinstance(env, utils.EnvBatcher) else action[0].detach().cpu().numpy())
        
        # next_observation = torch.as_tensor(np.array(next_observation).astype(np.float32))
        # utils.preprocess_observation_(next_observation, self.args.bit_depth)
        
        return belief, posterior_state, action, next_observation, reward, done

    def init_trajectories(self, env, replay_buffer):
        
        for s in range(1, self.args.seed_episodes + 1):
            observation, done, t = env.reset(), False, 0

            while not done:
                action = env.action_space.sample()
                next_observation, reward, done, _ = env.step(action)

                replay_buffer.add(observation, action, reward, next_observation, done)
                observation = next_observation
                t += 1
            
            self.metrics['steps'].append(t + ( 0 if len(self.metrics['steps']) == 0 else self.metrics['steps'][-1]))
            self.metrics['episodes'].append(s)
            print("episode : ",self.metrics['episodes'][-1])
            print("steps : ",self.metrics['steps'][-1])

    def lambda_return(self, imged_reward, value_pred, bootstrap, discount=0.99, lambda_=0.95):
        # Setting lambda=1 gives a discounted Monte Carlo return.
        # Setting lambda=0 gives a fixed 1-step return.
        next_values = torch.cat([value_pred[1:], bootstrap[None]], 0)
        discount_tensor = discount * torch.ones_like(imged_reward) #pcont
        inputs = imged_reward + discount_tensor * next_values * (1 - lambda_)
        last = bootstrap
        indices = reversed(range(len(inputs)))
        outputs = []
        for index in indices:
            inp, disc = inputs[index], discount_tensor[index]
            last = inp + disc*lambda_*last
            outputs.append(last)
        outputs = list(reversed(outputs))
        outputs = torch.stack(outputs, 0)
        returns = outputs
        return returns

    def imagine_ahead(self, prev_state, prev_belief, planning_horizon=12):
        '''
        imagine_ahead is the function to draw the imaginary tracjectory using the dynamics model, actor, critic.
        Input: current state (posterior), current belief (hidden), policy, transition_model  # torch.Size([50, 30]) torch.Size([50, 200]) 
        Output: generated trajectory of features includes beliefs, prior_states, prior_means, prior_std_devs
                torch.Size([49, 50, 200]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30])
        '''
        flatten = lambda x: x.view([-1]+list(x.size()[2:]))
        prev_belief = flatten(prev_belief)
        prev_state = flatten(prev_state)
        
        # Create lists for hidden states (cannot use single tensor as buffer because autograd won't work with inplace writes)
        T = planning_horizon
        beliefs, prior_states, prior_means, prior_std_devs = [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T
        beliefs[0], prior_states[0] = prev_belief, prev_state
        
        # Loop over time sequence
        for t in range(T - 1):
            _state = prior_states[t]
            actions = self.planner.get_action(beliefs[t].detach(),_state.detach())
            # Compute belief (deterministic hidden state)
            hidden = self.transition_model.act_fn(self.transition_model.fc_embed_state_action(torch.cat([_state, actions], dim=1)))
            beliefs[t + 1] = self.transition_model.rnn(hidden, beliefs[t])
            # Compute state prior by applying transition dynamics
            hidden = self.transition_model.act_fn(self.transition_model.fc_embed_belief_prior(beliefs[t + 1]))
            prior_means[t + 1], _prior_std_dev = torch.chunk(self.transition_model.fc_state_prior(hidden), 2, dim=1)
            prior_std_devs[t + 1] = F.softplus(_prior_std_dev) + self.transition_model.min_std_dev
            prior_states[t + 1] = prior_means[t + 1] + prior_std_devs[t + 1] * torch.randn_like(prior_means[t + 1])     
        # Return new hidden states
        # imagined_traj = [beliefs, prior_states, prior_means, prior_std_devs]
        imagined_traj = [torch.stack(beliefs[1:], dim=0), torch.stack(prior_states[1:], dim=0), torch.stack(prior_means[1:], dim=0), torch.stack(prior_std_devs[1:], dim=0)]
        return imagined_traj

    def dreamer_train(self, L, start_time, env, test_env_lst, replay_buffer, work_dir):
        self.writer = SummaryWriter(os.path.join(work_dir, 'logs'))
        print("Get initial trajectories for {} episodes".format(self.args.seed_episodes))
        self.init_trajectories(env, replay_buffer)
        print("Train start")
        
        for episode in range(self.metrics['episodes'][-1] + 1, self.episodes+1):
            
            if self.args.log_dir != 'logs/debug' and (self.metrics['steps'][-1] % self.args.eval_freq == 0 or episode == self.args.seed_episodes+1):
                print(work_dir)
                self.eval()
                L.log('eval/episode', episode, self.metrics['steps'][-1])
                
                if len(test_env_lst) == 1:
                    d_t1 = self.dreamer_eval(L, test_env_lst[0], episode, work_dir, "de")
                    print("Duration time {} ".format(np.round(d_t1,3)))
                else:
                    d_t1 = self.dreamer_eval(L, test_env_lst[0], episode, work_dir, "de")
                    d_t2 = self.dreamer_eval(L, test_env_lst[1], episode, work_dir, "ce")
                    d_t2 = self.dreamer_eval(L, test_env_lst[2], episode, work_dir, "ch")
                    d_t3 = self.dreamer_eval(L, test_env_lst[3], episode, work_dir, "ve")
                    d_t4 = self.dreamer_eval(L, test_env_lst[4], episode, work_dir, "vh")

                    print("Duration time {} {} {} {}".format(np.round(d_t1,3), np.round(d_t2,3), np.round(d_t3,3), np.round(d_t4,3)))
                # d_t6 = self.dreamer_eval(L, test_env_lst[5], episode, work_dir, "vn")
                L.dump(self.metrics['steps'][-1])
                if self.metrics['steps'][-1] % self.args.save_freq == 0:
                    self.model_save(work_dir, episode)
                self.train()

            start_time = time.time()
            
            # Model fitting
            losses = []
            model_modules = \
                self.transition_model.modules+\
                self.encoder.modules+\
                self.reward_model.modules

            if self.args.contrast_belief_post_loss:
                model_modules += self.contrast_belief_post_model.modules
            
            if self.args.observation_loss:
                model_modules += self.observation_model.modules
            if self.args.back_transition_loss:
                model_modules += self.back_transition_model.modules
            if self.args.inv_posterior_loss:
                model_modules += self.inv_posterior_model.modules
            if self.args.inv_belief_loss:
                model_modules += self.inv_belief_model.modules
            if self.args.inv_belief_post_loss:
                model_modules += self.inv_belief_post_model.modules
            if self.args.contrast_post_loss:
                model_modules += self.contrast_post_model.modules
            if self.args.contrast_belief_loss:
                model_modules += self.contrast_belief_model.modules        
            if self.args.contrast_encoder_loss:
                model_modules += self.contrast_encoder_model.modules
            if self.args.target_transition_loss:
                model_modules += self.contrast_transition_model.modules

            # print("training loop")
            self.update_with_trajectories(replay_buffer, model_modules, losses)
            
            # Update and plot loss self.metrics
            losses = tuple(zip(*losses))
            self.metrics['reward_loss'].append(losses[0])
            self.metrics['kl_loss'].append(losses[1])
            self.metrics['actor_loss'].append(losses[2])
            self.metrics['value_loss'].append(losses[3])
            if self.args.contrast_belief_post_loss:
                self.metrics['contrast_belief_post_loss'].append(losses[4])
            if self.args.observation_loss:
                self.metrics['observation_loss'].append(losses[5])
            if self.args.back_transition_loss:
                self.metrics['back_transition_loss'].append(losses[6])
            if self.args.inv_posterior_loss:
                self.metrics['inv_posterior_loss'].append(losses[7])
            if self.args.inv_belief_loss:
                self.metrics['inv_belief_loss'].append(losses[8])
            if self.args.inv_belief_post_loss:
                self.metrics['inv_belief_post_loss'].append(losses[9])
            if self.args.contrast_post_loss:
                self.metrics['contrast_post_loss'].append(losses[10])
            if self.args.contrast_belief_loss:
                self.metrics['contrast_belief_loss'].append(losses[11])
            if self.args.contrast_encoder_loss:
                self.metrics['contrast_encoder_loss'].append(losses[12])
            
            # Data collection
            with torch.no_grad():
                observation, total_reward = env.reset(), 0
            
                belief, posterior_state, action = \
                    torch.zeros(1, self.args.belief_size).cuda(), \
                    torch.zeros(1, self.args.state_size).cuda(), \
                    torch.zeros(1, self.action_size).cuda()
                
                # pbar = tqdm(range(self.args.episode_length // self.args.action_repeat))
                for step in range(1, (self.args.episode_length // self.args.action_repeat)+1):
                    
                    original_observation = observation
                    
                    belief, posterior_state, action, next_observation, reward, done = self.update_belief_and_act(env, belief, posterior_state, action, observation, explore=True)
                    
                    replay_buffer.add(original_observation, action.cpu(), reward, next_observation, done)
                    total_reward += reward
                    observation = next_observation # next_observation is original next_observation

                # Update and plot train reward self.metrics
                self.metrics['steps'].append(step + self.metrics['steps'][-1])
                self.metrics['episodes'].append(episode)
                self.metrics['train_rewards'].append(total_reward)
                
                L.log('train/duration', time.time() - start_time, self.metrics['steps'][-1])
                L.log('train/episode_reward', total_reward, self.metrics['steps'][-1])
                L.log('train/episode', episode, self.metrics['steps'][-1])
                L.dump(self.metrics['steps'][-1])
                
            # Test model
            self.writer.add_scalar("train_reward", self.metrics['train_rewards'][-1], self.metrics['steps'][-1])
            self.writer.add_scalar("train/episode_reward", self.metrics['train_rewards'][-1], self.metrics['steps'][-1]*self.args.action_repeat)
            self.writer.add_scalar("reward_loss", self.metrics['reward_loss'][0][-1], self.metrics['steps'][-1])
            self.writer.add_scalar("kl_loss", self.metrics['kl_loss'][0][-1], self.metrics['steps'][-1])
            self.writer.add_scalar("actor_loss", self.metrics['actor_loss'][0][-1], self.metrics['steps'][-1])
            self.writer.add_scalar("value_loss", self.metrics['value_loss'][0][-1], self.metrics['steps'][-1])  
            
            if self.args.contrast_belief_post_loss:
                self.writer.add_scalar("contrast_belief_post_loss", self.metrics['contrast_belief_post_loss'][0][-1], self.metrics['steps'][-1])

            if self.args.observation_loss:
                self.writer.add_scalar("observation_loss", self.metrics['observation_loss'][0][-1], self.metrics['steps'][-1])
            if self.args.back_transition_loss:
                self.writer.add_scalar("back_transition_loss", self.metrics['back_transition_loss'][0][-1], self.metrics['steps'][-1])
            if self.args.inv_posterior_loss:
                self.writer.add_scalar("inv_posterior_loss", self.metrics['inv_posterior_loss'][0][-1], self.metrics['steps'][-1])
            if self.args.inv_belief_loss:
                self.writer.add_scalar("inv_belief_loss", self.metrics['inv_belief_loss'][0][-1], self.metrics['steps'][-1])
            if self.args.inv_belief_post_loss:
                self.writer.add_scalar("inv_belief_post_loss", self.metrics['inv_belief_post_loss'][0][-1], self.metrics['steps'][-1])
            if self.args.contrast_post_loss:
                self.writer.add_scalar("contrast_post_loss", self.metrics['contrast_post_loss'][0][-1], self.metrics['steps'][-1])
            if self.args.contrast_belief_loss:
                self.writer.add_scalar("contrast_belief_loss", self.metrics['contrast_belief_loss'][0][-1], self.metrics['steps'][-1])
            if self.args.contrast_encoder_loss:
                self.writer.add_scalar("contrast_encoder_loss", self.metrics['contrast_encoder_loss'][0][-1], self.metrics['steps'][-1])

    def model_save(self, work_dir, episode):
        model_save_dict = {
            'transition_model': self.transition_model.state_dict(),
            'reward_model': self.reward_model.state_dict(),
            'encoder': self.encoder.state_dict(),
            'actor_model': self.actor_model.state_dict(),
            'value_model': self.value_model.state_dict(),
            'model_optimizer': self.model_optimizer.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'value_optimizer': self.value_optimizer.state_dict()
            }
        if self.args.contrast_belief_post_loss:
            model_save_dict['contrast_belief_post_model'] = self.contrast_belief_post_model.state_dict()
        if self.args.observation_loss:
            model_save_dict['observation_model'] = self.observation_model.state_dict()
        if self.args.back_transition_loss:
            model_save_dict['back_transition_model'] = self.back_transition_model.state_dict()
        if self.args.inv_posterior_loss:
            model_save_dict['inv_posterior_model'] = self.inv_posterior_model.state_dict()
        if self.args.inv_belief_loss:
            model_save_dict['inv_belief_model'] = self.inv_belief_model.state_dict()
        if self.args.inv_belief_post_loss:
            model_save_dict['inv_belief_post_model'] = self.inv_belief_post_model.state_dict()
        if self.args.contrast_post_loss:
            model_save_dict['contrast_post_model'] = self.contrast_post_model.state_dict()
        if self.args.contrast_belief_loss:
            model_save_dict['contrast_belief_model'] = self.contrast_belief_model.state_dict()
        if self.args.contrast_encoder_loss:
            model_save_dict['contrast_encoder_model'] = self.contrast_encoder_model.state_dict()
        if self.args.target_transition_loss:
            model_save_dict['contrast_transition_model'] = self.contrast_transition_model.state_dict()
            
        torch.save(model_save_dict, os.path.join(os.path.join(work_dir, 'model'), 'models_%d.pth' % episode))

        
    def update_with_trajectories(self, replay_buffer, model_modules, losses):
        
        for s in range(self.args.collect_interval):
            # Draw sequence chunks {(o_t, a_t, r_t+1, terminal_t+1)} ~ D uniformly at random from the dataset (including terminal flags)
            observations, aug_observations, actions, rewards, nonterminals = replay_buffer.drg_sample(self.args.dreamer_batch_size, self.args.chunk_size) # Transitions start at time t = 0

            # self.encoder, self.target_encoder
            encoded_observations = bottle(self.encoder, (observations, ) if not self.args.aug_loss else (aug_observations, )) 
            with torch.no_grad():
                target_encoded_observations = bottle(self.target_encoder, (aug_observations, ) if not self.args.aug_loss else (observations, ))
            
            # Create initial belief and state for time t = 0
            init_belief, init_state = torch.zeros(self.args.dreamer_batch_size, self.args.belief_size).cuda(), torch.zeros(self.args.dreamer_batch_size, self.args.state_size).cuda()
            
            # Update belief/state using posterior from previous belief/state, previous action and current observation (over entire sequence at once)
            # original dreamer
            beliefs, prior_states, prior_means, prior_std_devs, posterior_states, posterior_means, posterior_std_devs \
                = self.transition_model(init_state, actions[:-1], init_belief, encoded_observations[1:], nonterminals[:-1])
            
            if self.args.target_transition_loss:
                with torch.no_grad():
                    target_beliefs, target_prior_states, target_prior_means, target_prior_std_devs, \
                    target_posterior_states, target_posterior_means, target_posterior_std_devs \
                        = self.target_transition_model(init_state, actions[:-1], init_belief, target_encoded_observations[1:], nonterminals[:-1])
            
            if self.args.observation_loss:
                if self.args.worldmodel_LogProbLoss:
                    observation_dist = Normal(bottle(self.observation_model, (beliefs, posterior_states)), 1)
                    observation_loss = -observation_dist.log_prob(observations[1:]).sum((2, 3, 4)).mean(dim=(0, 1))
                else:
                    observation_loss = F.mse_loss(
                        bottle(self.observation_model, (beliefs, posterior_states)), observations[1:], reduction='none'
                        ).sum((2, 3, 4)).mean(dim=(0, 1))
            
            if self.args.worldmodel_LogProbLoss:
                reward_dist = Normal(bottle(self.reward_model, (beliefs, posterior_states)),1)
                reward_loss = -reward_dist.log_prob(rewards[:-1]).mean(dim=(0, 1))
            else:
                reward_loss = F.mse_loss(bottle(self.reward_model, (beliefs, posterior_states)), rewards[:-1], reduction='none').mean(dim=(0,1))
            
            if self.args.back_transition_loss:
                back_beliefs, back_prior_states, back_prior_means, back_prior_std_devs, back_posterior_states, back_posterior_means, back_posterior_std_devs \
                = self.back_transition_model(posterior_states[-1].detach(), torch.flipud(actions[1:]).detach(), beliefs[-1].detach(), torch.flipud(encoded_observations[:-1]).detach(), torch.flipud(nonterminals[1:]).detach())
                # predict_before_embedding = bottle(self.back_transition_model, (beliefs, posterior_states, actions[:-1]))
                # back_transition_loss = F.mse_loss(predict_before_embedding, encoded_observations[:-1], reduction='none').mean(dim=(0, 1, 2))
                back_div = kl_divergence(Normal(back_posterior_means, back_posterior_std_devs), Normal(back_prior_means, back_prior_std_devs)).sum(dim=2)
                back_transition_loss = torch.max(back_div, self.args.free_nats).mean(dim=(0, 1))  # Note that normalisation by overshooting distance and weighting by overshooting distance cancel out
                
                back_posterior_means = torch.flipud(back_posterior_means)
                back_posterior_std_devs = torch.flipud(back_posterior_std_devs)
                back_posterior_states = torch.flipud(back_posterior_states)
            
                transition_div = kl_divergence(Normal(posterior_means.detach(), posterior_std_devs.detach()), Normal(back_posterior_means, back_posterior_std_devs)).sum(dim=2)
                transition_kl_loss = torch.max(transition_div, self.args.free_nats).mean(dim=(0, 1))  # Note that normalisation by overshooting distance and weighting by overshooting distance cancel out
                back_transition_loss += transition_kl_loss
                back_transition_loss += 1 - F.cosine_similarity(posterior_means.detach(), back_posterior_means).mean(dim=(0, 1))                
            
            if self.args.inv_posterior_loss: 
                posterior_pred_action = bottle(self.inv_posterior_model, (posterior_states[:-1], posterior_states[1:]))
                inv_posterior_loss = F.mse_loss(posterior_pred_action, actions[1:-1], reduction='none').mean(dim=(0, 1, 2))
            if self.args.inv_belief_loss:
                belief_pred_action = bottle(self.inv_belief_model, (beliefs[:-1], beliefs[1:]))
                inv_belief_loss = F.mse_loss(belief_pred_action, actions[1:-1], reduction='none').mean(dim=(0, 1, 2))
            if self.args.inv_belief_post_loss:
                # CROSS
                if self.args.cross:
                    belief_post_pred_action = bottle(self.inv_belief_post_model, (torch.cat([beliefs[:-1], posterior_states[:-1]],dim=-1), torch.cat([beliefs[1:], prior_states[1:]],dim=-1)))
                    inv_belief_post_loss = F.mse_loss(belief_post_pred_action, actions[1:-1], reduction='none').mean(dim=(0, 1, 2))
                    belief_prior_pred_action = bottle(self.inv_belief_post_model, (torch.cat([beliefs[:-1], prior_states[:-1]],dim=-1), torch.cat([beliefs[1:], posterior_states[1:]],dim=-1)))
                    inv_belief_post_loss += F.mse_loss(belief_prior_pred_action, actions[1:-1], reduction='none').mean(dim=(0, 1, 2))
                    inv_belief_post_loss = inv_belief_post_loss / 2.
                # ORIGINAL
                else:
                    belief_post_pred_action = bottle(self.inv_belief_post_model, (torch.cat([beliefs[:-1], posterior_states[:-1]],dim=-1), torch.cat([beliefs[1:], posterior_states[1:]],dim=-1)))
                    inv_belief_post_loss = F.mse_loss(belief_post_pred_action, actions[1:-1], reduction='none').mean(dim=(0, 1, 2))

            if self.args.contrast_belief_post_loss:
                if self.args.contrast_obs_target_loss:
                    contrast_belief_post_loss = self.contrast_belief_post_model(torch.cat([beliefs, posterior_states],dim=-1), target_encoded_observations[1:])
                else:
                    contrast_belief_post_loss = self.contrast_belief_post_model(torch.cat([beliefs, posterior_states],dim=-1), encoded_observations[1:].detach())

            if self.args.contrast_post_loss:
                contrast_post_loss = self.contrast_post_model(posterior_states, target_encoded_observations[1:])
            if self.args.contrast_belief_loss:
                contrast_belief_loss = self.contrast_belief_model(beliefs, target_encoded_observations[1:])
            
            if self.args.contrast_encoder_loss:
                contrast_encoder_loss = self.contrast_encoder_model(encoded_observations, target_encoded_observations)
            
            if self.args.target_transition_loss:
                contrast_transition_loss = self.contrast_transition_model(torch.cat([beliefs, posterior_states],dim=-1), torch.cat([target_beliefs, target_posterior_states],dim=-1))

            # transition loss
            if self.args.use_adv_loss:
                valid = Variable(self.Tensor((self.args.dreamer_batch_size-1)*self.args.chunk_size, 1).fill_(1.0), requires_grad=False)
                fake = Variable(self.Tensor((self.args.dreamer_batch_size-1)*self.args.chunk_size, 1).fill_(0.0), requires_grad=False)
                adv_loss = self.adversarial_loss(self.discriminator(prior_states), valid)
                kl_loss = adv_loss
            else:
                if self.args.use_kl_balancing:
                    kl_lhs = kl_divergence(Normal(posterior_means.detach(), posterior_std_devs.detach()), Normal(prior_means, prior_std_devs)).sum(dim=2)
                    kl_rhs = kl_divergence(Normal(posterior_means, posterior_std_devs), Normal(prior_means.detach(), prior_std_devs.detach())).sum(dim=2)
                    if self.args.use_free_nats:
                        kl_lhs = torch.max(kl_lhs, self.args.free_nats).mean(dim=(0, 1))  # Note that normalisation by overshooting distance and weighting by overshooting distance cancel out
                        kl_rhs = torch.max(kl_rhs, self.args.free_nats).mean(dim=(0, 1))  # Note that normalisation by overshooting distance and weighting by overshooting distance cancel out
                    else:
                        kl_lhs = torch.mean(kl_lhs)
                        kl_rhs = torch.mean(kl_rhs)
                    kl_loss = self.args.kl_balance_scale*kl_lhs + (1-self.args.kl_balance_scale)*kl_rhs
                else:
                    div = kl_divergence(Normal(posterior_means, posterior_std_devs), Normal(prior_means, prior_std_devs)).sum(dim=2)
                    if self.args.use_free_nats:
                        kl_loss = torch.max(div, self.args.free_nats).mean(dim=(0, 1))  # Note that normalisation by overshooting distance and weighting by overshooting distance cancel out
                    else:
                        kl_loss = torch.mean(div)
                kl_loss = self.args.kl_loss_scaling * kl_loss

            if self.args.global_kl_beta != 0:
                kl_loss += self.args.global_kl_beta * kl_divergence(
                    Normal(posterior_means, posterior_std_devs), self.global_prior
                    ).sum(dim=2).mean(dim=(0, 1))
            # Calculate latent overshooting objective for t > 0
            if self.args.overshooting_kl_beta != 0:
                overshooting_vars = []  # Collect variables for overshooting to process in batch
                for t in range(1, self.args.chunk_size - 1):
                    d = min(t + self.args.overshooting_distance, self.args.chunk_size - 1)  # Overshooting distance
                    t_, d_ = t - 1, d - 1  # Use t_ and d_ to deal with different time indexing for latent states
                    seq_pad = (0, 0, 0, 0, 0, t - d + self.args.overshooting_distance)  # Calculate sequence padding so overshooting terms can be calculated in one batch
                    # Store (0) actions, (1) nonterminals, (2) rewards, (3) beliefs, (4) prior states, (5) posterior means, (6) posterior standard deviations and (7) sequence masks
                    overshooting_vars.append(
                        (
                            F.pad(actions[t:d], seq_pad), 
                            F.pad(nonterminals[t:d], seq_pad), 
                            F.pad(rewards[t:d], seq_pad[2:]), 
                            beliefs[t_], 
                            prior_states[t_], 
                            F.pad(posterior_means[t_ + 1:d_ + 1].detach(), seq_pad), 
                            F.pad(posterior_std_devs[t_ + 1:d_ + 1].detach(), seq_pad, value=1), 
                            F.pad(torch.ones(d - t, self.args.dreamer_batch_size, self.args.state_size).cuda(), seq_pad)))  # Posterior standard deviations must be padded with > 0 to prevent infinite KL divergences
                overshooting_vars = tuple(zip(*overshooting_vars))
                # Update belief/state using prior from previous belief/state and previous action (over entire sequence at once)
                beliefs, prior_states, prior_means, prior_std_devs = \
                    self.transition_model(
                        torch.cat(overshooting_vars[4], dim=0), 
                        torch.cat(overshooting_vars[0], dim=1), 
                        torch.cat(overshooting_vars[3], dim=0), 
                        None, 
                        torch.cat(overshooting_vars[1], dim=1))
                seq_mask = torch.cat(overshooting_vars[7], dim=1)
                # Calculate overshooting KL loss with sequence mask
                # Update KL loss (compensating for extra average over each overshooting/open loop sequence) 
                kl_loss += (1 / self.args.overshooting_distance) * \
                    self.args.overshooting_kl_beta * \
                    torch.max(
                        (kl_divergence(
                            Normal(torch.cat(overshooting_vars[5], dim=1), torch.cat(overshooting_vars[6], dim=1)), 
                            Normal(prior_means, prior_std_devs)) * seq_mask
                        ).sum(dim=2), self.args.free_nats).mean(dim=(0, 1)) * (self.args.chunk_size - 1)  
                # Calculate overshooting reward prediction loss with sequence mask
                if self.args.overshooting_reward_scale != 0: 
                    reward_loss += \
                        (1 / self.args.overshooting_distance) * \
                        self.args.overshooting_reward_scale * \
                        F.mse_loss(
                            bottle(
                                self.reward_model, 
                                (beliefs, prior_states)) * seq_mask[:, :, 0],
                                torch.cat(overshooting_vars[2], dim=1), 
                                reduction='none').mean(dim=(0, 1)) * (self.args.chunk_size - 1)  # Update reward loss (compensating for extra average over each overshooting/open loop sequence) 
            
            # Apply linearly ramping learning rate schedule
            if self.args.learning_rate_schedule != 0:
                for group in self.model_optimizer.param_groups:
                    group['lr'] = min(group['lr'] + self.args.model_learning_rate / self.args.learning_rate_schedule, self.args.model_learning_rate)
            
            
            model_loss = reward_loss + kl_loss

            if self.args.contrast_belief_post_loss:
                model_loss += contrast_belief_post_loss

            if self.args.observation_loss:
                model_loss += observation_loss
            if self.args.back_transition_loss:
                model_loss += back_transition_loss
                
            if self.args.inv_posterior_loss:
                model_loss += inv_posterior_loss
            if self.args.inv_belief_loss:
                model_loss += inv_belief_loss
            if self.args.inv_belief_post_loss:
                model_loss += inv_belief_post_loss

            if self.args.contrast_post_loss:
                model_loss += contrast_post_loss
            if self.args.contrast_belief_loss:
                model_loss += contrast_belief_loss

            if self.args.contrast_encoder_loss:
                model_loss += contrast_encoder_loss

            if self.args.target_transition_loss:
                model_loss += contrast_transition_loss

            # Update model parameters
            self.model_optimizer.zero_grad()
            model_loss.backward()
            nn.utils.clip_grad_norm_(self.param_list, self.args.grad_clip_norm, norm_type=2)
            self.model_optimizer.step()

            if self.args.use_adv_loss:
                # _, adv_prior_states, _, _, adv_posterior_states, _, _ \
                #     = self.transition_model(init_state, actions[:-1], init_belief, encoded_observations[1:].detach(), nonterminals[:-1])
                
                real_pred = self.discriminator(posterior_states.detach())
                fake_pred = self.discriminator(prior_states.detach())

                real_loss = self.adversarial_loss(real_pred - fake_pred.mean(0, keepdim=True), valid)
                fake_loss = self.adversarial_loss(fake_pred - real_pred.mean(0, keepdim=True), fake)
                d_loss = (real_loss + fake_loss) / 2

                self.D_optimizer.zero_grad()
                d_loss.backward()
                nn.utils.clip_grad_norm_(self.D_param_list, self.args.grad_clip_norm, norm_type=2)
                self.D_optimizer.step()

            if s % self.args.aux_update_freq == 0:
                self.soft_update_params(self.encoder, self.target_encoder, self.args.encoder_tau)
                if self.args.target_transition_loss:
                    self.soft_update_params(self.transition_model, self.target_transition_model, self.args.encoder_tau)
            
            #Dreamer implementation: actor loss calculation and optimization    
            with torch.no_grad():
                actor_states = posterior_states.detach()
                actor_beliefs = beliefs.detach()
            with utils.FreezeParameters(model_modules):
                imagination_traj = self.imagine_ahead(actor_states, actor_beliefs, self.args.planning_horizon)
            imged_beliefs, imged_prior_states, imged_prior_means, imged_prior_std_devs = imagination_traj
            with utils.FreezeParameters(model_modules + self.value_model.modules):
                imged_reward = bottle(self.reward_model, (imged_beliefs, imged_prior_states))
                value_pred = bottle(self.value_model, (imged_beliefs, imged_prior_states))
            returns = self.lambda_return(imged_reward, value_pred, bootstrap=value_pred[-1], discount=self.args.discount, lambda_=self.args.disclam)
            actor_loss = -torch.mean(returns)
            
            # Update model parameters
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor_model.parameters(), self.args.grad_clip_norm, norm_type=2)
            self.actor_optimizer.step()
            
            #Dreamer implementation: value loss calculation and optimization
            with torch.no_grad():
                value_beliefs = imged_beliefs.detach()
                value_prior_states = imged_prior_states.detach()
                target_return = returns.detach()
            value_dist = Normal(bottle(self.value_model, (value_beliefs, value_prior_states)),1) # detach the input tensor from the transition network.
            value_loss = -value_dist.log_prob(target_return).mean(dim=(0, 1)) 
            
            # Update model parameters
            self.value_optimizer.zero_grad()
            value_loss.backward()
            nn.utils.clip_grad_norm_(self.value_model.parameters(), self.args.grad_clip_norm, norm_type=2)
            self.value_optimizer.step()
            
            # # Store (0) reward loss (1) KL loss (2) actor loss (3) value loss
            if self.args.observation_loss: observation_loss = observation_loss.item()
            else: observation_loss = 0
            if self.args.back_transition_loss: back_transition_loss = back_transition_loss.item()
            else: back_transition_loss = 0
            
            if self.args.inv_posterior_loss: inv_posterior_loss = inv_posterior_loss.item()
            else: inv_posterior_loss = 0
            if self.args.inv_belief_loss: inv_belief_loss = inv_belief_loss.item()
            else: inv_belief_loss = 0
            if self.args.inv_belief_post_loss: inv_belief_post_loss = inv_belief_post_loss.item()
            else: inv_belief_post_loss = 0

            if self.args.contrast_belief_post_loss: contrast_belief_post_loss = contrast_belief_post_loss.item()
            else: contrast_belief_post_loss = 0

            if self.args.contrast_post_loss: contrast_post_loss = contrast_post_loss.item()
            else: contrast_post_loss = 0
            if self.args.contrast_belief_loss: contrast_belief_loss = contrast_belief_loss.item()
            else: contrast_belief_loss = 0
            if self.args.contrast_encoder_loss: contrast_encoder_loss = contrast_encoder_loss.item()
            else: contrast_encoder_loss = 0
            
            # if self.encoder_loss: encoder_loss = encoder_loss.item()
            # else: encoder_loss = 0

            losses.append([
                reward_loss.item(), kl_loss.item(), actor_loss.item(), value_loss.item(), contrast_belief_post_loss,\
                observation_loss, back_transition_loss, \
                inv_posterior_loss, inv_belief_loss, inv_belief_post_loss, \
                contrast_post_loss, contrast_belief_loss, contrast_encoder_loss
            ])
    
    def dreamer_eval(self, L, test_env, episode, work_dir, eval_mode):
        # L, test_env, episode, work_dir, eval_mode
        s_t = time.time()
        
        with torch.no_grad():
            episode_reward_lst = []
            for i in range(self.args.test_episodes):
                observation, episode_reward, video_frames = test_env.reset(), 0, []
                
                belief, posterior_state, action = \
                    torch.zeros(1, self.args.belief_size).cuda(), \
                    torch.zeros(1, self.args.state_size).cuda(), \
                    torch.zeros(1, self.action_size).cuda()

                for step in range(self.args.episode_length // self.args.action_repeat):
                    
                    belief, posterior_state, action, next_observation, reward, done = self.update_belief_and_act(test_env, belief, posterior_state, action, observation)

                    episode_reward += reward
                    
                    observation = next_observation

                episode_reward_lst.append(episode_reward)

        total_rewards = np.mean(episode_reward_lst)
        # Update and plot reward self.metrics (and write video if applicable) and save self.metrics
        self.metrics['test_episodes_{}'.format(eval_mode)].append(episode)
        self.metrics['test_rewards_{}'.format(eval_mode)].append(total_rewards)
                
        L.log('eval/episode_reward_{}'.format(eval_mode), total_rewards, self.metrics['steps'][-1])
        return time.time()-s_t
    
    def print_info():
        print("\n\tDRG\n")
        print("ENV\t\t\t\t : ", self.args.domain_name+"_"+self.args.task_name)
        print("frame_stack\t\t\t : ", self.args.frame_stack)
        print("action_repeat\t\t\t : ", self.args.action_repeat)
        print("algorithm\t\t\t : ",self.args.algorithm)
        print("observation_loss\t\t : ",self.args.observation_loss)
        print("back_transition_loss\t\t : ",self.args.back_transition_loss)
        print("inv_posterior_loss\t\t : ",self.args.inv_posterior_loss)
        print("inv_belief_loss\t\t\t : ",self.args.inv_belief_loss)
        print("inv_belief_post_loss\t\t : ",self.args.inv_belief_post_loss)
        # print("encoder_loss\t\t\t : ",self.encoder_loss)
        print("contrast_post_loss\t\t : ",self.args.contrast_post_loss)
        print("contrast_belief_loss\t\t : ",self.args.contrast_belief_loss)
        print("contrast_belief_post_loss\t : ",self.args.contrast_belief_post_loss)
        print("contrast_obs_target_loss\t : ",self.args.contrast_obs_target_loss)
        print("contrast_encoder_loss\t\t : ",self.args.contrast_encoder_loss)
        print("aug_loss\t\t\t : ",self.args.aug_loss)
        print("same_action\t\t\t : ",self.args.same_action)
        print("worldmodel_LogProbLoss\t\t : ",self.args.worldmodel_LogProbLoss)
        print("latent_cosine_similarity\t : ",self.args.latent_cosine_similarity)
        
        print("\n#### augmenation #### ")
        print("random\t\t\t\t : ",self.args.ran)
        print("random_2\t\t\t : ",self.args.ran_2)
        print("random_3\t\t\t : ",self.args.ran_3)
        print("cutout_color\t\t\t : ",self.args.cc)
        print("color_jitter\t\t\t : ",self.args.cj)
        print("overlay\t\t\t\t : ",self.args.ol)
        print("random_conv\t\t\t : ",self.args.con)
        print("gray_scale\t\t\t : ",self.args.gs)
        print("planning_horizon\t\t : ",self.args.planning_horizon)
        print()