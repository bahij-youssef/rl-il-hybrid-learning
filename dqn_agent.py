import numpy as np
from collections import deque
from vizdoom import GameVariable
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.dqn_preprocessing import *
from replay import ReplayMemory

from collections import namedtuple
from tensorboardX import SummaryWriter

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'dones'))

class Agent:
    
    def __init__(self, possible_actions, scenario, max_size = 1000, stack_size = 3, batch_size = 64, resize = (60, 80)):
        """
        Description
        --------------
        Constructor of Agent class.
        
        Attributes
        --------------
        possible_actions : List, contains the one-hot encoded possible actions to take for the agent.
        scenario         : String, either 'basic' or 'deadly_corridor'
        memory           : String with values in ['uniform', 'prioritized'] depending on the type of replay memory to be used     (defaul='uniform')
        max_size         : Int, maximum size of the replay buffer (default=1000)
        stack_size       : Int, the number of frames to stack to create motion (default=4)
        batch_size       : Int, the batch size used for backpropagation (default=64)
        resize           : tuple, shape of the resized frame (default=(120,160))
        """
        
        self.memory = ReplayMemory(max_size)
        self.stack_size = stack_size
        self.possible_actions = possible_actions
        self.scenario = scenario
        self.batch_size = batch_size
        self.resize = resize
        self.t = transform((60,80),normalize=False)
        self.frames_deque = deque([],self.stack_size)
        
    def get_reward(self, variables_cur, variables_prev):
        """
        Description
        --------------
        Reward reshaping
        
        Parameters
        --------------
        variables_cur  : dict, dictionnary containing current variables (kills, health and ammo.
        variables_prev : dict, dictionnary containing previous variables (kills, health and ammo.
        
        Returns
        --------------
        Float, a reshaped reward.
        """
        
        r = 0
        if self.scenario == 'health_gathering_supreme':
            if variables_cur['health'] > variables_prev['health']:
                r += 5

            if variables_cur['health'] < variables_prev['health']:
                r -= 0.1
                
        elif self.scenario == 'deathmatch':
            r += (variables_cur['kills'] - variables_prev['kills'])*5
            if variables_cur['ammo'] < variables_prev['ammo']:
                r -= 0.1

            if variables_cur['health'] < variables_prev['health']:
                r -= 1
                
        return r
        
            
    def train(self, 
              game, 
              total_episodes = 100, 
              pretrain = 50, 
              frame_skip = 4, 
              lr = 1e-4, 
              max_tau = 100, 
              explore_start = 1.0, 
              explore_stop = 0.01, 
              decay_rate = 0.0001, 
              gamma = 0.99, 
              freq = 50, 
              init_zeros = False,
              bc_model=None):
        """
        Description
        --------------
        Unroll trajectories to gather experiences and train the model.
        
        Parameters
        --------------
        game               : VizDoom game instance.
        total_episodes     : Int, the number of training episodes (default=100)
        pretrain           : Int, the number of initial experiences to put in the replay buffer (default=100)
        frame_skip         : Int, the number of frames to repeat the action on (default=4)
        enhance            : String in ['none', 'dueling'] (default='none')
        lr                 : Float, the learning rate (default=1e-4)
        max_tau            : Int, number of steps to performe double q-learning parameters update (default=100)
        explore_start      : Float, the initial exploration probaboility (default=1.0)
        explore_stop       : Float, the final exploration probability (default=0.01)
        decay_rate         : Float, the decay rate of the exploration probability (default=1e-3)
        gamma              : Float, the reward discoundting coefficient, should be between 0 and 1 (default=0.99)
        freq               : Int, number of episodes to save model weights (default=50)
        init_zeros         : Boolean, whether to initialize the weights to zero or not.
        """
        # Create

        # Setting tensorboadX and variables of interest
        writer = SummaryWriter(log_dir = '/home/runs/' + self.scenario)
        kill_count = np.zeros(10) # This list will contain kill counts of each 10 episodes in order to compute moving average
        ammo = np.zeros(10) # This list will contain ammo of each 10 episodes in order to compute moving average
        rewards = np.zeros(10) # This list will contain rewards of each 10 episodes in order to compute moving average
        losses = np.zeros(10) # This list will contain losses of each 10 episodes in order to compute moving average
        # Pretraining phase
        game.init()
        game.new_episode()
        # Initialize current and previous game variables dictionnaries
        variables_cur = {'kills' : game.get_game_variable(GameVariable.KILLCOUNT), 'health' : game.get_game_variable(GameVariable.HEALTH), 
                        'ammo' : game.get_game_variable(GameVariable.AMMO2)}
        variables_prev = variables_cur.copy()

        # Get 1st state
        # state = game.get_state().screen_buffer
        # state = self.t(np.transpose(state,(1,2,0)))[None]
        frame = get_frame(game)
        for _ in range(self.stack_size):
          self.frames_deque.append(frame)
        state = stack_frames(self.frames_deque)
        
        for i in range(pretrain):
            # Get action and reward
            action = random.choice(self.possible_actions)
            reward = game.make_action(action, frame_skip)
            # Update the game vaiables dictionnaries and get the reshaped reward
            variables_cur['kills'] = game.get_game_variable(GameVariable.KILLCOUNT)
            variables_cur['health'] = game.get_game_variable(GameVariable.HEALTH)
            variables_cur['ammo'] = game.get_game_variable(GameVariable.AMMO2)
            reward += self.get_reward(variables_cur, variables_prev)
            variables_prev = variables_cur.copy()
            # Put reward and action in tensor form
            reward = torch.tensor([reward/100], dtype = torch.float)
            action = torch.tensor([action], dtype = torch.float)
            done = game.is_episode_finished()
            if done:
                # Set next state to zeros
                # next_state = np.zeros((480, 640), dtype='uint8')[:, :, None] # (480, 640) is the screen resolution, see cfg files /scenarios
                # next_state, stacked_frames = stack_frames(stacked_frames, next_state, False, self.stack_size, self.resize)
                self.frames_deque.append(torch.zeros((1,self.resize[0],self.resize[1])))
                next_state = stack_frames(self.frames_deque)
                # Add experience to replay buffer
                self.memory.add((state, action, reward, next_state, torch.tensor([not done], dtype = torch.float)))
                # Start a new episode
                game.new_episode()
                frame = get_frame(game)
                state = stack_frames(self.frames_deque)

            else:
                # Get next state
                self.frames_deque.append(get_frame(game))
                next_state = stack_frames(self.frames_deque)
                # Add experience to memory
                self.memory.add((state, action, reward, next_state, torch.tensor([not done], dtype = torch.float)))
                # update state variable
                state = next_state
        
        # Exploration-Exploitation phase
        decay_step = 0
        if bc_model:
          dqn_model = Mnih2015(self.resize,3,len(self.possible_actions))
          target_dqn_model = Mnih2015(self.resize,3,len(self.possible_actions))
          dqn_model.load_state_dict(torch.load(bc_model))
          target_dqn_model.load_state_dict(torch.load(bc_model))
          dqn_model.cuda()
          target_dqn_model.cuda()
        else:
          print("Creating DQN model")
          dqn_model = Mnih2015(self.resize,3,len(self.possible_actions))
          print("Creating DQN target model")
          target_dqn_model = Mnih2015(self.resize,3,len(self.possible_actions))
          dqn_model.cuda()
          target_dqn_model.cuda()
        optimizer = torch.optim.Adam(dqn_model.parameters(), lr=lr)
        for episode in range(total_episodes):
            # When tau > max_tau perform double q-learning update.
            tau = 0
            episode_rewards = []
            game.new_episode()
            variables_cur = {'kills' : game.get_game_variable(GameVariable.KILLCOUNT), 'health' : game.get_game_variable(GameVariable.HEALTH), 
                            'ammo' : game.get_game_variable(GameVariable.AMMO2)}
            variables_prev = variables_cur.copy()
            # Get 1st state
            done = game.is_episode_finished()
            self.frames_deque.append(get_frame(game))
            state = stack_frames(self.frames_deque)
            # stacked_frames = deque([torch.zeros(self.resize, dtype=torch.int) for i in range(self.stack_size)], maxlen = self.stack_size)
            # state, stacked_frames = stack_frames(stacked_frames, state, True, self.stack_size, self.resize)
            while (not done):
                tau += 1
                decay_step += 1
                # Predict the action to take
                action, explore_probability = predict_action(explore_start, explore_stop, decay_rate, decay_step, state, dqn_model, self.possible_actions)
                # Perform the chosen action on frame_skip frames
                reward = game.make_action(action, frame_skip)
                # Update the game vaiables dictionnaries and get the reshaped reward
                variables_cur['kills'] = game.get_game_variable(GameVariable.KILLCOUNT)
                variables_cur['health'] = game.get_game_variable(GameVariable.HEALTH)
                variables_cur['ammo'] = game.get_game_variable(GameVariable.AMMO2)
                reward += self.get_reward(variables_cur, variables_prev)
                variables_prev = variables_cur.copy()
                # Check if the episode is done
                done = game.is_episode_finished()
                # Add the reward to total reward
                episode_rewards.append(reward/100)
                reward = torch.tensor([reward/100], dtype = torch.float)
                action = torch.tensor([action], dtype = torch.float)
                if done:
                    # next_state = np.zeros((240, 320), dtype='uint8')[:, :, None]
                    # next_state, stacked_frames = stack_frames(stacked_frames, next_state, False, self.stack_size, self.resize)
                    self.frames_deque.append(torch.zeros((1,self.resize[0],self.resize[1])))
                    next_state = stack_frames(self.frames_deque)
                    total_reward = np.sum(episode_rewards)
                    print('Episode: {}'.format(episode),
                              'Total reward: {:.2f}'.format(total_reward),
                              'Training loss: {:.4f}'.format(loss),
                              'Explore P: {:.4f}'.format(explore_probability))
                    # Add experience to the replay buffer
                    self.memory.add((state, action, reward, next_state, torch.tensor([not done], dtype = torch.float)))
                    # Add number of kills and ammo variables
                    kill_count[episode%10] = game.get_game_variable(GameVariable.KILLCOUNT)
                    ammo[episode%10] = game.get_game_variable(GameVariable.AMMO2)
                    rewards[episode%10] = total_reward
                    losses[episode%10] = loss
                    # Update writer
                    if (episode > 0) and (episode%10 == 0):
                        writer.add_scalar('Game variables/Kills', kill_count.mean(), episode)
                        writer.add_scalar('Game variables/Ammo', ammo.mean(), episode)
                        writer.add_scalar('Reward Loss/Reward', rewards.mean(), episode)
                        writer.add_scalar('Reward Loss/loss', losses.mean(), episode)

                else:
                    # Get the next state
                    # next_state = get_state(game)
                    # next_state, stacked_frames = stack_frames(stacked_frames, next_state, False, self.stack_size, self.resize)
                    self.frames_deque.append(get_frame(game))
                    next_state = stack_frames(self.frames_deque)
                    # Add experience to memory
                    self.memory.add((state, action, reward, next_state, torch.tensor([not done], dtype = torch.float)))
                    # Update state variable
                    state = next_state

                # Learning phase
                transitions = self.memory.sample(self.batch_size)
                batch = Transition(*zip(*transitions))
                #print(f'Number of states in a batch: {len(batch.state)} | Type of a single state: {type(batch.state[0])} | Shape of a state {batch.state[0].shape}')
                states_mb = torch.cat(batch.state)
                actions_mb = torch.cat(batch.action)
                rewards_mb = torch.cat(batch.reward)
                next_states_mb = torch.cat(batch.next_state)
                dones_mb = torch.cat(batch.dones)
                next_states_mb = next_states_mb.cuda()
                states_mb = states_mb.cuda()
                q_next_state = dqn_model(next_states_mb).cpu()
                q_target_next_state = target_dqn_model(next_states_mb).cpu()
                q_state = dqn_model(states_mb).cpu()

                targets_mb = rewards_mb + (gamma*dones_mb*torch.max(q_target_next_state, 1)[0])
                output = (q_state * actions_mb).sum(1)
                loss = F.mse_loss(output, targets_mb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if tau > max_tau:
                    # Update the parameters of our target_dqn_model with DQN_weights
                    update_target(dqn_model, target_dqn_model)
                    print('model updated')
                    tau = 0
                    
            if (episode % freq) == 0: # +1 just to avoid the conditon episode != 0
                model_file = '/home/weights/' + self.scenario + '_' + str(episode) + '.pth'
                torch.save(dqn_model.state_dict(), model_file)
                print('\nSaved model to ' + model_file)
        
        writer.export_scalars_to_json("/home/all_scalars.json")
        writer.close()