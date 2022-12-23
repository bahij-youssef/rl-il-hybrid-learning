from argparse import ArgumentParser
from PIL import Image
from random import randint
import numpy as np
import vizdoom as vzd
from collections import deque, namedtuple
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from utils.networks import Mnih2015
from utils.dqn_preprocessing import *

parser = ArgumentParser("Play and evaluate vizdoom games with a trained network.")
parser.add_argument("models", type=str, nargs="+",
                    help="Path of the file(s) where the model will be loaded from.")
parser.add_argument("--rate", type=int, default=2,
                    help="Aka frameskip, number of frames per prediction")
parser.add_argument("--width", "-x", type=int, default=84,
                    help="Width of the image")
parser.add_argument("--height", "-y", type=int, default=84,
                    help="Height of the image")
parser.add_argument("--games", type=int, default=1,
                    help="How many games (per process) to run.")
parser.add_argument("--framestack", type=int, default=3,
                    help="Size of the frames stack.")
parser.add_argument("--no-cuda", type=bool, default=False,
                    help="How many games (per process) to run.")

args = parser.parse_args()



if args.no_cuda:
    device = torch.device("cpu")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def play_game(model_name, config, writer):
    # Determine the number of actions
    if Path(config).stem in ['deathmatch','deadly_corridor']:
        actions = 6
    else:
        actions = 3

    # Setup the environment
    env, possible_actions = initialise_environment(config, actions)
    env.set_mode(vzd.Mode.PLAYER)
    env.init()

    # Load the model
    model = Mnih2015((args.height, args.width),3,len(possible_actions))
    model.load_state_dict(torch.load(model_name))
    model.cuda()

    # Dict to store the array of screen buffers for each game
    game_capture_dict = dict()
    GameStats = namedtuple('GameStats',('images','total_reward','kills','ammo','game_count'))

    # Best reward game tracker
    max_reward = 0

    for game in range(args.games):
        print(f'Running game number {game+1} for model {Path(model_name).stem}')
        frame_stack = deque([],maxlen=3)
        env.new_episode()
        total_reward = 0
        total_tics = 0
        gif_images = list()

        observation = env.get_state()
        gif_images.append(observation.screen_buffer.transpose([1,2,0]))

        frame = get_frame(env)
        for _ in range(args.framestack):
          frame_stack.append(frame)
        state = stack_frames(frame_stack)

        done = env.is_episode_finished()

        while not done:
            total_tics += 1
            frame_stack.append(get_frame(env))
            state = stack_frames(frame_stack)

            # Capture image
            observation = env.get_state()
            gif_images.append(observation.screen_buffer.transpose([1,2,0]))

            # Get prediction
            q = model(state.cuda())

            action = possible_actions[int(torch.max(q, 1)[1][0])]

            reward = env.make_action(action, args.rate)
            done = env.is_episode_finished()

            total_reward += reward
            
            if done:
                kills = env.get_game_variable(GameVariable.KILLCOUNT)
                health = env.get_game_variable(GameVariable.HEALTH)
                ammo = env.get_game_variable(GameVariable.AMMO2)
                episode_timeout = env.get_episode_timeout()

                writer.add_scalar('Game variables/Kills', kills, game)
                writer.add_scalar('Game variables/Ammo', ammo, game)
                writer.add_scalar('Game variables/Health', health, game)
                writer.add_scalar('tics', (total_tics*args.rate)+1, game)
                writer.add_scalar('episode_timeout', episode_timeout, game)

                if total_reward > max_reward:
                    max_reward = total_reward
                    game_stats = GameStats(gif_images,total_reward,kills,ammo,game)
                    game_capture_dict[game] = game_stats
            else:
                continue
        
        #game_capture_dict[game] = game_stats
    
    return game_capture_dict


def get_models(model_path):
    models = Path(model_path)

    models_dict = dict()

    for config in models.iterdir():
        models_dict[config.stem] = config.iterdir()
    
    return models_dict


def get_best_game(game_stats):
    
    max_index = 0
    max_reward = 0

    for game, stats in game_stats.items():
        if stats.total_reward > max_reward:
            max_reward = stats.total_reward
            max_index = game
        else:
            continue
    
    return game_stats[max_index]


def main(model_path, games_count):
    # Get models from the given path
    models_dict = get_models(model_path)

    for config, models in models_dict.items():
        for model in models:
            print(f'Testing {model.stem}')
            # Initiate tensorboard summary writer
            writer = SummaryWriter(log_dir = '/home/runs/' + model.stem)
            game_stats = play_game(model, config, writer)

            writer.close()

            best_game = get_best_game(game_stats)

            make_gif(best_game.images,f'{model}.gif')


if __name__ == '__main__':
    main(args.models[0],args.games)