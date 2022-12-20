import numpy as np
from PIL import Image
import vizdoom as vzd
from vizdoom import GameVariable
import random
import moviepy.editor as mpy

import torch
import torchvision.transforms as T

def initialise_environment(scenario: str, actions: int)->tuple:
  # Create Doom Game env
  game = vzd.DoomGame()
  # Load desired scenario
  game.load_config(f'/home/config/{scenario}.cfg')
  game.set_doom_scenario_path(f'/home/config/{scenario}.wad')

  possible_actions = np.identity(actions, dtype=np.uint8).tolist()

  return game, possible_actions

def transform(resize:tuple=(84,84), normalize:bool=True):
  if normalize:
    return T.Compose([
        T.ToPILImage(),
        T.Resize(resize),
        T.ToTensor(),
        T.Normalize(0,255)
    ])
  else:
    return T.Compose([
        T.ToPILImage(),
        T.Resize(resize),
        T.ToTensor()
    ])

def resize_frame(img, output_h=60, output_w=80):
  # ViZDoom gives images as 
  # CHW, turn to HWC
  img = img.transpose([1, 2, 0])
  img = Image.fromarray(img)
  img = img.resize((output_w, output_h), Image.BILINEAR)
  #img = img.transpose([2,0,1])
  # Turn to float (normalization happens later)
  resized = np.asarray(img, dtype=np.float32)

  # Normalize
  normalized_resized = resized / 255
  #print(normalized_resized.shape)

  #normalized_resized = np.transpose(normalized_resized,(2,0,1))
  normalized_resized = np.expand_dims(normalized_resized[:,:,0],axis=0)

  return normalized_resized


def get_frame(game, transform=None, resize=(60,80)):
  if transform:
    frame = game.get_state().screen_buffer
    return transform(frame)
  else:
    return resize_frame(game.get_state().screen_buffer)


def stack_frames(frames_deque):
    assert len(frames_deque) == 3
    np_stack = np.concatenate(frames_deque, axis=0)
    np_stack = np.expand_dims(np_stack, axis=0)
    np_stack = np_stack.astype(np.float32)
    #logger.info(f'Shape of stack_frames output: {np_stack.shape}')
    return torch.from_numpy(np_stack)


"""
epsilon-greedy
"""
def predict_action(explore_start, explore_stop, decay_rate, decay_step, state, model, possible_actions):
    """
    Description
    -------------
    Epsilon-greedy policy
    
    Parameters
    -------------
    explore_start    : Float, the initial exploration probability.
    explore_stop     : Float, the last exploration probability.
    decay_rate       : Float, the rate at which the exploration probability decays.
    state            : 4D-tensor (batch, motion, image)
    model            : models.DQNetwork or models.DDDQNetwork object, the architecture used.
    possible_actions : List, the one-hot encoded possible actions.
    
    Returns
    -------------
    action              : np.array of shape (number_actions,), the action chosen by the greedy policy.
    explore_probability : Float, the exploration probability.
    """
    
    exp_exp_tradeoff = np.random.rand()
    explore_probability = explore_stop + (explore_start - explore_stop)*np.exp(-decay_rate*decay_step)
    if (explore_probability > exp_exp_tradeoff):
        action = random.choice(possible_actions)
        
    else:
        Qs = model.forward(state.cuda())
        action = possible_actions[int(torch.max(Qs, 1)[1][0])]

    return action, explore_probability


def update_target(current_model, target_model):
    """
    Description
    -------------
    Update the parameters of target_model with those of current_model
    
    Parameters
    -------------
    current_model, target_model : torch models
    """
    target_model.load_state_dict(current_model.state_dict())


def make_gif(images, fname, fps=20):

    def make_frame(t):
        try:
            x = images[int(fps*t)]
        except:
            x = images[-1]
        return x.astype(np.uint8)
    myfps = fps
    clip = mpy.VideoClip(make_frame, duration=len(images)/fps)
    clip.fps = fps
    clip.write_gif(fname, program='ffmpeg', fuzz=50, verbose=False)