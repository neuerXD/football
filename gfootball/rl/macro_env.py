# coding=utf-8
"""Gym environment for low-frequency whole-team tactical decisions."""

import time

from gfootball.env import config
from gfootball.env import football_env
from gfootball.rl import features
from gfootball.rl import protocol
from gfootball.rl import tactics
import gym
import numpy as np


class MacroTacticEnv(gym.Env):
  """Runs built-in AI for 100 low-level steps per tactical action."""

  metadata = {'render.modes': ['human', 'rgb_array']}

  def __init__(self,
               seed=1000,
               split='train',
               control_left=True,
               opponent_difficulty=0.6,
               controlled_difficulty=1.0,
               macro_steps=100,
               game_duration=3600,
               gamma=0.99,
               use_potential_shaping=True,
               render=False):
    self.seed_value = protocol.validate_seed(seed, split)
    self.split = split
    self.control_left = bool(control_left)
    self.opponent_difficulty = float(opponent_difficulty)
    self.controlled_difficulty = float(controlled_difficulty)
    self.macro_steps = int(macro_steps)
    self.game_duration = int(game_duration)
    self.gamma = float(gamma)
    self.use_potential_shaping = bool(use_potential_shaping)
    self.render_enabled = bool(render)
    self.action_space = gym.spaces.Discrete(tactics.NUM_TACTICS)
    self.observation_space = gym.spaces.Box(
        low=-1.0, high=1.0, shape=(features.FEATURE_DIM,),
        dtype=np.float32)
    self._encoder = features.MacroFeatureEncoder(self.game_duration)
    self._base_env = None
    self._raw_observation = None
    self._current_action = 0
    self._tactic_age_steps = 0
    self._macro_step = 0
    self._episode_return = 0.0

  def _make_base_env(self):
    values = {
        'action_set': 'full',
        'dump_full_episodes': False,
        'game_engine_random_seed': self.seed_value,
        'level': '11_vs_11_macro',
        'macro_control_left': self.control_left,
        'macro_controlled_difficulty': self.controlled_difficulty,
        'macro_game_duration': self.game_duration,
        'macro_opponent_difficulty': self.opponent_difficulty,
        'players': [],
        'real_time': False,
    }
    self._base_env = football_env.FootballEnv(config.Config(values))
    if self.render_enabled:
      self._base_env.render()

  def configure_episode(self, seed=None, control_left=None,
                        opponent_difficulty=None):
    """Updates episode parameters; changes take effect on the next reset."""
    changed = False
    if seed is not None:
      next_seed = protocol.validate_seed(seed, self.split)
      changed = changed or next_seed != self.seed_value
      self.seed_value = next_seed
    if control_left is not None:
      next_side = bool(control_left)
      changed = changed or next_side != self.control_left
      self.control_left = next_side
    if opponent_difficulty is not None:
      next_difficulty = float(opponent_difficulty)
      changed = changed or next_difficulty != self.opponent_difficulty
      self.opponent_difficulty = next_difficulty
    if changed and self._base_env is not None:
      self._base_env.close()
      self._base_env = None

  def reset(self):
    if self._base_env is None:
      self._make_base_env()
    self._raw_observation = self._base_env.reset()
    self._current_action = 0
    self._tactic_age_steps = 0
    self._macro_step = 0
    self._episode_return = 0.0
    self._base_env.set_team_plan(
        self.control_left, tactics.tactic_plan(self._current_action))
    return self._observation()

  def _observation(self):
    return self._encoder.encode(
        self._raw_observation, self.control_left, self._current_action,
        self._tactic_age_steps)

  def _score_diff(self, observation):
    score = observation['score']
    return score[0] - score[1] if self.control_left else score[1] - score[0]

  def step(self, action):
    if self._raw_observation is None:
      raise RuntimeError('reset() must be called before step()')
    action = int(action)
    if not self.action_space.contains(action):
      raise ValueError('Invalid tactic action: {}'.format(action))

    started_at = time.perf_counter()
    switched = action != self._current_action
    self._current_action = action
    if switched:
      self._tactic_age_steps = 0
    plan = self._base_env.set_team_plan(
        self.control_left, tactics.tactic_plan(action))

    initial = self._raw_observation
    initial_score_diff = self._score_diff(initial)
    initial_potential = features.potential(initial, self.control_left)
    initial_mode = int(initial.get('game_mode', 0))
    initial_steps_left = int(initial['steps_left'])
    half = self.game_duration // 2
    done = False
    low_level_steps = 0
    engine_info = {}
    event = 'interval'

    while low_level_steps < self.macro_steps and not done:
      self._raw_observation, _, done, engine_info = self._base_env.step([])
      low_level_steps += 1
      if self._score_diff(self._raw_observation) != initial_score_diff:
        event = 'goal'
        break
      game_mode = int(self._raw_observation.get('game_mode', 0))
      if initial_mode == 0 and game_mode != 0:
        event = 'dead_ball'
        break
      steps_left = int(self._raw_observation['steps_left'])
      if initial_steps_left > half >= steps_left:
        event = 'halftime'
        break
    if done:
      event = 'terminal'

    final_score_diff = self._score_diff(self._raw_observation)
    goal_reward = 5.0 * (final_score_diff - initial_score_diff)
    shaping_reward = 0.0
    final_potential = features.potential(
        self._raw_observation, self.control_left)
    if self.use_potential_shaping:
      shaping_reward = 0.3 * (
          self.gamma * final_potential - initial_potential)
    switch_penalty = -0.01 if switched else 0.0
    terminal_reward = 0.0
    if done:
      terminal_reward = float(np.sign(final_score_diff))
    reward = goal_reward + shaping_reward + switch_penalty + terminal_reward

    self._tactic_age_steps += low_level_steps
    self._macro_step += 1
    self._episode_return += reward
    info = dict(engine_info)
    info.update({
        'action_id': action,
        'action_name': tactics.TACTIC_NAMES[action],
        'control_left': self.control_left,
        'decision_latency_ms': (time.perf_counter() - started_at) * 1000.0,
        'engine_seed': self.seed_value,
        'episode_return': self._episode_return,
        'event': event,
        'goal_reward': goal_reward,
        'low_level_steps': low_level_steps,
        'macro_step': self._macro_step,
        'opponent_difficulty': self.opponent_difficulty,
        'plan': plan,
        'potential_after': final_potential,
        'potential_before': initial_potential,
        'score': list(self._raw_observation['score']),
        'score_diff': final_score_diff,
        'shaping_reward': shaping_reward,
        'switch_penalty': switch_penalty,
        'terminal_reward': terminal_reward,
    })
    return self._observation(), float(reward), bool(done), info

  def render(self, mode='human'):
    if self._base_env is None:
      return None
    return self._base_env.render(mode=mode)

  def close(self):
    if self._base_env is not None:
      self._base_env.close()
      self._base_env = None
    self._raw_observation = None
