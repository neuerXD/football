# coding=utf-8
# Copyright 2019 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Runs evaluation matches where both teams use the built-in engine AI."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import time

from absl import app
from absl import flags
import cv2
import numpy as np

from gfootball.env import config
from gfootball.env import football_env
from gfootball.env import observation_processor


FLAGS = flags.FLAGS

_DEFAULT_SEED_SCHEDULE = (1, 4, 5, 6, 7, 8, 10, 11, 13, 15)

flags.DEFINE_integer('episodes', len(_DEFAULT_SEED_SCHEDULE),
                     'Number of episodes to run.')
flags.DEFINE_string('level', '11_vs_11_official_ai', 'Scenario level to run.')
flags.DEFINE_enum('action_set', 'full', ['default', 'v2', 'full'],
                  'Action set used by the environment.')
flags.DEFINE_integer('max_steps', 0, 'Stop an episode after this many steps.')
flags.DEFINE_integer('min_goals', 2, 'Expected minimum total goals per match.')
flags.DEFINE_integer('max_goals', 4, 'Expected maximum total goals per match.')
flags.DEFINE_integer('sample_interval', 100,
                     'How often to sample team shape metrics.')
flags.DEFINE_bool('render', False, 'Enable rendering.')
flags.DEFINE_bool('real_time', False, 'Run with real-time throttling.')
flags.DEFINE_string(
    'seed_schedule', ','.join(str(seed) for seed in _DEFAULT_SEED_SCHEDULE),
    'Comma-separated game_engine_random_seed values. Empty uses fresh random '
    'seeds from the scenario builder.')
flags.DEFINE_string(
    'topdown_dir', '',
    'Optional directory for 2D top-down PNG frames generated from raw '
    'observations. Useful when 3D rendering is unavailable.')
flags.DEFINE_string(
    'topdown_steps', '0,900,1800,2700,3601',
    'Comma-separated episode steps to save when topdown_dir is set.')
flags.DEFINE_string(
    'output_path', '/tmp/gfootball_official_ai_eval.jsonl',
    'JSONL path for per-episode summaries. Empty disables writes.')


def _parse_seed_schedule(value):
  value = value.strip()
  if not value:
    return []
  return [int(seed.strip()) for seed in value.split(',') if seed.strip()]


def _parse_int_list(value):
  value = value.strip()
  if not value:
    return []
  return [int(item.strip()) for item in value.split(',') if item.strip()]


def _make_config(seed):
  values = {
      'action_set': FLAGS.action_set,
      'dump_full_episodes': False,
      'level': FLAGS.level,
      'players': [],
      'real_time': FLAGS.real_time,
  }
  if seed is not None:
    values['game_engine_random_seed'] = seed
  return config.Config(values)


def _write_event(path, event):
  if not path:
    return
  directory = os.path.dirname(path)
  if directory and not os.path.exists(directory):
    os.makedirs(directory)
  with open(path, 'a') as f:
    f.write(json.dumps(event, sort_keys=True) + '\n')


def _team_shape(observation, team):
  positions = np.asarray(observation[team], dtype=np.float32)
  directions = np.asarray(
      observation['{}_direction'.format(team)], dtype=np.float32)
  active = np.asarray(observation['{}_active'.format(team)], dtype=bool)
  outfield = positions[active][1:] if np.any(active) else positions[1:]
  speed = np.linalg.norm(directions[active], axis=1)
  if len(outfield) == 0:
    return {
        'active_players': int(np.sum(active)),
        'depth': 0.0,
        'width': 0.0,
        'mean_speed': float(np.mean(speed)) if len(speed) else 0.0,
        'min_pair_distance': 0.0,
    }
  pair_distances = []
  for i in range(len(outfield)):
    for j in range(i + 1, len(outfield)):
      pair_distances.append(float(np.linalg.norm(outfield[i] - outfield[j])))
  return {
      'active_players': int(np.sum(active)),
      'depth': round(float(np.max(outfield[:, 0]) - np.min(outfield[:, 0])), 4),
      'width': round(float(np.max(outfield[:, 1]) - np.min(outfield[:, 1])), 4),
      'mean_speed': round(float(np.mean(speed)) if len(speed) else 0.0, 4),
      'min_pair_distance': round(min(pair_distances) if pair_distances else 0.0,
                                 4),
  }


def _summarize_shapes(samples, team):
  if not samples:
    return {}
  keys = ('active_players', 'depth', 'width', 'mean_speed',
          'min_pair_distance')
  result = {}
  for key in keys:
    values = [sample[team][key] for sample in samples]
    result[key] = {
        'min': round(float(min(values)), 4),
        'avg': round(float(sum(values) / len(values)), 4),
        'max': round(float(max(values)), 4),
    }
  return result


def _write_topdown_frame(observation, directory, episode, step, seed):
  if not directory:
    return ''
  if not os.path.exists(directory):
    os.makedirs(directory)
  state = observation_processor.ObservationState({
      'observation': observation,
      'debug': {
          'action': []
      },
  })
  frame = observation_processor.get_frame(state)
  cv2.putText(
      frame,
      'official AI seed {} episode {} step {} score {}-{}'.format(
          seed, episode, step, observation['score'][0],
          observation['score'][1]), (18, 585), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
      (255, 255, 255), 2)
  path = os.path.join(
      directory, 'official_ai_episode_{:02d}_seed_{}_step_{:04d}.png'.format(
          episode, seed, step))
  cv2.imwrite(path, frame)
  return path


def _run_episode(env, episode, seed, topdown_steps):
  started_at = time.time()
  observation = env.reset()
  left_agents = int(env._env._env.config.left_agents)
  right_agents = int(env._env._env.config.right_agents)
  done = False
  steps = 0
  samples = []
  topdown_frames = []
  score = list(observation['score'])
  topdown_targets = set(topdown_steps)
  if 0 in topdown_targets:
    topdown_frames.append(
        _write_topdown_frame(observation, FLAGS.topdown_dir, episode, 0, seed))

  while not done:
    observation, _, done, info = env.step([])
    steps += 1
    score = list(observation['score'])
    if FLAGS.sample_interval > 0 and steps % FLAGS.sample_interval == 0:
      samples.append({
          'left': _team_shape(observation, 'left_team'),
          'right': _team_shape(observation, 'right_team'),
      })
    if steps in topdown_targets:
      topdown_frames.append(
          _write_topdown_frame(observation, FLAGS.topdown_dir, episode, steps,
                               seed))
    if FLAGS.max_steps > 0 and steps >= FLAGS.max_steps:
      break

  total_goals = int(score[0] + score[1])
  return {
      'episode': episode,
      'level': FLAGS.level,
      'game_engine_random_seed': seed,
      'steps': steps,
      'score': score,
      'total_goals': total_goals,
      'goal_band_ok': FLAGS.min_goals <= total_goals <= FLAGS.max_goals,
      'expected_goal_band': [FLAGS.min_goals, FLAGS.max_goals],
      'done': done,
      'duration_sec': round(time.time() - started_at, 3),
      'left_agents': left_agents,
      'right_agents': right_agents,
      'shape_samples': len(samples),
      'topdown_frames': [path for path in topdown_frames if path],
      'left_shape': _summarize_shapes(samples, 'left'),
      'right_shape': _summarize_shapes(samples, 'right'),
      'score_reward': float(info.get('score_reward', 0.0)),
  }


def main(unused_argv):
  seed_schedule = _parse_seed_schedule(FLAGS.seed_schedule)
  topdown_steps = _parse_int_list(FLAGS.topdown_steps)
  events = []
  for episode in range(FLAGS.episodes):
    seed = (seed_schedule[episode % len(seed_schedule)]
            if seed_schedule else None)
    env = football_env.FootballEnv(_make_config(seed))
    if FLAGS.render:
      env.render()
    try:
      event = _run_episode(env, episode, seed, topdown_steps)
      events.append(event)
      _write_event(FLAGS.output_path, event)
      print(json.dumps(event, sort_keys=True))
    finally:
      env.close()

  total = len(events)
  if total:
    passed = sum(1 for event in events if event['goal_band_ok'])
    goals = [event['total_goals'] for event in events]
    summary = {
        'episodes': total,
        'goal_band_ok': passed,
        'goal_band_rate': round(float(passed) / total, 4),
        'total_goals': goals,
        'avg_total_goals': round(float(sum(goals)) / total, 4),
        'seed_schedule': seed_schedule,
        'official_ai_only': all(
            event['left_agents'] == 0 and event['right_agents'] == 0
            for event in events),
    }
    print(json.dumps({'summary': summary}, sort_keys=True))


if __name__ == '__main__':
  app.run(main)
