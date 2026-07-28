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

"""Runs no-render evaluation matches between two API LLM coaches."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import time

from absl import app
from absl import flags
from absl import logging

from gfootball.env import config
from gfootball.env import football_env


FLAGS = flags.FLAGS

flags.DEFINE_integer('episodes', 1, 'Number of episodes to run.')
flags.DEFINE_string('level', '11_vs_11_stochastic', 'Scenario level to run.')
flags.DEFINE_enum('action_set', 'full', ['v2', 'full'], 'Action set.')
flags.DEFINE_integer('max_steps', 0, 'Stop an episode after this many steps.')
flags.DEFINE_integer('interval_steps', 100, 'LLM planning interval in steps.')
flags.DEFINE_bool('mock', False, 'Use api_llm mock mode instead of API calls.')
flags.DEFINE_bool('execute_plan', True, 'Apply coach tactics and overrides.')
flags.DEFINE_bool('render', False, 'Enable rendering.')
flags.DEFINE_bool('real_time', False, 'Run with real-time throttling.')
flags.DEFINE_string('left_model', '', 'Optional left coach model override.')
flags.DEFINE_string('right_model', '', 'Optional right coach model override.')
flags.DEFINE_string('left_initial_formation', '',
                    'Optional left coach starting formation, e.g. 10-0-0.')
flags.DEFINE_string('right_initial_formation', '',
                    'Optional right coach starting formation, e.g. 4-3-3.')
flags.DEFINE_bool('lock_formation', False,
                  'Prevent API responses from changing configured formations.')
flags.DEFINE_string(
    'coach_log_path', '',
    'Optional JSONL path for raw api_llm coach decisions.')
flags.DEFINE_string(
    'output_path', '/tmp/gfootball_llm_eval.jsonl',
    'JSONL path for per-episode evaluation summaries. Empty disables writes.')


def _bool_flag(value):
  return '1' if value else '0'


def _api_llm_player(team, model):
  side_key = 'left_players' if team == 'left' else 'right_players'
  initial_formation = (FLAGS.left_initial_formation
                       if team == 'left' else FLAGS.right_initial_formation)
  params = [
      '{}=11'.format(side_key),
      'team={}'.format(team),
      'interval_steps={}'.format(FLAGS.interval_steps),
      'execute_plan={}'.format(_bool_flag(FLAGS.execute_plan)),
  ]
  if initial_formation:
    params.append('initial_formation={}'.format(initial_formation))
  if FLAGS.lock_formation:
    params.append('lock_formation=1')
  if FLAGS.mock:
    params.append('mock=1')
  if model:
    params.append('model={}'.format(model))
  if FLAGS.coach_log_path:
    params.append('log_path={}'.format(FLAGS.coach_log_path))
  return 'api_llm:{}'.format(','.join(params))


def _make_config():
  return config.Config({
      'action_set':
          FLAGS.action_set,
      'dump_full_episodes':
          False,
      'level':
          FLAGS.level,
      'players': [
          _api_llm_player('left', FLAGS.left_model),
          _api_llm_player('right', FLAGS.right_model),
      ],
      'real_time':
          FLAGS.real_time,
  })


def _write_event(path, event):
  if not path:
    return
  directory = os.path.dirname(path)
  if directory and not os.path.exists(directory):
    os.makedirs(directory)
  with open(path, 'a') as f:
    f.write(json.dumps(event, sort_keys=True) + '\n')


def _run_episode(env, episode):
  started_at = time.time()
  observation = env.reset()
  done = False
  steps = 0
  cumulative_score_reward = 0.0
  score = list(observation['score'])

  while not done:
    observation, _, done, info = env.step([])
    steps += 1
    score = list(observation['score'])
    cumulative_score_reward += float(info.get('score_reward', 0.0))
    if FLAGS.max_steps > 0 and steps >= FLAGS.max_steps:
      break

  return {
      'episode': episode,
      'level': FLAGS.level,
      'steps': steps,
      'score': score,
      'score_diff': score[0] - score[1],
      'score_reward': cumulative_score_reward,
      'done': done,
      'duration_sec': round(time.time() - started_at, 3),
      'left_model': FLAGS.left_model or os.environ.get('LLM_MODEL_LEFT', ''),
      'right_model': FLAGS.right_model or os.environ.get('LLM_MODEL_RIGHT', ''),
      'mock': FLAGS.mock,
      'execute_plan': FLAGS.execute_plan,
  }


def main(unused_argv):
  env = football_env.FootballEnv(_make_config())
  try:
    for episode in range(FLAGS.episodes):
      event = _run_episode(env, episode)
      _write_event(FLAGS.output_path, event)
      logging.info('Episode %d score=%s steps=%d reward=%.1f',
                   event['episode'], event['score'], event['steps'],
                   event['score_reward'])
      print(json.dumps(event, sort_keys=True))
  finally:
    env.close()


if __name__ == '__main__':
  app.run(main)
