# coding=utf-8
"""Evaluates macro-tactic policies on the immutable held-out seed split."""

from __future__ import absolute_import

import argparse
import json
import os
import time

import numpy as np
import torch

from gfootball.rl import collect_states
from gfootball.rl import features
from gfootball.rl import models
from gfootball.rl import provenance
from gfootball.rl import protocol
from gfootball.rl import tactics
from gfootball.rl import teacher
from gfootball.rl import vector_env


def make_scenarios(seeds, difficulties, opponent='builtin'):
  """Builds the paired seed x side x difficulty evaluation schedule."""
  scenarios = []
  for difficulty in difficulties:
    for seed in seeds:
      protocol.validate_seed(seed, 'eval')
      for control_left in (True, False):
        side = 'left' if control_left else 'right'
        scenarios.append({
            'seed': int(seed),
            'control_left': control_left,
            'opponent_difficulty': float(difficulty),
            'opponent': opponent,
            'scenario_key': '{}:{}:{:.1f}:{}'.format(
                seed, side, float(difficulty), opponent),
        })
  return scenarios


def _episode_config(scenario):
  return {
      'seed': scenario['seed'],
      'control_left': scenario['control_left'],
      'opponent_difficulty': scenario['opponent_difficulty'],
  }


def _amortized_latency(started_at, count, device=None):
  if device is not None and device.type == 'cuda':
    torch.cuda.synchronize(device)
  elapsed_ms = (time.perf_counter() - started_at) * 1000.0
  return [elapsed_ms / max(1, count)] * count


class FixedPolicy(object):

  def __init__(self, action=0):
    self.action = int(action)
    if not 0 <= self.action < tactics.NUM_TACTICS:
      raise ValueError('Fixed action is outside the tactic space')
    self.metadata = {'type': 'fixed', 'action': self.action}

  def select(self, observations, contexts):
    del contexts
    started_at = time.perf_counter()
    actions = [self.action] * len(observations)
    return actions, _amortized_latency(started_at, len(actions))


class RandomPolicy(object):
  """Stateless per-scenario random baseline, invariant to batch order."""

  def __init__(self, seed=53):
    self.seed = int(seed)
    self.metadata = {'type': 'random', 'seed': self.seed}

  def select(self, observations, contexts):
    del observations
    started_at = time.perf_counter()
    actions = []
    for context in contexts:
      value = (self.seed + context['seed'] * 1009 +
               int(context['control_left']) * 9176 +
               context['macro_step'] * 7919 +
               int(round(context['opponent_difficulty'] * 10)) * 101)
      actions.append(int(np.random.RandomState(value).randint(
          tactics.NUM_TACTICS)))
    return actions, _amortized_latency(started_at, len(actions))


class RulePolicy(object):

  def __init__(self):
    self.metadata = {'type': 'rule'}

  def select(self, observations, contexts):
    del contexts
    started_at = time.perf_counter()
    actions = [collect_states.rule_action(state) for state in observations]
    return actions, _amortized_latency(started_at, len(actions))


class CheckpointPolicy(object):
  """Loads either a PPO checkpoint or a behavior-cloning actor checkpoint."""

  def __init__(self, checkpoint, device=None, deterministic=True):
    self.checkpoint = os.path.abspath(checkpoint)
    self.device = torch.device(device or 'cpu')
    self.deterministic = bool(deterministic)
    payload = torch.load(
        self.checkpoint, map_location='cpu', weights_only=False)
    self.model = models.ActorCritic(features.FEATURE_DIM,
                                    tactics.NUM_TACTICS).to(self.device)
    self.global_step = int(payload.get('global_step', 0))
    self.optimization_seed = payload.get(
        'optimization_seed', payload.get('seed'))
    if 'model_state_dict' in payload:
      self.kind = 'ppo'
      self.model.load_state_dict(payload['model_state_dict'])
      self.normalizer = models.RunningMeanStd(features.FEATURE_DIM)
      self.normalizer.load_state_dict(payload['obs_normalizer'])
      self._bc_mean = self._bc_std = None
    else:
      self.kind = 'bc'
      state = payload.get('actor_state_dict', payload.get('actor', payload))
      self.model.actor.load_state_dict(state)
      self.normalizer = None
      self._bc_mean = np.asarray(
          payload.get('normalizer_mean', np.zeros(features.FEATURE_DIM)),
          dtype=np.float32)
      self._bc_std = np.asarray(
          payload.get('normalizer_std', np.ones(features.FEATURE_DIM)),
          dtype=np.float32)
      self._bc_std[self._bc_std < 1e-6] = 1.0
    self.model.eval()
    self.metadata = {
        'type': 'checkpoint',
        'kind': self.kind,
        'checkpoint': self.checkpoint,
        'global_step': self.global_step,
        'optimization_seed': self.optimization_seed,
        'deterministic': self.deterministic,
        'device': str(self.device),
    }

  def _normalize(self, observations):
    observations = np.asarray(observations, dtype=np.float32)
    if self.normalizer is not None:
      return self.normalizer.normalize(observations)
    return (observations - self._bc_mean) / self._bc_std

  def select(self, observations, contexts):
    del contexts
    normalized = self._normalize(observations)
    tensor = torch.as_tensor(
        normalized, dtype=torch.float32, device=self.device)
    if self.device.type == 'cuda':
      torch.cuda.synchronize(self.device)
    started_at = time.perf_counter()
    with torch.no_grad():
      actions, _, _ = self.model.act(
          tensor, deterministic=self.deterministic)
    latencies = _amortized_latency(
        started_at, len(observations), self.device)
    return actions.cpu().numpy().astype(int).tolist(), latencies


class ZeroShotLLMPolicy(object):
  """Online Qwen baseline. Learned policies never call this at runtime."""

  def __init__(self, model_path='', quantization='4bit', num_samples=1,
               temperature=0.2, max_new_tokens=96, seed=59, mock=False):
    self.model_path = model_path
    self.num_samples = int(num_samples)
    if self.num_samples < 1:
      raise ValueError('num_samples must be positive')
    self.temperature = float(temperature)
    self.max_new_tokens = int(max_new_tokens)
    self.seed = int(seed)
    self.mock = bool(mock)
    self.calls = 0
    self.tokenizer = self.model = None
    if not self.mock:
      if not self.model_path:
        raise ValueError('model_path is required for zero-shot LLM evaluation')
      self.tokenizer, self.model = teacher._load_model(
          self.model_path, quantization)
    self.metadata = {
        'type': 'zero_shot_llm',
        'model_path': self.model_path or 'mock-rule-teacher',
        'quantization': quantization,
        'num_samples': self.num_samples,
        'temperature': self.temperature,
        'seed': self.seed,
        'mock': self.mock,
    }

  def select(self, observations, contexts):
    del contexts
    started_at = time.perf_counter()
    if self.mock:
      actions = [collect_states.rule_action(state) for state in observations]
      return actions, _amortized_latency(started_at, len(actions))
    prompts = []
    for state in observations:
      prompts.extend([teacher.build_prompt(state)] * self.num_samples)
    torch.manual_seed(self.seed + self.calls)
    self.calls += 1
    raw = teacher._generate_batch(
        self.tokenizer, self.model, prompts, self.temperature,
        self.max_new_tokens)
    actions = []
    for index, state in enumerate(observations):
      parsed = []
      responses = raw[index * self.num_samples:(index + 1) * self.num_samples]
      for response in responses:
        try:
          parsed.append(teacher.parse_response(response))
        except (TypeError, ValueError, KeyError, json.JSONDecodeError):
          parsed.append(None)
      action, _, _, valid = teacher.majority_vote(
          parsed, confidence_threshold=0.0)
      if valid == 0 or action < 0:
        action = collect_states.rule_action(state)
      actions.append(int(action))
    return actions, _amortized_latency(started_at, len(actions))


def _result_label(score_diff):
  if score_diff > 0:
    return 'win'
  if score_diff < 0:
    return 'loss'
  return 'draw'


def _new_tracker(scenario):
  return {
      'scenario': scenario,
      'started_at': time.time(),
      'episode_return': 0.0,
      'macro_decisions': 0,
      'low_level_steps': 0,
      'action_counts': np.zeros(tactics.NUM_TACTICS, dtype=np.int64),
      'policy_latencies': [],
      'engine_latencies': [],
  }


def _context(tracker, fallback):
  scenario = tracker['scenario'] if tracker is not None else fallback
  return {
      'seed': scenario['seed'],
      'control_left': scenario['control_left'],
      'opponent_difficulty': scenario['opponent_difficulty'],
      'macro_step': tracker['macro_decisions'] if tracker is not None else 0,
  }


def _episode_record(policy_name, tracker, info, policy):
  scenario = tracker['scenario']
  score_diff = int(info['score_diff'])
  action_counts = tracker['action_counts']
  decisions = max(1, tracker['macro_decisions'])
  policy_latencies = np.asarray(tracker['policy_latencies'], dtype=np.float64)
  engine_latencies = np.asarray(tracker['engine_latencies'], dtype=np.float64)
  result = _result_label(score_diff)
  return {
      'policy': policy_name,
      'policy_type': policy.metadata['type'],
      'checkpoint_global_step': int(getattr(policy, 'global_step', 0)),
      'optimization_seed': getattr(policy, 'optimization_seed', None),
      'scenario_key': scenario['scenario_key'],
      'engine_seed': scenario['seed'],
      'control_left': scenario['control_left'],
      'opponent': scenario['opponent'],
      'opponent_difficulty': scenario['opponent_difficulty'],
      'score': list(info['score']),
      'score_diff': score_diff,
      'result': result,
      'win': int(result == 'win'),
      'draw': int(result == 'draw'),
      'loss': int(result == 'loss'),
      'episode_return': float(tracker['episode_return']),
      'macro_decisions': tracker['macro_decisions'],
      'low_level_steps': tracker['low_level_steps'],
      'action_counts': action_counts.tolist(),
      'action_frequencies': (action_counts / float(decisions)).tolist(),
      'mean_policy_latency_ms': float(np.mean(policy_latencies)),
      'p50_policy_latency_ms': float(np.percentile(policy_latencies, 50)),
      'p95_policy_latency_ms': float(np.percentile(policy_latencies, 95)),
      'mean_engine_step_latency_ms': float(np.mean(engine_latencies)),
      'duration_sec': float(time.time() - tracker['started_at']),
  }


def _summary(records):
  if not records:
    return {}
  return {
      'games': len(records),
      'wins': int(sum(item['win'] for item in records)),
      'draws': int(sum(item['draw'] for item in records)),
      'losses': int(sum(item['loss'] for item in records)),
      'win_rate': float(np.mean([item['win'] for item in records])),
      'mean_score_diff': float(np.mean(
          [item['score_diff'] for item in records])),
      'mean_policy_latency_ms': float(np.mean(
          [item['mean_policy_latency_ms'] for item in records])),
  }


def evaluate(policy, policy_name, output_dir, scenarios, num_envs=8,
             macro_steps=100, game_duration=3600, start_method='spawn',
             tizero_model_dir=''):
  """Runs scenarios and writes one complete JSONL result set."""
  if not scenarios:
    raise ValueError('At least one evaluation scenario is required')
  output_dir = os.path.abspath(output_dir)
  os.makedirs(output_dir, exist_ok=True)
  run_provenance = provenance.experiment_metadata()
  num_envs = min(int(num_envs), len(scenarios))
  queue = list(scenarios)
  initial = [queue.pop(0) for _ in range(num_envs)]
  env_kwargs = [{
      'seed': item['seed'],
      'split': 'eval',
      'control_left': item['control_left'],
      'opponent_difficulty': item['opponent_difficulty'],
      'macro_steps': macro_steps,
      'game_duration': game_duration,
      'use_potential_shaping': True,
      'opponent': item['opponent'],
      'tizero_model_dir': tizero_model_dir,
  } for item in initial]
  envs = vector_env.SubprocessMacroVecEnv(env_kwargs, start_method)
  trackers = [_new_tracker(item) for item in initial]
  fallback = list(initial)
  records = []
  try:
    observations = np.asarray(
        envs.reset([_episode_config(item) for item in initial]),
        dtype=np.float32)
    while any(tracker is not None for tracker in trackers):
      active_indices = [
          index for index, tracker in enumerate(trackers)
          if tracker is not None
      ]
      contexts = [
          _context(trackers[index], fallback[index])
          for index in active_indices
      ]
      selected_actions, selected_latencies = policy.select(
          observations[active_indices], contexts)
      if (len(selected_actions) != len(active_indices) or
          len(selected_latencies) != len(active_indices)):
        raise ValueError('Policy returned an invalid batch size')
      actions = [0] * num_envs
      latencies = [0.0] * num_envs
      for offset, index in enumerate(active_indices):
        actions[index] = selected_actions[offset]
        latencies[index] = selected_latencies[offset]
      results = envs.step(actions)
      next_observations = np.empty_like(observations)
      for index, (next_obs, reward, done, info) in enumerate(results):
        tracker = trackers[index]
        next_observations[index] = next_obs
        if tracker is not None:
          action = int(actions[index])
          tracker['episode_return'] += float(reward)
          tracker['macro_decisions'] += 1
          tracker['low_level_steps'] += int(info['low_level_steps'])
          tracker['action_counts'][action] += 1
          tracker['policy_latencies'].append(float(latencies[index]))
          tracker['engine_latencies'].append(float(
              info.get('engine_step_latency_ms', 0.0)))
          if done:
            records.append(_episode_record(
                policy_name, tracker, info, policy))
            if queue:
              scenario = queue.pop(0)
              fallback[index] = scenario
              trackers[index] = _new_tracker(scenario)
              next_observations[index] = envs.reset_at(
                  index, _episode_config(scenario))
            else:
              trackers[index] = None
              if any(item is not None for item in trackers):
                next_observations[index] = envs.reset_at(
                    index, _episode_config(fallback[index]))
        elif done and any(item is not None for item in trackers):
          next_observations[index] = envs.reset_at(
              index, _episode_config(fallback[index]))
      observations = next_observations
  finally:
    envs.close()

  records.sort(key=lambda item: item['scenario_key'])
  with open(os.path.join(output_dir, 'episodes.jsonl'), 'w') as output:
    for record in records:
      output.write(json.dumps(record, sort_keys=True))
      output.write('\n')
  manifest = {
      'policy': policy_name,
      'policy_metadata': policy.metadata,
      'games': len(records),
      'macro_steps': int(macro_steps),
      'game_duration': int(game_duration),
      'num_envs': num_envs,
      'seed_split': 'eval',
      'summary': _summary(records),
      'tactic_names': list(tactics.TACTIC_NAMES),
      'provenance': run_provenance,
  }
  with open(os.path.join(output_dir, 'evaluation_manifest.json'), 'w') as f:
    json.dump(manifest, f, indent=2, sort_keys=True)
  return manifest


def _parse_args():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output-dir', required=True)
  parser.add_argument('--policy-name', required=True)
  parser.add_argument('--policy', required=True,
                      choices=('fixed', 'random', 'rule', 'checkpoint', 'llm'))
  parser.add_argument('--fixed-action', type=int, default=0)
  parser.add_argument('--policy-seed', type=int, default=53)
  parser.add_argument('--checkpoint', default='')
  parser.add_argument('--device', default=None)
  parser.add_argument('--stochastic', action='store_true')
  parser.add_argument('--model-path', default='')
  parser.add_argument('--quantization', choices=('none', '4bit'), default='4bit')
  parser.add_argument('--llm-samples', type=int, default=1)
  parser.add_argument('--llm-temperature', type=float, default=0.2)
  parser.add_argument('--llm-mock', action='store_true')
  parser.add_argument('--num-seeds', type=int, default=50)
  parser.add_argument('--difficulties', type=float, nargs='+',
                      default=list(protocol.EVAL_DIFFICULTIES))
  parser.add_argument('--num-envs', type=int, default=8)
  parser.add_argument('--macro-steps', type=int, default=100)
  parser.add_argument('--game-duration', type=int, default=3600)
  parser.add_argument('--start-method', choices=('spawn', 'fork'),
                      default='spawn')
  parser.add_argument('--opponent', choices=('builtin', 'tizero'),
                      default='builtin')
  parser.add_argument('--tizero-model-dir', default='')
  return parser.parse_args()


def _make_policy(args):
  if args.policy == 'fixed':
    return FixedPolicy(args.fixed_action)
  if args.policy == 'random':
    return RandomPolicy(args.policy_seed)
  if args.policy == 'rule':
    return RulePolicy()
  if args.policy == 'checkpoint':
    if not args.checkpoint:
      raise ValueError('--checkpoint is required for checkpoint policies')
    return CheckpointPolicy(
        args.checkpoint, args.device, deterministic=not args.stochastic)
  return ZeroShotLLMPolicy(
      model_path=args.model_path,
      quantization=args.quantization,
      num_samples=args.llm_samples,
      temperature=args.llm_temperature,
      seed=args.policy_seed,
      mock=args.llm_mock)


def main():
  args = _parse_args()
  if not 1 <= args.num_seeds <= len(protocol.EVAL_ENV_SEEDS):
    raise ValueError('num_seeds must be in [1, 50]')
  scenarios = make_scenarios(
      protocol.EVAL_ENV_SEEDS[:args.num_seeds], args.difficulties,
      args.opponent)
  policy = _make_policy(args)
  manifest = evaluate(
      policy=policy,
      policy_name=args.policy_name,
      output_dir=args.output_dir,
      scenarios=scenarios,
      num_envs=args.num_envs,
      macro_steps=args.macro_steps,
      game_duration=args.game_duration,
      start_method=args.start_method,
      tizero_model_dir=args.tizero_model_dir)
  print(json.dumps(manifest, sort_keys=True), flush=True)


if __name__ == '__main__':
  main()
