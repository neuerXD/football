# coding=utf-8
"""Qwen teacher labeling with strict JSON validation and majority voting."""

from __future__ import absolute_import

import argparse
import json
import os
import re

import numpy as np

from gfootball.rl import tactics
from gfootball.rl import provenance


def build_prompt(state):
  fields = ', '.join('{}={:.4f}'.format(name, float(value))
                     for name, value in zip(_feature_names(), state))
  action_text = '; '.join('{}:{}'.format(index, name)
                          for index, name in enumerate(tactics.TACTIC_NAMES))
  return (
      'You are a football tactics teacher. Choose exactly one macro tactic '
      'for the controlled team. The low-level players are handled by the '
      'built-in game AI. Prefer a tactically coherent action based on score, '
      'time, possession and field position. Return only JSON with keys '
      'action_id (integer 0-11), confidence (number 0-1), and reason (short '
      'string).\nActions: {}\nState: {}').format(action_text, fields)


def _feature_names():
  from gfootball.rl.features import FEATURE_NAMES
  return FEATURE_NAMES


def parse_response(text):
  match = re.search(r'\{.*\}', str(text), flags=re.DOTALL)
  if match is None:
    raise ValueError('No JSON object in teacher response')
  payload = json.loads(match.group(0))
  action_id = int(payload['action_id'])
  confidence = float(payload['confidence'])
  if action_id < 0 or action_id >= tactics.NUM_TACTICS:
    raise ValueError('Teacher action outside action space')
  if not 0.0 <= confidence <= 1.0:
    raise ValueError('Teacher confidence outside [0, 1]')
  return {
      'action_id': action_id,
      'confidence': confidence,
      'reason': str(payload.get('reason', ''))[:500],
  }


def majority_vote(responses, confidence_threshold=0.55):
  valid = [response for response in responses if response is not None]
  if not valid:
    return -1, 0.0, '', 0
  counts = np.bincount([item['action_id'] for item in valid],
                       minlength=tactics.NUM_TACTICS)
  action_id = int(np.argmax(counts))
  winners = [item for item in valid if item['action_id'] == action_id]
  required_votes = len(responses) // 2 + 1
  if len(winners) < required_votes:
    return -1, 0.0, '', len(valid)
  confidence = float(np.mean([item['confidence'] for item in winners]))
  reason = winners[0]['reason'] if winners else ''
  label = action_id if confidence >= confidence_threshold else -1
  return label, confidence, reason, len(valid)


def _load_model(model_path, quantization):
  import torch
  from transformers import AutoModelForCausalLM, AutoTokenizer

  tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
  if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
  kwargs = {
      'trust_remote_code': True,
      'device_map': 'auto',
      'torch_dtype': torch.float16,
  }
  if quantization == '4bit':
    from transformers import BitsAndBytesConfig
    kwargs['quantization_config'] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
    )
  model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
  model.eval()
  return tokenizer, model


def _generate_batch(tokenizer, model, prompts, temperature, max_new_tokens):
  import torch
  if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template:
    prompts = [tokenizer.apply_chat_template(
        [{'role': 'user', 'content': prompt}],
        tokenize=False,
        add_generation_prompt=True) for prompt in prompts]
  encoded = tokenizer(prompts, return_tensors='pt', padding=True,
                      truncation=True, max_length=2048)
  device = next(model.parameters()).device
  encoded = {key: value.to(device) for key, value in encoded.items()}
  with torch.no_grad():
    generated = model.generate(
        **encoded,
        do_sample=True,
        temperature=temperature,
        top_p=0.9,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
    )
  prompt_length = encoded['input_ids'].shape[1]
  return tokenizer.batch_decode(generated[:, prompt_length:],
                                skip_special_tokens=True)


def label_clusters(cluster_dir, output_dir, model_path='', quantization='4bit',
                   num_samples=3, batch_size=4, seed=31,
                   confidence_threshold=0.55, temperature=0.7,
                   max_new_tokens=96, mock=False, shard_index=0,
                   num_shards=1):
  from gfootball.rl.collect_states import rule_action

  output_dir = os.path.abspath(output_dir)
  os.makedirs(output_dir, exist_ok=True)
  cluster_data = np.load(os.path.join(cluster_dir, 'clusters.npz'))
  all_states = cluster_data['representative_states']
  shard_index = int(shard_index)
  num_shards = int(num_shards)
  if num_shards < 1 or not 0 <= shard_index < num_shards:
    raise ValueError('Invalid shard {}/{}'.format(shard_index, num_shards))
  cluster_indices = np.arange(
      shard_index, len(all_states), num_shards, dtype=np.int64)
  states = all_states[cluster_indices]
  rng = np.random.RandomState(seed)
  tokenizer = model = None
  if not mock:
    if not model_path:
      raise ValueError('--model-path is required unless --mock is used')
    tokenizer, model = _load_model(model_path, quantization)

  labels = np.full(len(states), -1, dtype=np.int64)
  confidences = np.zeros(len(states), dtype=np.float32)
  valid_counts = np.zeros(len(states), dtype=np.int8)
  records = []
  for start in range(0, len(states), batch_size):
    stop = min(len(states), start + batch_size)
    prompts = []
    prompt_indices = []
    for index in range(start, stop):
      for sample in range(num_samples):
        prompt_indices.append(index)
        prompts.append(build_prompt(states[index]))
    if mock:
      responses = []
      for index in prompt_indices:
        action = rule_action(states[index])
        responses.append({
            'action_id': action,
            'confidence': 0.95,
            'reason': 'Deterministic mock teacher for pipeline validation.',
        })
    else:
      import torch
      torch.manual_seed(seed + shard_index * 100000 + start)
      raw_responses = _generate_batch(tokenizer, model, prompts, temperature,
                                      max_new_tokens)
      responses = []
      for raw in raw_responses:
        try:
          responses.append(parse_response(raw))
        except (TypeError, ValueError, KeyError, json.JSONDecodeError):
          responses.append(None)
    for index in range(start, stop):
      selected = [
          responses[offset] for offset, item in enumerate(prompt_indices)
          if item == index
      ]
      label, confidence, reason, valid = majority_vote(
          selected, confidence_threshold)
      labels[index] = label
      confidences[index] = confidence
      valid_counts[index] = valid
      records.append({
          'cluster_index': int(cluster_indices[index]),
          'action_id': int(label),
          'confidence': confidence,
          'valid_samples': valid,
          'reason': reason,
      })
    print('labeled {}/{}'.format(stop, len(states)), flush=True)

  np.savez_compressed(
      os.path.join(output_dir, 'teacher_labels.npz'),
      labels=labels,
      confidence=confidences,
      valid_samples=valid_counts,
      cluster_indices=cluster_indices,
  )
  with open(os.path.join(output_dir, 'teacher_labels.jsonl'), 'w') as output:
    for record in records:
      output.write(json.dumps(record, sort_keys=True))
      output.write('\n')
  manifest = {
      'clusters': int(len(all_states)),
      'shard_clusters': int(len(states)),
      'accepted': int(np.sum(labels >= 0)),
      'acceptance_rate': float(np.mean(labels >= 0)),
      'num_samples': num_samples,
      'confidence_threshold': confidence_threshold,
      'model_path': model_path or 'mock-rule-teacher',
      'seed': seed,
      'shard_index': shard_index,
      'num_shards': num_shards,
      'provenance': provenance.experiment_metadata(),
  }
  with open(os.path.join(output_dir, 'teacher_manifest.json'), 'w') as f:
    json.dump(manifest, f, indent=2, sort_keys=True)
  return manifest


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--cluster-dir', required=True)
  parser.add_argument('--output-dir', required=True)
  parser.add_argument('--model-path', default='')
  parser.add_argument('--quantization', choices=('none', '4bit'), default='4bit')
  parser.add_argument('--num-samples', type=int, default=3)
  parser.add_argument('--batch-size', type=int, default=4)
  parser.add_argument('--seed', type=int, default=31)
  parser.add_argument('--confidence-threshold', type=float, default=0.55)
  parser.add_argument('--temperature', type=float, default=0.7)
  parser.add_argument('--max-new-tokens', type=int, default=96)
  parser.add_argument('--mock', action='store_true')
  parser.add_argument('--shard-index', type=int, default=0)
  parser.add_argument('--num-shards', type=int, default=1)
  args = parser.parse_args()
  print(label_clusters(**vars(args)), flush=True)


if __name__ == '__main__':
  main()
