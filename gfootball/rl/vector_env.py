# coding=utf-8
"""Process-isolated vector environments for CPU-bound GRF simulation."""

from __future__ import absolute_import

import multiprocessing as mp

from gfootball.rl.macro_env import MacroTacticEnv


def _worker(remote, env_kwargs):
  env = MacroTacticEnv(**env_kwargs)
  try:
    while True:
      command, data = remote.recv()
      try:
        if command == 'reset':
          env.configure_episode(**data)
          result = env.reset()
        elif command == 'step':
          result = env.step(data)
        elif command == 'close':
          remote.send(('ok', None))
          return
        else:
          raise ValueError('Unknown vector command: {}'.format(command))
        remote.send(('ok', result))
      except Exception as exc:  # pylint: disable=broad-except
        remote.send(('error', '{}: {}'.format(type(exc).__name__, exc)))
        if command == 'close':
          return
  finally:
    env.close()


class SubprocessMacroVecEnv(object):
  """Synchronous vector wrapper with one GRF engine per child process."""

  def __init__(self, env_kwargs_list, start_method='spawn'):
    if not env_kwargs_list:
      raise ValueError('At least one environment is required')
    self.num_envs = len(env_kwargs_list)
    context = mp.get_context(start_method)
    self._remotes = []
    self._processes = []
    for env_kwargs in env_kwargs_list:
      parent_remote, child_remote = context.Pipe()
      process = context.Process(target=_worker,
                                args=(child_remote, dict(env_kwargs)))
      process.daemon = True
      process.start()
      child_remote.close()
      self._remotes.append(parent_remote)
      self._processes.append(process)
    self._closed = False

  def _receive(self, remote):
    status, payload = remote.recv()
    if status != 'ok':
      raise RuntimeError('Vector environment failed: {}'.format(payload))
    return payload

  def reset(self, configs):
    if len(configs) != self.num_envs:
      raise ValueError('Expected {} reset configs'.format(self.num_envs))
    for remote, data in zip(self._remotes, configs):
      remote.send(('reset', dict(data)))
    return [self._receive(remote) for remote in self._remotes]

  def reset_at(self, index, data):
    self._remotes[index].send(('reset', dict(data)))
    return self._receive(self._remotes[index])

  def step(self, actions):
    if len(actions) != self.num_envs:
      raise ValueError('Expected {} actions'.format(self.num_envs))
    for remote, action in zip(self._remotes, actions):
      remote.send(('step', int(action)))
    return [self._receive(remote) for remote in self._remotes]

  def close(self):
    if self._closed:
      return
    self._closed = True
    for remote in self._remotes:
      try:
        remote.send(('close', None))
      except (BrokenPipeError, EOFError):
        pass
    for remote in self._remotes:
      try:
        self._receive(remote)
      except (EOFError, RuntimeError):
        pass
      remote.close()
    for process in self._processes:
      process.join(timeout=5.0)
      if process.is_alive():
        process.terminate()
        process.join(timeout=2.0)

  def __enter__(self):
    return self

  def __exit__(self, unused_type, unused_value, unused_traceback):
    self.close()
