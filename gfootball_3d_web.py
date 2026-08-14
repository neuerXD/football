#!/usr/bin/env python3
"""Streams a live Google Research Football match as rendered 3D frames."""

from __future__ import annotations

import argparse
import os
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
from urllib.parse import urlsplit

import numpy as np


INDEX_HTML = b"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Google Research Football 3D</title>
  <style>
    html, body {
      width: 100%;
      height: 100%;
      margin: 0;
      overflow: hidden;
      background: #050807;
    }
    img {
      width: 100vw;
      height: 100vh;
      object-fit: contain;
      display: block;
      background: #050807;
    }
  </style>
</head>
<body>
  <img id="frame" src="/frame.jpg" alt="Google Research Football 3D match">
  <script>
    const frame = document.getElementById('frame');
    function refresh() {
      frame.src = '/frame.jpg?t=' + Date.now();
    }
    frame.addEventListener('load', () => setTimeout(refresh, 50));
    frame.addEventListener('error', () => setTimeout(refresh, 250));
    refresh();
  </script>
</body>
</html>
"""


def _parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--host', default='127.0.0.1')
  parser.add_argument('--port', type=int, default=8765)
  parser.add_argument('--level', default='11_vs_11_official_ai')
  parser.add_argument('--seed', type=int, default=8)
  parser.add_argument('--width', type=int, default=1280)
  parser.add_argument('--fps', type=int, default=20)
  parser.add_argument('--jpeg-quality', type=int, default=92)
  parser.add_argument('--brightness', type=float, default=1.35,
                      help='Display brightness multiplier for 3D frames.')
  parser.add_argument('--gamma', type=float, default=0.72,
                      help='Display gamma correction. Values below 1 brighten.')
  parser.add_argument('--action_set', choices=['default', 'v2', 'full'],
                      default='full')
  return parser.parse_args()


class MatchStreamer:

  def __init__(self, args):
    self._args = args
    self._condition = threading.Condition()
    self._jpeg = None
    self._sequence = 0
    self._stopped = False
    self._thread = threading.Thread(target=self._run, daemon=True)

  def start(self):
    self._thread.start()

  def stop(self):
    with self._condition:
      self._stopped = True
      self._condition.notify_all()

  def latest(self, last_sequence):
    with self._condition:
      while (not self._stopped and
             (self._jpeg is None or self._sequence == last_sequence)):
        self._condition.wait(timeout=1.0)
      return self._sequence, self._jpeg

  def _run(self):
    # Off-screen rendering is the reliable path in WSLg for this engine.
    os.environ.pop('DISPLAY', None)
    os.environ.pop('SDL_VIDEODRIVER', None)

    import cv2
    from gfootball.env import config
    from gfootball.env import football_env

    height = int(self._args.width * 0.5625)
    cfg = config.Config({
        'action_set': self._args.action_set,
        'game_engine_random_seed': self._args.seed,
        'level': self._args.level,
        'players': [],
        'real_time': False,
        'render_resolution_x': self._args.width,
        'render_resolution_y': height,
    })
    env = football_env.FootballEnv(cfg)
    env.render()
    env.reset()

    frame_interval = 1.0 / max(1, self._args.fps)
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), self._args.jpeg_quality]
    while True:
      with self._condition:
        if self._stopped:
          return

      started = time.monotonic()
      _, _, done, _ = env.step([])
      frame = env.render('rgb_array')
      frame = self._adjust_frame(frame)
      ok, encoded = cv2.imencode('.jpg', frame[..., ::-1], encode_params)
      if ok:
        with self._condition:
          self._jpeg = encoded.tobytes()
          self._sequence += 1
          self._condition.notify_all()
      if done:
        env.reset()

      elapsed = time.monotonic() - started
      if elapsed < frame_interval:
        time.sleep(frame_interval - elapsed)

  def _adjust_frame(self, frame):
    frame = frame.astype(np.float32) / 255.0
    frame = np.power(frame, max(0.1, self._args.gamma))
    frame = np.clip(frame * self._args.brightness, 0.0, 1.0)
    return (frame * 255.0).astype(np.uint8)


def _make_handler(streamer):

  class Handler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
      return

    def do_GET(self):
      path = urlsplit(self.path).path

      if path in ('/', '/index.html'):
        self.send_response(HTTPStatus.OK)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', str(len(INDEX_HTML)))
        self.end_headers()
        self.wfile.write(INDEX_HTML)
        return

      if path == '/health':
        self.send_response(HTTPStatus.OK)
        self.send_header('Content-Type', 'text/plain; charset=utf-8')
        self.end_headers()
        self.wfile.write(b'ok\n')
        return

      if path == '/frame.jpg':
        _, jpeg = streamer.latest(0)
        if jpeg is None:
          self.send_error(HTTPStatus.SERVICE_UNAVAILABLE)
          return
        self.send_response(HTTPStatus.OK)
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Type', 'image/jpeg')
        self.send_header('Content-Length', str(len(jpeg)))
        self.end_headers()
        self.wfile.write(jpeg)
        return

      if path != '/stream.mjpg':
        self.send_error(HTTPStatus.NOT_FOUND)
        return

      self.send_response(HTTPStatus.OK)
      self.send_header('Age', '0')
      self.send_header('Cache-Control', 'no-cache, private')
      self.send_header('Pragma', 'no-cache')
      self.send_header(
          'Content-Type', 'multipart/x-mixed-replace; boundary=frame')
      self.end_headers()

      sequence = 0
      while True:
        sequence, jpeg = streamer.latest(sequence)
        if jpeg is None:
          return
        try:
          self.wfile.write(b'--frame\r\n')
          self.wfile.write(b'Content-Type: image/jpeg\r\n')
          self.wfile.write(
              b'Content-Length: ' + str(len(jpeg)).encode('ascii') + b'\r\n')
          self.wfile.write(b'\r\n')
          self.wfile.write(jpeg)
          self.wfile.write(b'\r\n')
        except (BrokenPipeError, ConnectionResetError):
          return

  return Handler


def main():
  args = _parse_args()
  streamer = MatchStreamer(args)
  streamer.start()
  server = ThreadingHTTPServer((args.host, args.port), _make_handler(streamer))
  print('Serving 3D match at http://{}:{}/'.format(args.host, args.port),
        flush=True)
  try:
    server.serve_forever()
  except KeyboardInterrupt:
    pass
  finally:
    streamer.stop()
    server.server_close()


if __name__ == '__main__':
  main()
