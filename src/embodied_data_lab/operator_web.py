from __future__ import annotations

import json
import queue
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np


PAGE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Embodied Data Lab - Operator</title>
  <style>
    * { box-sizing: border-box; }
    body { margin: 0; background: #111417; color: #eef2f5; font-family: Arial, sans-serif; letter-spacing: 0; }
    header { min-height: 58px; display: flex; align-items: center; justify-content: space-between; gap: 20px; padding: 10px 22px; border-bottom: 1px solid #3b4248; background: #191d21; }
    h1 { margin: 0; font-size: 20px; font-weight: 650; }
    #status { color: #7ee7ae; font-size: 14px; font-weight: 650; }
    main { width: 100%; max-width: 1600px; margin: 0 auto; }
    #stream { display: block; width: 100%; height: calc(100vh - 58px); object-fit: contain; background: #080a0c; }
    .instructions { display: grid; grid-template-columns: minmax(300px, 1.25fr) minmax(300px, 1fr); gap: 0; border-top: 1px solid #3b4248; }
    .workflow, .controls { padding: 18px 22px; }
    .workflow { border-right: 1px solid #3b4248; }
    h2 { margin: 0 0 10px; font-size: 15px; color: #f2cf67; }
    ol { margin: 0; padding-left: 22px; line-height: 1.55; color: #d4dae0; font-size: 14px; }
    .keys { display: grid; grid-template-columns: repeat(3, minmax(86px, 1fr)); gap: 8px; margin-bottom: 14px; }
    .key { min-height: 38px; display: flex; align-items: center; justify-content: center; border: 1px solid #535d66; background: #23292f; color: #f5f7f9; font-size: 13px; }
    .actions { display: flex; gap: 8px; flex-wrap: wrap; }
    button { min-height: 40px; padding: 0 16px; border: 1px solid #66717b; background: #2a3137; color: white; font-weight: 650; cursor: pointer; }
    button:hover, button:focus { border-color: #80d7ff; outline: none; }
    #start { background: #16734a; border-color: #2aaa71; }
    #discard { background: #702d32; border-color: #a64a52; }
    .note { margin: 12px 0 0; color: #aeb7bf; font-size: 13px; line-height: 1.4; }
    @media (max-width: 760px) { .instructions { grid-template-columns: 1fr; } .workflow { border-right: 0; border-bottom: 1px solid #3b4248; } }
  </style>
</head>
<body tabindex="0">
  <header>
    <h1>Two-tray demonstration</h1>
    <div id="status">Connecting...</div>
  </header>
  <main>
    <img id="stream" src="/stream.mjpg" alt="Behind, right-side, top-down, and front presentation robot camera views">
    <section class="instructions">
      <div class="workflow">
        <h2>Complete 3 recordings in order</h2>
        <ol>
          <li>Demo 1/3 opens in practice on seed 0. Practice for as long as needed.</li>
          <li>Press Enter once to start recording the current demo.</li>
          <li>Move above the green cube, lower the open gripper, and press Space to close it.</li>
          <li>Raise the cube, move over the red tray, lower it, and press Space to release.</li>
          <li>Move the gripper away and wait. Acceptance and replay validation are automatic.</li>
          <li>Demo 2/3 then opens in practice on seed 1. Press Enter and repeat. Do the same for Demo 3/3 on seed 2.</li>
          <li>Backspace discards and restarts only the current take. Earlier accepted demos remain.</li>
          <li>The batch is complete after all 3 demos are accepted.</li>
        </ol>
      </div>
      <div class="controls">
        <h2>Controls from behind the robot</h2>
        <div class="keys">
          <div class="key">W forward</div><div class="key">S backward</div><div class="key">R raise</div>
          <div class="key">A left</div><div class="key">D right</div><div class="key">F lower</div>
          <div class="key">Space grip</div><div class="key">Enter start</div><div class="key">Esc discard</div>
        </div>
        <div class="actions">
          <button id="start" type="button">Start recording</button>
          <button id="reset" type="button">Restart current take</button>
          <button id="discard" type="button">Discard and stop</button>
        </div>
        <p class="note">Imagine standing behind the Panda and looking toward the trays. The first three views are straight, orthographic, and hide the arm for control. The presentation view remains perspective, keeps the arm visible, and is saved as an MP4 during recording. Backspace discards the current take and video. Recording is limited to 500 simulation steps.</p>
      </div>
    </section>
  </main>
  <script>
    const movementKeys = new Set(['w', 's', 'a', 'd', 'r', 'f']);
    const discreteKeys = new Map([
      [' ', 'space'], ['enter', 'enter'], ['backspace', 'backspace'], ['escape', 'escape']
    ]);
    const heldKeys = new Set();
    async function send(command) {
      try {
        await fetch('/key', {method: 'POST', headers: {'Content-Type': 'text/plain'}, body: command});
      } catch (_) {
        document.getElementById('status').textContent = 'Simulator disconnected';
      }
    }
    document.addEventListener('keydown', (event) => {
      const key = event.key.toLowerCase();
      if (movementKeys.has(key)) {
        event.preventDefault();
        if (heldKeys.has(key)) return;
        heldKeys.add(key);
        send(`${key}_down`);
        return;
      }
      const command = discreteKeys.get(key);
      if (command && !event.repeat) {
        event.preventDefault();
        send(command);
      }
    });
    document.addEventListener('keyup', (event) => {
      const key = event.key.toLowerCase();
      if (!movementKeys.has(key)) return;
      event.preventDefault();
      heldKeys.delete(key);
      send(`${key}_up`);
    });
    window.addEventListener('blur', () => {
      if (heldKeys.size === 0) return;
      heldKeys.clear();
      send('release_all');
    });
    document.getElementById('start').onclick = () => send('enter');
    document.getElementById('reset').onclick = () => send('backspace');
    document.getElementById('discard').onclick = () => send('escape');
    async function updateStatus() {
      try {
        const state = await (await fetch('/state', {cache: 'no-store'})).json();
        document.getElementById('status').textContent = state.message || 'Ready';
      } catch (_) {
        document.getElementById('status').textContent = 'Simulator disconnected';
      }
    }
    setInterval(updateStatus, 500);
    updateStatus();
    window.focus();
  </script>
</body>
</html>"""


class OperatorWebServer:
    def __init__(self, port: int = 8765):
        self.port = int(port)
        self.commands: queue.Queue[str] = queue.Queue()
        self._condition = threading.Condition()
        self._frame = self._placeholder("Loading simulator...")
        self._sequence = 0
        self._state = {"message": "Loading simulator..."}
        self._stopping = threading.Event()
        self._httpd = ThreadingHTTPServer(("127.0.0.1", self.port), self._handler())
        self.port = self._httpd.server_port
        self._thread = threading.Thread(target=self._httpd.serve_forever, name="operator-http", daemon=True)

    @staticmethod
    def _placeholder(message: str) -> bytes:
        image = np.full((520, 1080, 3), (18, 21, 24), dtype=np.uint8)
        cv2.putText(image, message, (40, 270), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (235, 240, 244), 2, cv2.LINE_AA)
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if not ok:
            raise RuntimeError("could not encode placeholder frame")
        return encoded.tobytes()

    def _handler(self):
        app = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/":
                    payload = PAGE.encode("utf-8")
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                elif self.path.startswith("/state"):
                    with app._condition:
                        payload = json.dumps(app._state).encode("utf-8")
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                elif self.path.startswith("/stream.mjpg"):
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    sequence = -1
                    try:
                        while not app._stopping.is_set():
                            with app._condition:
                                app._condition.wait_for(
                                    lambda: app._sequence != sequence or app._stopping.is_set(),
                                    timeout=2.0,
                                )
                                frame = app._frame
                                sequence = app._sequence
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                            self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii"))
                            self.wfile.write(frame)
                            self.wfile.write(b"\r\n")
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                else:
                    self.send_error(HTTPStatus.NOT_FOUND)

            def do_POST(self):
                if self.path != "/key":
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                length = int(self.headers.get("Content-Length", "0"))
                command = self.rfile.read(length).decode("utf-8").strip().lower()
                movement = {f"{key}_{edge}" for key in "wsadrf" for edge in ("down", "up")}
                allowed = movement | {"release_all", "space", "enter", "backspace", "escape"}
                if command not in allowed:
                    self.send_error(HTTPStatus.BAD_REQUEST)
                    return
                app.commands.put(command)
                self.send_response(HTTPStatus.NO_CONTENT)
                self.end_headers()

            def log_message(self, format, *args):
                return

        return Handler

    def start(self) -> None:
        self._thread.start()

    def publish(self, image: np.ndarray, state: dict) -> None:
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if not ok:
            raise RuntimeError("could not encode operator frame")
        with self._condition:
            self._frame = encoded.tobytes()
            self._state = dict(state)
            self._sequence += 1
            self._condition.notify_all()

    def publish_message(self, message: str) -> None:
        frame = self._placeholder(message)
        with self._condition:
            self._frame = frame
            self._state = {"message": message}
            self._sequence += 1
            self._condition.notify_all()

    def next_command(self) -> str | None:
        try:
            return self.commands.get_nowait()
        except queue.Empty:
            return None

    def stop(self) -> None:
        self._stopping.set()
        with self._condition:
            self._condition.notify_all()
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=3.0)
