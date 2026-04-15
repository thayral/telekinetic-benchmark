import threading
from pynput import keyboard
from telekinetics.core.action import TelekineticAction

class Teleop:
    def __init__(self, n_objects):
        self.n_objects = n_objects
        self.selected = None
        self._pressed = set()
        self._lock = threading.Lock()
        self.listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        self.listener.daemon = True
        self.listener.start()

    def close(self):
        try:
            self.listener.stop()
        except Exception:
            pass

    def _on_press(self, key):
        with self._lock:
            if key == keyboard.Key.up:
                self._pressed.add("up")
            elif key == keyboard.Key.down:
                self._pressed.add("down")
            elif key == keyboard.Key.left:
                self._pressed.add("left")
            elif key == keyboard.Key.right:
                self._pressed.add("right")
            else:
                try:
                    if key.char.isdigit():
                        idx = int(key.char) - 1
                        if 0 <= idx < self.n_objects:
                            self.selected = idx
                except Exception:
                    pass

    def _on_release(self, key):
        with self._lock:
            if key == keyboard.Key.up:
                self._pressed.discard("up")
            elif key == keyboard.Key.down:
                self._pressed.discard("down")
            elif key == keyboard.Key.left:
                self._pressed.discard("left")
            elif key == keyboard.Key.right:
                self._pressed.discard("right")

    def action(self, speed=0.0001, steps=4):


        if self.selected is None:
            # return TelekineticAction(object_index=-1, dxy=(0.0, 0.0), steps=steps, frame="world")
            return None

        dx = dy = 0.0
        with self._lock:
            if "up" in self._pressed:
                dy += speed
            if "down" in self._pressed:
                dy -= speed
            if "left" in self._pressed:
                dx -= speed
            if "right" in self._pressed:
                dx += speed
            return TelekineticAction(object_index=self.selected, dxy=(dx, dy), steps=steps, frame="camera")
