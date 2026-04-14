import threading


class GameState:
    def __init__(self):
        self.lock = threading.Lock()
        self._quit = False
        self._reset = False

    def get_quit(self):
        with self.lock:
            return self._quit

    def set_quit(self, value):
        with self.lock:
            self._quit = value

    def get_reset(self):
        with self.lock:
            return self._reset

    def set_reset(self, value):
        with self.lock:
            self._reset = value
