import time

class Stopwatch:
    def __init__(self):
        self.start = 0
        self.finish = 0
        self.duration = 0

    def __entry__(self):
        self.start = time.time()

    def __exit__(self):
        self.finish = time.time()
        self.duration = self.finish - self.start

class TimeTracker:
    def __init__(self):
        self.tick_list = list()

    def stopwatch(self):
        t = Stopwatch()
        self.tick_list.append(t)
        return t

