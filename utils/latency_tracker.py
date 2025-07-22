import time

class LatencyTracker:
    def __init__(self):
        self.times = []

    def track(self, func, *args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()

        latency = end - start
        self.times.append(latency)
        return result, latency

    def summary(self):
        if not self.times:
            return {"min": 0, "max": 0, "avg": 0}

        return {
            "min": round(min(self.times) * 1000, 2),  # ms
            "max": round(max(self.times) * 1000, 2),
            "avg": round(sum(self.times) / len(self.times) * 1000, 2)
        }
