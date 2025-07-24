from river import linear_model, metrics, preprocessing, compose, anomaly, drift, optim
from river import base
from collections import defaultdict, deque
import statistics
import random
import heapq
from joblib import Memory

# Global cache counters
CACHE_STATS = {
    "fare_hits": 0,
    "fare_misses": 0,
    "eta_hits": 0,
    "eta_misses": 0
}

# Track rolling hit ratios
CACHE_HIT_HISTORY = {
    "fare": [],
    "eta": []
}

def log_cache_ratio():
    total_fare = CACHE_STATS["fare_hits"] + CACHE_STATS["fare_misses"]
    total_eta = CACHE_STATS["eta_hits"] + CACHE_STATS["eta_misses"]

    fare_ratio = CACHE_STATS["fare_hits"] / total_fare if total_fare > 0 else 0
    eta_ratio = CACHE_STATS["eta_hits"] / total_eta if total_eta > 0 else 0

    CACHE_HIT_HISTORY["fare"].append(round(fare_ratio * 100, 2))
    CACHE_HIT_HISTORY["eta"].append(round(eta_ratio * 100, 2))

def reset_cache_stats():
    CACHE_STATS["fare_hits"] = 0
    CACHE_STATS["fare_misses"] = 0
    CACHE_STATS["eta_hits"] = 0
    CACHE_STATS["eta_misses"] = 0
    CACHE_HIT_HISTORY["fare"].clear()
    CACHE_HIT_HISTORY["eta"].clear()


# Setup joblib caching directory
memory = Memory(location="cache", verbose=0)


class BinningTransformer(base.Transformer):
    def __init__(self, max_bins_per_feature=20):
        self.max_bins_per_feature = max_bins_per_feature
        self.seen_bins = defaultdict(lambda: deque(maxlen=max_bins_per_feature))

    def transform_one(self, x):
        binned = {}
        for k, v in x.items():
            if isinstance(v, float):
                if "lat" in k or "lon" in k:
                    bin_key = round(v, 2)
                    feature_name = f"{k}_{bin_key}"
                    if bin_key not in self.seen_bins[k]:
                        self.seen_bins[k].append(bin_key)
                    binned[feature_name] = 1
                elif "distance" in k:
                    bin_key = int(v // 2) * 2
                    feature_name = f"{k}_{bin_key}"
                    if bin_key not in self.seen_bins[k]:
                        self.seen_bins[k].append(bin_key)
                    binned[feature_name] = 1
                else:
                    binned[k] = v
            else:
                binned[k] = v
        return binned


class DriftMonitor:
    def __init__(self):
        self.fare_input = anomaly.HalfSpaceTrees(seed=42, n_trees=5, height=2, window_size=10)
        self.eta_input = anomaly.HalfSpaceTrees(seed=24, n_trees=5, height=2, window_size=10)
        self.fare_output = drift.KSWIN(alpha=0.005, window_size=50, stat_size=20)
        self.eta_output = drift.ADWIN()
        self.fare_input_scores = []
        self.eta_input_scores = []
        self.fare_output_flags = []
        self.eta_output_flags = []

    def update(self, x, fare_error, eta_error):
        self.fare_input.learn_one(x)
        self.eta_input.learn_one(x)
        self.fare_input_scores.append(self.fare_input.score_one(x))
        self.eta_input_scores.append(self.eta_input.score_one(x))
        self.fare_output_flags.append(self.fare_output.update(fare_error))
        self.eta_output_flags.append(self.eta_output.update(eta_error))


class MetricsTracker:
    def __init__(self):
        self.fare_mae = metrics.MAE()
        self.fare_rmse = metrics.RMSE()
        self.eta_mae = metrics.MAE()
        self.eta_rmse = metrics.RMSE()
        self.fare_rmse_history = []
        self.eta_rmse_history = []
        self.fare_baseline_rmse = []
        self.eta_baseline_rmse = []
        self.fare_values = []
        self.eta_values = []

    def update(self, fare_true, fare_pred, eta_true, eta_pred):
        self.fare_mae.update(fare_true, fare_pred)
        self.fare_rmse.update(fare_true, fare_pred)
        self.eta_mae.update(eta_true, eta_pred)
        self.eta_rmse.update(eta_true, eta_pred)

        self.fare_rmse_history.append(self.fare_rmse.get())
        self.eta_rmse_history.append(self.eta_rmse.get())

        self.fare_values.append(fare_true)
        self.eta_values.append(eta_true)

        fare_mean = statistics.mean(self.fare_values)
        eta_mean = statistics.mean(self.eta_values)
        self.fare_baseline_rmse.append(((fare_true - fare_mean) ** 2) ** 0.5)
        self.eta_baseline_rmse.append(((eta_true - eta_mean) ** 2) ** 0.5)


class ErrorTracker:
    def __init__(self):
        self.top_fare_errors = []
        self.top_eta_errors = []

    def update(self, x, fare_true, fare_pred, eta_true, eta_pred):
        fare_error = abs(fare_true - fare_pred)
        eta_error = abs(eta_true - eta_pred)
        heapq.heappush(self.top_fare_errors, (-fare_error, x, fare_true, fare_pred))
        heapq.heappush(self.top_eta_errors, (-eta_error, x, eta_true, eta_pred))
        self.top_fare_errors = self.top_fare_errors[:5]
        self.top_eta_errors = self.top_eta_errors[:5]

    def get_top_errors(self):
        fare_errors = []
        eta_errors = []
        for err, x, true, pred in sorted(self.top_fare_errors):
            abs_error = -err
            pct_error = (abs_error / true) * 100 if true != 0 else 0.0
            fare_errors.append((abs_error, x, true, pred, round(pct_error, 2)))
        for err, x, true, pred in sorted(self.top_eta_errors):
            abs_error = -err
            pct_error = (abs_error / true) * 100 if true != 0 else 0.0
            eta_errors.append((abs_error, x, true, pred, round(pct_error, 2)))
        return fare_errors, eta_errors


@memory.cache
def cached_fare_prediction(x_dict_serialized):
    global _fare_model
    CACHE_STATS["fare_misses"] += 1  # Joblib only calls this if cache MISS
    return _fare_model.predict_one(x_dict_serialized)


@memory.cache
def cached_eta_prediction(x_dict_serialized):
    global _eta_model
    CACHE_STATS["eta_misses"] += 1  # Called only if cache MISS
    return _eta_model.predict_one(x_dict_serialized)


class ModelTrainer:
    def __init__(self):
        self.scaler = preprocessing.StandardScaler()
        self.encoder = preprocessing.OneHotEncoder()
        self.binner = BinningTransformer()
        self.features = compose.TransformerUnion(
            ('scaled', self.scaler),
            ('encoded', compose.Pipeline(self.binner, self.encoder))
        )
        self.fare_model = compose.Pipeline(self.features, linear_model.LinearRegression(optimizer=optim.SGD(0.01)))
        self.eta_model = compose.Pipeline(self.features, linear_model.LinearRegression(optimizer=optim.SGD(0.01)))

    def predict(self, x):
        # Serialize the feature dict to make it hashable for caching
        x_serialized = {k: float(v) if isinstance(v, (int, float)) else str(v) for k, v in x.items()}
        global _fare_model, _eta_model
        _fare_model = self.fare_model
        _eta_model = self.eta_model

        # Manual cache check
        if cached_fare_prediction.check_call_in_cache(x_serialized):
            CACHE_STATS["fare_hits"] += 1
        fare_pred = cached_fare_prediction(x_serialized)

        if cached_eta_prediction.check_call_in_cache(x_serialized):
            CACHE_STATS["eta_hits"] += 1
        eta_pred = cached_eta_prediction(x_serialized)

        log_cache_ratio()

        return {
            "fare_pred": max(0.0, fare_pred),
            "eta_pred": max(0.0, eta_pred)
        }

    def learn(self, x, y):
        self.fare_model.learn_one(x, y['fare_amount'])
        self.eta_model.learn_one(x, y['duration_min'])

    def get_weights(self):
        fare_weights = self.fare_model[-1].weights if hasattr(self.fare_model[-1], "weights") else {}
        eta_weights = self.eta_model[-1].weights if hasattr(self.eta_model[-1], "weights") else {}
        return dict(fare_weights), dict(eta_weights)


class RideModel:
    def __init__(self):
        self.trainer = ModelTrainer()
        self.drift_monitor = DriftMonitor()
        self.metrics = MetricsTracker()
        self.errors = ErrorTracker()
        self.total_updates = 0
        self.inject_drift = True
        self.enable_online_update = True

    def predict(self, x):
        return self.trainer.predict(x)

    def update(self, x, y):
        self.total_updates += 1

        fare_true = y['fare_amount']
        eta_true = y['duration_min']

        if self.inject_drift:
            fare_true *= 1.2
            eta_true += 5 * random.random()

        preds = self.trainer.predict(x)
        fare_pred, eta_pred = preds["fare_pred"], preds["eta_pred"]

        fare_error = abs(fare_true - fare_pred)
        eta_error = abs(eta_true - eta_pred)

        self.metrics.update(fare_true, fare_pred, eta_true, eta_pred)

        if self.enable_online_update:
            self.trainer.learn(x, {"fare_amount": fare_true, "duration_min": eta_true})

        self.drift_monitor.update(x, fare_error, eta_error)
        self.errors.update(x, fare_true, fare_pred, eta_true, eta_pred)

    def get_top_errors(self):
        return self.errors.get_top_errors()

    def get_model_weights(self):
        return self.trainer.get_weights()
