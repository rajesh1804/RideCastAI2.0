from river import linear_model, metrics, preprocessing, compose, anomaly, drift, optim, preprocessing  
import statistics
import random
import heapq

from river import base


class BinningTransformer(base.Transformer):
    def transform_one(self, x):
        binned = {}
        for k, v in x.items():
            if isinstance(v, float):
                if "lat" in k or "lon" in k:
                    # Round lat/lon to 2 decimal places
                    binned[f"{k}_{round(v, 2)}"] = 1
                elif "distance" in k:
                    # Bin distances in steps of 2 km (e.g., 7.8 → "distance_km_6")
                    bin_key = int(v // 2) * 2
                    binned[f"{k}_{bin_key}"] = 1
                else:
                    binned[k] = v
            else:
                binned[k] = v
        return binned

class RideModel:
    def __init__(self):
        self.scaler = preprocessing.StandardScaler()
        self.encoder = preprocessing.OneHotEncoder()
        self.binner = BinningTransformer()

        self.features = compose.TransformerUnion(
            ('scaled', self.scaler),
            ('encoded', compose.Pipeline(self.binner, self.encoder))
        )

        self.fare_model = compose.Pipeline(
            self.features,
            linear_model.LinearRegression(optimizer=optim.SGD(0.01))
        )

        self.eta_model = compose.Pipeline(
            self.features,
            linear_model.LinearRegression(optimizer=optim.SGD(0.01))
        )

        # Drift detectors
        self.fare_drift_input = anomaly.HalfSpaceTrees(seed=42, n_trees=5, height=2, window_size=10)
        self.eta_drift_input = anomaly.HalfSpaceTrees(seed=24, n_trees=5, height=2, window_size=10)
        self.fare_drift_output = drift.KSWIN(alpha=0.005, window_size=50, stat_size=20)
        self.eta_drift_output = drift.ADWIN()

        # Metrics & buffers
        self.fare_mae = metrics.MAE()
        self.fare_rmse = metrics.RMSE()
        self.eta_mae = metrics.MAE()
        self.eta_rmse = metrics.RMSE()

        self.total_updates = 0
        self.fare_drift_scores = []
        self.eta_drift_scores = []
        self.fare_drift_output_flags = []
        self.eta_drift_output_flags = []
        self.fare_rmse_history = []
        self.eta_rmse_history = []
        self.fare_values = []
        self.eta_values = []
        self.fare_baseline_rmse = []
        self.eta_baseline_rmse = []

        # Config toggles
        self.inject_drift = True
        self.enable_online_update = True

        # Top-N error buffers
        self.top_fare_errors = []
        self.top_eta_errors = []


    def predict(self, x):
        fare = self.fare_model.predict_one(x)
        eta = self.eta_model.predict_one(x)
        return {
            "fare_pred": max(0.0, fare),
            "eta_pred": max(0.0, eta)
        }

    def update(self, x, y):
        self.total_updates += 1

        fare_true = y['fare_amount']
        eta_true = y['duration_min']

        if self.inject_drift:
            if 10 <= self.total_updates <= 20:
                fare_true *= 1.2
            if 20 <= self.total_updates <= 30:
                eta_true += 5 * random.random()

        fare_pred = self.fare_model.predict_one(x)
        eta_pred = self.eta_model.predict_one(x)

        fare_error = abs(fare_true - fare_pred)
        eta_error = abs(eta_true - eta_pred)

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

        if self.enable_online_update:
            self.fare_model.learn_one(x, fare_true)
            self.eta_model.learn_one(x, eta_true)

        self.fare_drift_input.learn_one(x)
        self.eta_drift_input.learn_one(x)

        self.fare_drift_scores.append(self.fare_drift_input.score_one(x))
        self.eta_drift_scores.append(self.eta_drift_input.score_one(x))

        self.fare_drift_output_flags.append(self.fare_drift_output.update(fare_error))
        self.eta_drift_output_flags.append(self.eta_drift_output.update(eta_error))

        heapq.heappush(self.top_fare_errors, (-fare_error, x, fare_true, fare_pred))
        heapq.heappush(self.top_eta_errors, (-eta_error, x, eta_true, eta_pred))

        self.top_fare_errors = self.top_fare_errors[:5]
        self.top_eta_errors = self.top_eta_errors[:5]

    def get_top_errors(self):
        fare_errors = [(-e[0], e[1], e[2], e[3]) for e in sorted(self.top_fare_errors)]
        eta_errors = [(-e[0], e[1], e[2], e[3]) for e in sorted(self.top_eta_errors)]
        return fare_errors, eta_errors

    def get_model_weights(self):
        fare_weights = self.fare_model[-1].weights if hasattr(self.fare_model[-1], "weights") else {}
        eta_weights = self.eta_model[-1].weights if hasattr(self.eta_model[-1], "weights") else {}
        return dict(fare_weights), dict(eta_weights)
