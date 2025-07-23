# from river import linear_model, metrics, preprocessing, compose, anomaly, optim
# import statistics
# import random

# class RideModel:
#     def __init__(self):
#         self.scaler = preprocessing.StandardScaler()
#         self.encoder = preprocessing.OneHotEncoder()

#         self.features = compose.TransformerUnion(
#             ('scaled', self.scaler),
#             ('encoded', self.encoder)
#         )

#         self.fare_model = compose.Pipeline(
#             self.features,
#             linear_model.LinearRegression(optimizer=optim.SGD(0.01))
#         )

#         self.eta_model = compose.Pipeline(
#             self.features,
#             linear_model.LinearRegression(optimizer=optim.SGD(0.01))
#         )

#         # Drift detection with distinct seeds for visual divergence
#         self.fare_drift = anomaly.HalfSpaceTrees(seed=42, n_trees=5, height=2, window_size=10)
#         self.eta_drift = anomaly.HalfSpaceTrees(seed=24, n_trees=5, height=2, window_size=10)

#         # Rolling metrics
#         self.fare_mae = metrics.MAE()
#         self.fare_rmse = metrics.RMSE()
#         self.eta_mae = metrics.MAE()
#         self.eta_rmse = metrics.RMSE()

#         # Buffers
#         self.total_updates = 0
#         self.fare_drift_scores = []
#         self.eta_drift_scores = []
#         self.fare_rmse_history = []
#         self.eta_rmse_history = []

#         # Baseline buffers
#         self.fare_values = []
#         self.eta_values = []
#         self.fare_baseline_rmse = []
#         self.eta_baseline_rmse = []

#     def predict(self, x):
#         fare = self.fare_model.predict_one(x)
#         eta = self.eta_model.predict_one(x)
#         return {
#             "fare_pred": max(0.0, fare),
#             "eta_pred": max(0.0, eta)
#         }

#     def update(self, x, y):
#         self.total_updates += 1

#         fare_true = y['fare_amount']
#         eta_true = y['duration_min']

#         # ✅ Inject controlled drift for demo purposes
#         if 10 <= self.total_updates <= 20:
#             fare_true *= 1.2  # simulate fare surge
#         if 20 <= self.total_updates <= 30:
#             eta_true += 5 * random.random()  # simulate ETA noise

#         # Predict before update
#         fare_pred = self.fare_model.predict_one(x)
#         eta_pred = self.eta_model.predict_one(x)

#         # Update metrics
#         self.fare_mae.update(fare_true, fare_pred)
#         self.fare_rmse.update(fare_true, fare_pred)
#         self.eta_mae.update(eta_true, eta_pred)
#         self.eta_rmse.update(eta_true, eta_pred)

#         self.fare_rmse_history.append(self.fare_rmse.get())
#         self.eta_rmse_history.append(self.eta_rmse.get())

#         # Baseline model (mean predictor)
#         self.fare_values.append(fare_true)
#         self.eta_values.append(eta_true)

#         fare_mean = statistics.mean(self.fare_values)
#         eta_mean = statistics.mean(self.eta_values)

#         fare_baseline_rmse = ((fare_true - fare_mean) ** 2) ** 0.5
#         eta_baseline_rmse = ((eta_true - eta_mean) ** 2) ** 0.5

#         self.fare_baseline_rmse.append(fare_baseline_rmse)
#         self.eta_baseline_rmse.append(eta_baseline_rmse)

#         # Online model update
#         self.fare_model.learn_one(x, fare_true)
#         self.eta_model.learn_one(x, eta_true)

#         # Drift
#         self.fare_drift.learn_one(x)
#         self.eta_drift.learn_one(x)

#         fare_score = self.fare_drift.score_one(x)
#         eta_score = self.eta_drift.score_one(x)

#         print(f"[Update] Total updates: {self.total_updates}")
#         print(f"[Drift Scores] Fare: {fare_score:.3f} | ETA: {eta_score:.3f}")

#         self.fare_drift_scores.append(fare_score)
#         self.eta_drift_scores.append(eta_score)


from river import linear_model, metrics, preprocessing, compose, anomaly, drift, optim
import statistics
import random

class RideModel:
    def __init__(self):
        self.scaler = preprocessing.StandardScaler()
        self.encoder = preprocessing.OneHotEncoder()

        self.features = compose.TransformerUnion(
            ('scaled', self.scaler),
            ('encoded', self.encoder)
        )

        self.fare_model = compose.Pipeline(
            self.features,
            linear_model.LinearRegression(optimizer=optim.SGD(0.01))
        )

        self.eta_model = compose.Pipeline(
            self.features,
            linear_model.LinearRegression(optimizer=optim.SGD(0.01))
        )

        # Input drift
        self.fare_drift_input = anomaly.HalfSpaceTrees(seed=42, n_trees=5, height=2, window_size=10)
        self.eta_drift_input = anomaly.HalfSpaceTrees(seed=24, n_trees=5, height=2, window_size=10)

        # Output drift
        self.fare_drift_output = drift.KSWIN(alpha=0.005, window_size=50, stat_size=20)
        self.eta_drift_output = drift.ADWIN()

        # Rolling metrics
        self.fare_mae = metrics.MAE()
        self.fare_rmse = metrics.RMSE()
        self.eta_mae = metrics.MAE()
        self.eta_rmse = metrics.RMSE()

        # Buffers
        self.total_updates = 0
        self.fare_drift_scores = []
        self.eta_drift_scores = []
        self.fare_rmse_history = []
        self.eta_rmse_history = []

        self.fare_drift_output_flags = []
        self.eta_drift_output_flags = []

        # Baseline buffers
        self.fare_values = []
        self.eta_values = []
        self.fare_baseline_rmse = []
        self.eta_baseline_rmse = []

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

        # Inject controlled drift
        if 10 <= self.total_updates <= 20:
            fare_true *= 1.2
        if 20 <= self.total_updates <= 30:
            eta_true += 5 * random.random()

        # Predict before update
        fare_pred = self.fare_model.predict_one(x)
        eta_pred = self.eta_model.predict_one(x)

        # Update metrics
        self.fare_mae.update(fare_true, fare_pred)
        self.fare_rmse.update(fare_true, fare_pred)
        self.eta_mae.update(eta_true, eta_pred)
        self.eta_rmse.update(eta_true, eta_pred)

        self.fare_rmse_history.append(self.fare_rmse.get())
        self.eta_rmse_history.append(self.eta_rmse.get())

        # Baseline RMSE
        self.fare_values.append(fare_true)
        self.eta_values.append(eta_true)

        fare_mean = statistics.mean(self.fare_values)
        eta_mean = statistics.mean(self.eta_values)

        self.fare_baseline_rmse.append(((fare_true - fare_mean) ** 2) ** 0.5)
        self.eta_baseline_rmse.append(((eta_true - eta_mean) ** 2) ** 0.5)

        # Online model update
        self.fare_model.learn_one(x, fare_true)
        self.eta_model.learn_one(x, eta_true)

        # Input drift
        self.fare_drift_input.learn_one(x)
        self.eta_drift_input.learn_one(x)

        fare_input_score = self.fare_drift_input.score_one(x)
        eta_input_score = self.eta_drift_input.score_one(x)

        self.fare_drift_scores.append(fare_input_score)
        self.eta_drift_scores.append(eta_input_score)

        # Output drift (based on absolute error)
        fare_error = abs(fare_true - fare_pred)
        eta_error = abs(eta_true - eta_pred)

        fare_drift_flag = self.fare_drift_output.update(fare_error)
        eta_drift_flag = self.eta_drift_output.update(eta_error)

        self.fare_drift_output_flags.append(fare_drift_flag)
        self.eta_drift_output_flags.append(eta_drift_flag)

        print(f"[Update] #{self.total_updates} | Fare Drift (Input): {fare_input_score:.3f} | ETA Drift (Input): {eta_input_score:.3f}")
        if fare_drift_flag:
            print("⚠️ Output Drift Detected in Fare Prediction!")
        if eta_drift_flag:
            print("⚠️ Output Drift Detected in ETA Prediction!")
