from river import linear_model, metrics, preprocessing, compose, anomaly, optim

class RideModel:
    def __init__(self):
        # Feature preprocessor
        self.scaler = preprocessing.StandardScaler()
        self.encoder = preprocessing.OneHotEncoder()

        # Feature pipeline
        self.features = compose.TransformerUnion(
            ('scaled', self.scaler),
            ('encoded', self.encoder)
        )

        # Models for fare and duration
        self.fare_model = compose.Pipeline(
            self.features,
            linear_model.LinearRegression(optimizer=optim.SGD(0.01))
        )

        self.eta_model = compose.Pipeline(
            self.features,
            linear_model.LinearRegression(optimizer=optim.SGD(0.01))
        )

        # Drift detector with scoring enabled
        self.fare_drift = anomaly.HalfSpaceTrees(
            seed=42, n_trees=10, height=3, window_size=25
        )
        self.eta_drift = anomaly.HalfSpaceTrees(
            seed=42, n_trees=10, height=3, window_size=25
        )

        # Rolling metrics
        self.fare_mae = metrics.MAE()
        self.fare_rmse = metrics.RMSE()
        self.eta_mae = metrics.MAE()
        self.eta_rmse = metrics.RMSE()

        # Drift buffers
        self.total_updates = 0
        self.fare_drift_scores = []
        self.eta_drift_scores = []

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

        # Predict before update
        fare_pred = self.fare_model.predict_one(x)
        eta_pred = self.eta_model.predict_one(x)

        # Update metrics
        self.fare_mae.update(fare_true, fare_pred)
        self.fare_rmse.update(fare_true, fare_pred)
        self.eta_mae.update(eta_true, eta_pred)
        self.eta_rmse.update(eta_true, eta_pred)

        # Online model update
        self.fare_model.learn_one(x, fare_true)
        self.eta_model.learn_one(x, eta_true)

        # Drift scoring + update
        self.fare_drift.learn_one(x)
        self.eta_drift.learn_one(x)

        fare_score = self.fare_drift.score_one(x)
        eta_score = self.eta_drift.score_one(x)

        print(f"[Update] Total updates: {self.total_updates}")
        print(f"[Drift Scores] Fare: {fare_score:.3f} | ETA: {eta_score:.3f}")

        self.fare_drift_scores.append(fare_score)
        self.eta_drift_scores.append(eta_score)
