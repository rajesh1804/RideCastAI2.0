from river import linear_model, metrics, preprocessing, compose, anomaly
from river import optim

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

        # Drift detector (per model)
        self.fare_drift = anomaly.HalfSpaceTrees(seed=42)
        self.eta_drift = anomaly.HalfSpaceTrees(seed=42)

        # Rolling metrics
        self.fare_mae = metrics.MAE()
        self.fare_rmse = metrics.RMSE()
        self.eta_mae = metrics.MAE()
        self.eta_rmse = metrics.RMSE()

    def predict(self, x):
        fare = self.fare_model.predict_one(x)
        eta = self.eta_model.predict_one(x)

        return {
            "fare_pred": max(0.0, fare),
            "eta_pred": max(0.0, eta)
        }

    def update(self, x, y):
        fare_true = y['fare_amount']
        eta_true = y['duration_min']

        # Update drift detectors
        self.fare_drift.learn_one(x)
        self.eta_drift.learn_one(x)

        # Update metrics
        fare_pred = self.fare_model.predict_one(x)
        eta_pred = self.eta_model.predict_one(x)

        self.fare_mae.update(fare_true, fare_pred)
        self.fare_rmse.update(fare_true, fare_pred)
        self.eta_mae.update(eta_true, eta_pred)
        self.eta_rmse.update(eta_true, eta_pred)

        # Online model update
        self.fare_model.learn_one(x, fare_true)
        self.eta_model.learn_one(x, eta_true)
