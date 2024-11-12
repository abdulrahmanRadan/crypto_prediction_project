import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
from collections import defaultdict

# Load decimal places for each pair from coin_decimals.csv
decimals = pd.read_csv('coin_decimals.csv').set_index('feed_name').to_dict()['decimals']

# Strategy pattern for models
class ModelStrategy:
    def train_and_predict(self, X, y, future_point):
        raise NotImplementedError("Subclasses should implement this!")

class RidgeModelStrategy(ModelStrategy):
    def train_and_predict(self, X, y, future_point):
        model = Ridge(alpha=1.0)
        model.fit(X, y)
        return model.predict(np.array([future_point]))[0]

class XGBoostModelStrategy(ModelStrategy):
    def train_and_predict(self, X, y, future_point):
        model = XGBRegressor(n_estimators=100, learning_rate=0.1)
        model.fit(X, y)
        return model.predict(np.array([future_point]))[0]

# Main class using the strategy pattern
class PricePrediction:
    def __init__(self, model_strategy):
        self.model_strategy = model_strategy
        self.previous_bounds = {}
        self.historical_data = defaultdict(list)
        self.previous_predictions = {}

    def update_data(self, new_data):
        for entry in new_data:
            feed_name = entry['feed_name']
            self.historical_data[feed_name].append((entry['TimeSinceStart'], entry['ask_price']))
            if len(self.historical_data[feed_name]) > 20:  # Keep last 20 records
                self.historical_data[feed_name].pop(0)

    def predict_price(self, feed_name):
        prices = pd.DataFrame(self.historical_data[feed_name], columns=["TimeSinceStart", "ask_price"])
        if len(prices) < 5:
            return None

        # Feature engineering
        prices['price_diff'] = prices['ask_price'].diff().fillna(0)
        prices['rolling_mean'] = prices['ask_price'].rolling(window=3).mean().fillna(prices['ask_price'])
        X = prices[['TimeSinceStart', 'price_diff', 'rolling_mean']].values
        y = prices['ask_price'].values
        future_point = [90, 0, prices['rolling_mean'].iloc[-1]]

        predicted_price = self.model_strategy.train_and_predict(X, y, future_point)
        return max(predicted_price, 0.0)

    def get_predictions(self, required_feeds):
        predictions = []
        for feed_name in sorted(required_feeds):
            predicted_price = self.predict_price(feed_name)
            if predicted_price is None:
                predicted_price = self.previous_predictions.get(feed_name, 1.0)

            if feed_name in self.previous_bounds:
                lower, upper = self.previous_bounds[feed_name]
                predicted_price = max(lower, min(predicted_price, upper))

            rounded_price = round(predicted_price, decimals.get(feed_name, 5))
            formatted_price = f"{rounded_price:.10f}".rstrip('0').rstrip('.')
            predictions.append(f"{feed_name} {formatted_price}")
            self.previous_predictions[feed_name] = predicted_price

        return predictions

    def update_bounds(self, feed_name, lower_bound, upper_bound):
        self.previous_bounds[feed_name] = (lower_bound, upper_bound)

# Main function
def main():
    prediction_model = PricePrediction(XGBoostModelStrategy())
    n = int(input().strip())
    c = int(input().strip())
    k = int(input().strip())

    while True:
        try:
            m = int(input().strip())
        except ValueError:
            continue
        
        if m == 0:
            break

        new_data = []
        required_feeds = set()
        for _ in range(m):
            fields = input().strip().split(',')
            feed_name = fields[3]
            record = {
                "feed_name": feed_name,
                "TimeSinceStart": float(fields[4]) if fields[4] else None,
                "ask_price": float(fields[5]) if fields[5] else None
            }
            if record['feed_name'] and record['ask_price'] is not None:
                new_data.append(record)
                required_feeds.add(feed_name)

        prediction_model.update_data(new_data)
        predictions = prediction_model.get_predictions(required_feeds)

        print(len(predictions))
        for prediction in predictions:
            print(prediction)

        q = int(input().strip())
        for _ in range(q):
            fields = input().strip().split()
            feed_name = fields[0]
            lower_bound = float(fields[1])
            upper_bound = float(fields[2])
            prediction_model.update_bounds(feed_name, lower_bound, upper_bound)

if __name__ == "__main__":
    main()
#Score = 11.654269297581399