import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from collections import defaultdict

# Load decimal places for each pair from coin_decimals.csv
decimals = pd.read_csv('coin_decimals.csv').set_index('feed_name').to_dict()['decimals']

class PricePrediction:
    def __init__(self):
        self.previous_bounds = {}  # Stores min and max bounds for pairs
        self.historical_data = defaultdict(list)  # Stores historical data for each pair (only last 5 records)
        self.previous_predictions = {}  # Stores previous predictions for pairs

    def update_data(self, new_data):
        for entry in new_data:
            feed_name = entry['feed_name']
            self.historical_data[feed_name].append((entry['TimeSinceStart'], entry['ask_price']))
            if len(self.historical_data[feed_name]) > 5:
                self.historical_data[feed_name].pop(0)

    def predict_price(self, feed_name):
        prices = pd.DataFrame(self.historical_data[feed_name], columns=["TimeSinceStart", "ask_price"])
        if len(prices) < 2:
            return None

        prices = prices.tail(5)  # Use last 5 records for prediction
        X = prices['TimeSinceStart'].values.reshape(-1, 1)  # Independent variable
        y = prices['ask_price'].values  # Dependent variable

        model = LinearRegression()
        model.fit(X, y)
        
        predicted_price = model.predict(np.array([[90]]))[0]
        
        if predicted_price < 0:
            predicted_price = 0.0
        
        return predicted_price

    def get_predictions(self, required_feeds):
        predictions = []
        for feed_name in sorted(required_feeds):
            predicted_price = self.predict_price(feed_name)
            
            if predicted_price is None:
                predicted_price = self.previous_predictions.get(feed_name, self.historical_data[feed_name][-1][1] if self.historical_data[feed_name] else 1.0)

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

def main():
    prediction_model = PricePrediction()

    n = int(input().strip())  # Number of pairs
    c = int(input().strip())  # Number of rounds
    k = int(input().strip())  # Number of cases per round

    while True:
        try:
            m = int(input().strip())
        except ValueError:
            continue
        
        if m == 0:
            break
        
        new_data = []
        required_feeds = set()  # Required pairs for this round
        for _ in range(m):
            line = input().strip()
            fields = line.split(',')
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
            line = input().strip()
            fields = line.split()
            feed_name = fields[0]
            lower_bound = float(fields[1])
            upper_bound = float(fields[2])
            prediction_model.update_bounds(feed_name, lower_bound, upper_bound)

if __name__ == "__main__":
    main()

# Score = 11.522872032426172