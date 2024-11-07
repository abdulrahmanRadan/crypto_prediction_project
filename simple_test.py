import sys
import numpy as np
import pandas as pd
import pickle
import logging
from collections import defaultdict
import xgboost as xgb

logging.basicConfig(level=logging.DEBUG)

# دالة لاستبدال القيم المفقودة بالوسيط (Median) أو المتوسط
def replace_missing_values(prices):
    if not prices or all(p is None or p <= 0 for p in prices):
        return [0] * len(prices)
    
    valid_prices = [p for p in prices if p is not None and p > 0]
    median = np.median(valid_prices) if valid_prices else 0
    return [median if p is None or p <= 0 else p for p in prices]

# دالة لقراءة ملف decimals وتخزينه في قاموس
def read_decimal_file(file_path):
    df = pd.read_csv(file_path)
    decimal_dict = {row['feed_name']: row['decimals'] for _, row in df.iterrows()}
    return decimal_dict

# تحميل النموذج والمحول scaler من الملف
with open("xgb_model.pkl", "rb") as model_file:
    model, scaler = pickle.load(model_file)

def generate_prediction(prices, time_steps, lower_bound, upper_bound):
    if len(prices) < 2:
        return None
    
    logging.debug(f"Prices: {prices}")
    logging.debug(f"Time Steps: {time_steps}")

    # تنسيق البيانات لاستخدامها في التنبؤ
    X = np.array(time_steps).reshape(-1, 1)
    y = np.array(prices).reshape(-1, 1)

    # تطبيع الأسعار باستخدام المحول المحفوظ
    y_scaled = scaler.transform(y)
    dtest = xgb.DMatrix(X, label=y_scaled)

    # التنبؤ بالقيمة التالية
    next_time_step = np.array([[len(prices)]])
    dnext = xgb.DMatrix(next_time_step)
    next_price_scaled = model.predict(dnext)
    next_price = scaler.inverse_transform(next_price_scaled.reshape(-1, 1))[0][0]

    # ضبط التوقع ليكون ضمن الحدود
    next_price = max(lower_bound, min(next_price, upper_bound))
    
    logging.debug(f"Predicted Price (bounded): {next_price}")

    return round(next_price, 5)

def main():
    decimal_file_path = "coin_decimals.csv"
    decimals = read_decimal_file(decimal_file_path)

    N = int(sys.stdin.readline().strip())
    C = int(sys.stdin.readline().strip())
    K = int(sys.stdin.readline().strip())

    price_data = []
    for _ in range(K):
        row = sys.stdin.readline().strip().split(',')
        price_data.append(row)

    prices = defaultdict(list)
    bounds = {}

    # معالجة البيانات
    for row in price_data:
        feed_name = row[3]
        ask_price = float(row[5]) if row[5] else None
        bid_price = float(row[7]) if row[7] else None

        ask_price = replace_missing_values([ask_price])[0]
        bid_price = replace_missing_values([bid_price])[0]

        lower_bound = float(row[9]) if row[9] else 0
        upper_bound = float(row[10]) if row[10] else float('inf')

        prices[feed_name].append(ask_price)
        bounds[feed_name] = (lower_bound, upper_bound)

    predictions = []

    for feed_name, price_list in prices.items():
        if price_list:
            time_steps = list(range(len(price_list)))
            lower_bound, upper_bound = bounds[feed_name]
            prediction = generate_prediction(price_list, time_steps, lower_bound, upper_bound)
            if prediction is not None:
                decimals_count = decimals.get(feed_name, 5)
                predictions.append(f"{feed_name} {prediction:.{decimals_count}f}")
            else:
                predictions.append(f"{feed_name} Insufficient data for prediction")
        else:
            predictions.append(f"{feed_name} No data for prediction")

    if 0 <= len(predictions) <= 50:
        print(len(predictions))
        for prediction in predictions:
            print(prediction)
    else:
        print("Error: Number of predictions out of bounds.")
    
    Q = int(sys.stdin.readline().strip())
    for _ in range(Q):
        sys.stdin.readline()

if __name__ == "__main__":
    main()
