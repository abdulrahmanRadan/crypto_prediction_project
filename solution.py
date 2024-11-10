import pandas as pd
import sys
from sklearn.linear_model import LinearRegression
import numpy as np

# تحميل عدد الكسور العشرية المطلوبة لكل زوج من ملف coin_decimals.csv
decimals = pd.read_csv('coin_decimals.csv').set_index('feed_name').to_dict()['decimals']

# تهيئة البيانات الأولية
def load_initial_data(file_path, n):
    data = pd.read_csv(file_path, nrows=n)
    return data

# وظيفة التوقع باستخدام الانحدار الخطي
def predict_price(data, feed_name):
    prices = data[data['feed_name'] == feed_name][['TimeSinceStart', 'ask_price']].dropna().tail(10)
    if len(prices) < 2:
        return None
    
    X = prices['TimeSinceStart'].values.reshape(-1, 1)
    y = prices['ask_price'].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    predicted_price = model.predict(np.array([[90]]))[0]
    if predicted_price < 0:
        predicted_price = 0.0
    
    return predicted_price

# التحقق من صحة وترتيب البيانات
def parse_record(fields):
    if len(fields) != 12:  # التأكد من أن عدد الأعمدة يساوي 12
        return None

    try:
        record = {
            "voting_round": int(fields[0]),
            "exchange_id": int(fields[1]),
            "currency_id": int(fields[2]),
            "feed_name": fields[3],
            "TimeSinceStart": float(fields[4]) if fields[4] else None,
            "ask_price": float(fields[5]) if fields[5] else None,
            "ask_quantity": float(fields[6]) if fields[6] else None,
            "bid_price": float(fields[7]) if fields[7] else None,
            "bid_quantity": float(fields[8]) if fields[8] else None,
            "LowerBound": float(fields[9]) if fields[9] else None,
            "UpperBound": float(fields[10]) if fields[10] else None,
            "quote": fields[11] if fields[11] else None
        }
    except ValueError:
        return None

    if not record["feed_name"]:
        return None

    return record

def main():
    # فتح الملف لتسجيل المخرجات
    log_file = open("output_log.txt", "w")

    n = int(input().strip())  
    c = int(input().strip())  
    k = int(input().strip())  

    data = load_initial_data('data/train.csv', k)
    latest_data = data.copy()
    previous_bounds = {}

    while True:
        try:
            m = int(input().strip())
        except ValueError:
            continue
        
        if m == 0:
            break
        
        new_data = []
        for _ in range(m):
            line = input().strip()
            fields = line.split(',')
            record = parse_record(fields)
            if record:
                new_data.append(record)

        if new_data:
            new_df = pd.DataFrame(new_data)
            new_df.fillna(method='ffill', inplace=True)
            latest_data = pd.concat([latest_data, new_df], ignore_index=True).tail(1000)

        predictions = []
        feed_names = latest_data['feed_name'].unique()
        for feed_name in feed_names:
            predicted_price = predict_price(latest_data, feed_name)
            if predicted_price is not None:
                if feed_name in previous_bounds:
                    lower, upper = previous_bounds[feed_name]
                    predicted_price = max(lower, min(predicted_price, upper))
                
                rounded_price = round(predicted_price, decimals.get(feed_name, 5))
                # تحويل السعر المتنبأ به إلى صيغة عشرية محددة
                formatted_price = f"{rounded_price:.10f}".rstrip('0').rstrip('.')
                predictions.append(f"{feed_name} {formatted_price}")

        # إكمال التوقعات إذا كان العدد أقل من C
        missing_feed_names = [fn for fn in latest_data['feed_name'].unique() if not any(pred.startswith(fn) for pred in predictions)]
        for feed_name in missing_feed_names[:c - len(predictions)]:
            last_known_price = latest_data[latest_data['feed_name'] == feed_name]['ask_price'].dropna().values[-1] if not latest_data[latest_data['feed_name'] == feed_name]['ask_price'].dropna().empty else 1.0
            rounded_price = round(last_known_price, decimals.get(feed_name, 5))
            formatted_price = f"{rounded_price:.10f}".rstrip('0').rstrip('.')
            predictions.append(f"{feed_name} {formatted_price}")

        # التأكد من أن عدد التوقعات مطابق لـ C
        if len(predictions) > c:
            predictions = predictions[:c]

        # تسجيل عدد التوقعات في الملف وطباعة المخرجات
        log_file.write(f"{len(predictions)}\n")
        print(len(predictions))  # طباعة عدد التوقعات
        
        for prediction in predictions:
            log_file.write(f"{prediction}\n")
            print(prediction)  # طباعة التوقع

        # تأكيد كتابة المخرجات على القرص
        log_file.flush()

        q = int(input().strip())
        for _ in range(q):
            line = input().strip()
            fields = line.split()
            feed_name = fields[0]
            lower_bound = float(fields[1])
            upper_bound = float(fields[2])
            previous_bounds[feed_name] = (lower_bound, upper_bound)

    # إغلاق ملف التسجيل
    log_file.close()

if __name__ == "__main__":
    main()
