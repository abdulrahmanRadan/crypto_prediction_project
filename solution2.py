import pandas as pd
import numpy as np

# تحميل عدد الكسور العشرية المطلوبة لكل زوج من ملف coin_decimals.csv
decimals = pd.read_csv('coin_decimals.csv').set_index('feed_name').to_dict()['decimals']

# تهيئة البيانات الأولية
def load_initial_data(file_path, n):
    data = pd.read_csv(file_path, nrows=n)
    return data

# وظيفة التوقع المبسطة بناءً على آخر سعر متاح فقط
def predict_price_simple(data, feed_name):
    # استرجاع آخر سعر متاح لهذه العملة
    prices = data[data['feed_name'] == feed_name]['ask_price'].dropna()
    if len(prices) == 0:
        return None
    return prices.iloc[-1]

def main():
    log_file = open("output_log.txt", "w")

    n = int(input().strip())  
    c = int(input().strip())  
    k = int(input().strip())  

    data = load_initial_data('data/train.csv', k)
    latest_data = data.copy()

    # ترتيب أسماء العملات كما هي في ملف decimals لضمان التطابق
    feed_names_in_order = list(decimals.keys())

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
            record = {
                "feed_name": fields[3],
                "ask_price": float(fields[5]) if fields[5] else None
            }
            if record['feed_name'] and record['ask_price'] is not None:
                new_data.append(record)

        if new_data:
            new_df = pd.DataFrame(new_data)
            latest_data = pd.concat([latest_data, new_df], ignore_index=True).tail(1000)

        predictions = []

        # التوقعات بالترتيب المطلوب وضمان الأعداد العشرية لكل عملة
        for feed_name in feed_names_in_order:
            predicted_price = predict_price_simple(latest_data, feed_name)
            if predicted_price is not None:
                rounded_price = round(predicted_price, decimals.get(feed_name, 5))
                formatted_price = f"{rounded_price:.10f}".rstrip('0').rstrip('.')
                predictions.append(f"{feed_name} {formatted_price}")

        if len(predictions) > c:
            predictions = predictions[:c]

        log_file.write(f"{len(predictions)}\n")
        print(len(predictions)) 
        
        for prediction in predictions:
            log_file.write(f"{prediction}\n")
            print(prediction)

        log_file.flush()

        # تحديث الحدود الجديدة حسب المدخلات
        q = int(input().strip())
        for _ in range(q):
            input().strip()

    log_file.close()

if __name__ == "__main__":
    main()
# نفذ بشكل سريع 
# يحل مشكلة انه يتوقف ولاكنه فيه مشكلة 