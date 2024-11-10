import pandas as pd
import numpy as np

# تحميل عدد الكسور العشرية المطلوبة لكل زوج من ملف coin_decimals.csv
decimals = pd.read_csv('coin_decimals.csv').set_index('feed_name').to_dict()['decimals']

# التوقع المبسط بناءً على آخر سعر متاح فقط
def predict_price_simple(data, feed_name):
    prices = data[data['feed_name'] == feed_name]['ask_price'].dropna()
    return prices.iloc[-1] if len(prices) > 0 else None

def main():
    n = int(input().strip())  
    c = int(input().strip())  
    k = int(input().strip())  

    latest_data = pd.DataFrame(columns=["feed_name", "ask_price"])

    while True:
        try:
            m = int(input().strip())
        except ValueError:
            continue
        
        if m == 0:
            break
        
        new_data = []
        current_round_feeds = set()  # العملات في الجولة الحالية
        for _ in range(m):
            line = input().strip()
            fields = line.split(',')
            record = {
                "feed_name": fields[3],
                "ask_price": float(fields[5]) if fields[5] else None
            }
            if record['feed_name'] and record['ask_price'] is not None:
                new_data.append(record)
                current_round_feeds.add(record['feed_name'])

        if new_data:
            new_df = pd.DataFrame(new_data)
            # دمج البيانات الحالية مع الجديدة، ثم الاحتفاظ فقط بأفضل التوقعات أو القيم الجديدة
            latest_data = pd.concat([latest_data, new_df], ignore_index=True)
            # الاحتفاظ بأحدث البيانات فقط بما يتوافق مع العملات النشطة في الجولة
            latest_data = latest_data[latest_data['feed_name'].isin(current_round_feeds)].tail(5000)

        predictions = []

        # التوقعات بناءً على العملات الموجودة في الجولة الحالية فقط
        for feed_name in sorted(current_round_feeds):
            predicted_price = predict_price_simple(latest_data, feed_name)
            if predicted_price is not None:
                rounded_price = round(predicted_price, decimals.get(feed_name, 5))
                formatted_price = f"{rounded_price:.10f}".rstrip('0').rstrip('.')
                predictions.append(f"{feed_name} {formatted_price}")

        # التأكد من عدد التوقعات يطابق C
        if len(predictions) < c:
            missing_feed_names = [fn for fn in current_round_feeds if not any(pred.startswith(fn) for pred in predictions)]
            for feed_name in missing_feed_names[:c - len(predictions)]:
                last_known_price = predict_price_simple(latest_data, feed_name) or 1.0
                rounded_price = round(last_known_price, decimals.get(feed_name, 5))
                formatted_price = f"{rounded_price:.10f}".rstrip('0').rstrip('.')
                predictions.append(f"{feed_name} {formatted_price}")
        
        print(len(predictions))  # طباعة عدد التوقعات
        
        for prediction in predictions:
            print(prediction)

        # تحديث الحدود الجديدة حسب المدخلات
        q = int(input().strip())
        for _ in range(q):
            input().strip()

if __name__ == "__main__":
    main()
