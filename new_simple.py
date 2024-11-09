import pandas as pd
import sys
from sklearn.linear_model import LinearRegression
import numpy as np

# تحميل عدد الكسور العشرية المطلوبة لكل زوج من ملف coin_decimals.csv
decimals = pd.read_csv('data/coin_decimals.csv').set_index('feed_name').to_dict()['decimals']

# تهيئة البيانات الأولية
def load_initial_data(file_path, n):
    data = pd.read_csv(file_path, nrows=n)
    return data

# وظيفة لتحديد أفضل 100 نقطة استنادًا إلى التقلبات حول السعر الوسطي
def select_best_points(data, feed_name):
    prices = data[data['feed_name'] == feed_name][['TimeSinceStart', 'ask_price']].dropna()
    
    if len(prices) <= 100:
        return prices
    
    # حساب السعر الوسطي
    median_price = prices['ask_price'].median()
    
    # إضافة عمود جديد لحساب الانحراف عن السعر الوسطي
    prices['deviation'] = abs(prices['ask_price'] - median_price)
    
    # فرز البيانات بناءً على الانحراف الأصغر والاحتفاظ بأفضل 100 نقطة فقط
    best_points = prices.nsmallest(100, 'deviation')
    
    # إعادة تعيين الفهرس وإزالة عمود الانحراف
    return best_points[['TimeSinceStart', 'ask_price']].reset_index(drop=True)

# وظيفة التوقع باستخدام الانحدار الخطي
def predict_price(data, feed_name):
    # اختيار أفضل 100 نقطة
    prices = select_best_points(data, feed_name)
    if len(prices) < 2:
        return None
    
    # تجهيز البيانات لنموذج الانحدار الخطي
    X = prices['TimeSinceStart'].values.reshape(-1, 1)
    y = prices['ask_price'].values
    
    # تدريب نموذج الانحدار الخطي
    model = LinearRegression()
    model.fit(X, y)
    
    # التنبؤ بالسعر عند نهاية الجولة الحالية (وقت 90 ثانية)
    predicted_price = model.predict(np.array([[90]]))[0]
    
    # التأكد من أن السعر المتنبأ غير سالب
    if predicted_price < 0:
        predicted_price = 0.0
    
    return predicted_price

# البرنامج الأساسي
def main():
    # قراءة المدخلات الأولية من النظام
    n = int(input().strip())  # عدد جولات التصويت الأولية
    c = int(input().strip())  # عدد أزواج العملات
    k = int(input().strip())  # عدد الأسطر التي يجب قراءتها من بيانات التدريب الأولية

    # تحميل البيانات الأولية من الملف
    data = load_initial_data('data/train.csv', k)
    
    # تخزين آخر بيانات للتمكن من التحديث
    latest_data = data.copy()
    
    # تخزين الحدود السابقة
    previous_bounds = {}

    # تنفيذ التنبؤ لكل جولة
    while True:
        # قراءة عدد الأسطر التي سيتم تمريرها من البيانات الجديدة للجولة الحالية
        try:
            m = int(input().strip())
        except ValueError:
            # في حال حدوث خطأ، نتجاهل السطر الحالي
            continue
        
        if m == 0:
            break
        
        # تحديث البيانات بقراءة الأسطر الجديدة من البيانات
        new_data = []
        for _ in range(m):
            line = input().strip()
            fields = line.split(',')
            if len(fields) < 12:
                continue
            record = {
                "voting_round": int(fields[0]),
                "exchange_id": int(fields[1]),
                "currency_id": int(fields[2]),
                "feed_name": fields[3],
                "TimeSinceStart": float(fields[4]),
                "ask_price": float(fields[5]) if fields[5] else None,
                "ask_quantity": float(fields[6]) if fields[6] else None,
                "bid_price": float(fields[7]) if fields[7] else None,
                "bid_quantity": float(fields[8]) if fields[8] else None,
                "quote": fields[11] if fields[11] else None
            }
            new_data.append(record)

        if new_data:
            new_df = pd.DataFrame(new_data)
            
            # استبدال القيم المفقودة بآخر قيمة معروفة باستخدام forward fill
            new_df.fillna(method='ffill', inplace=True)
            
            # دمج البيانات الجديدة مع البيانات السابقة، والاحتفاظ فقط بآخر 1000 سطر لتقليل الذاكرة المستخدمة
            latest_data = pd.concat([latest_data, new_df], ignore_index=True).tail(1000)

        # حساب التوقعات لكل زوج من العملات
        predictions = []
        for feed_name in latest_data['feed_name'].unique():
            predicted_price = predict_price(latest_data, feed_name)
            if predicted_price is not None:
                # تعديل التوقع ليكون ضمن الحدود السابقة إن وجدت
                if feed_name in previous_bounds:
                    lower, upper = previous_bounds[feed_name]
                    # ضبط التوقع ليقع ضمن الحدود السابقة
                    predicted_price = max(lower, min(predicted_price, upper))
                
                # تقليل العدد العشري حسب coin_decimals.csv
                rounded_price = round(predicted_price, decimals.get(feed_name, 5))
                predictions.append(f"{feed_name} {rounded_price}")
        
        # إخراج التوقعات للتفاعل مع الـ Tester
        print(len(predictions))
        for prediction in predictions:
            print(prediction)
        
        # استقبال التقييم للحدود العليا والدنيا من Tester
        q = int(input().strip())
        for _ in range(q):
            line = input().strip()
            fields = line.split()
            feed_name = fields[0]
            lower_bound = float(fields[1])
            upper_bound = float(fields[2])
            previous_bounds[feed_name] = (lower_bound, upper_bound)

# تشغيل البرنامج
if __name__ == "__main__":
    main()
