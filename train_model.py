import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
from sklearn.preprocessing import MinMaxScaler

# تحميل البيانات من train.csv
train_data = pd.read_csv("data/train.csv")

# معالجة البيانات وتطبيعها
def preprocess_data(data):
    # إزالة القيم المفقودة باستخدام القيمة السابقة
    data = data.ffill()  # Use forward fill to fill missing values
    data = data[data['ask_price'] > 0]
    data = data[data['bid_price'] > 0]
    
    # استخراج الأسعار وتطبيعها
    prices = data['ask_price'].values
    time_steps = np.arange(len(prices)).reshape(-1, 1)
    
    scaler = MinMaxScaler()
    prices_scaled = scaler.fit_transform(prices.reshape(-1, 1))
    
    return time_steps, prices_scaled, scaler

# تجهيز البيانات للتدريب
time_steps, prices_scaled, scaler = preprocess_data(train_data)

# إعداد وتدريب النموذج باستخدام XGBoost
params = {
    'max_depth': 6,
    'eta': 0.1,
    'objective': 'reg:squarederror',
    'subsample': 0.8,
    'colsample_bytree': 0.8
}

dtrain = xgb.DMatrix(time_steps, label=prices_scaled)
model = xgb.train(params, dtrain, num_boost_round=500)

# حفظ النموذج والمحول scaler
with open("xgb_model.pkl", "wb") as model_file:
    pickle.dump((model, scaler), model_file)

print("تم تدريب النموذج وحفظه بنجاح!")

# تحسين النموذج تدريجياً مع كل توقع
def update_model(new_data):
    # معالجة البيانات الجديدة وتطبيعها
    time_steps_new, prices_scaled_new, _ = preprocess_data(new_data)
    dtrain_new = xgb.DMatrix(time_steps_new, label=prices_scaled_new)
    
    # تحسين النموذج عن طريق تدريبه باستخدام البيانات الجديدة
    model.update(dtrain_new, iteration_range=(0, 100))
    with open("xgb_model.pkl", "wb") as model_file:
        pickle.dump((model, scaler), model_file)
    print("تم تحسين النموذج باستخدام بيانات جديدة.")
