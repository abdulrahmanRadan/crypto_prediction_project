import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error
import xgboost as xgb

# تحميل البيانات
train_df = pd.read_csv('./data/train.csv')

# التعامل مع القيم المفقودة - سنستخدم fillna لتعبئة القيم المفقودة
train_df.fillna(train_df.mean(), inplace=True)  # أو استخدم ffill حسب الحاجة

# إذا كان هناك أعمدة نصية يجب تحويلها إلى قيم عددية
# استخدم LabelEncoder لتحويل النصوص إلى أرقام
label_encoder = LabelEncoder()

# افترض أن لدينا عمودًا يسمى 'category' يحتوي على قيم نصية
if 'category' in train_df.columns:
    train_df['category'] = label_encoder.fit_transform(train_df['category'])

# استخراج الميزات (Features) والهدف (Target)
# افترض أن الهدف هو 'target' وكل الأعمدة الأخرى هي الميزات
features = train_df.drop(columns=['target'])  # الأعمدة التي تحتوي على الميزات
target = train_df['target']  # العمود الذي يحتوي على الهدف

# تقسيم البيانات إلى تدريب واختبار
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# مقياس البيانات (إذا كانت البيانات عددية)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# تحويل البيانات إلى DMatrix (نوع البيانات الذي يستخدمه XGBoost)
train_dmatrix = xgb.DMatrix(X_train_scaled, label=y_train)
test_dmatrix = xgb.DMatrix(X_test_scaled, label=y_test)

# إعداد النموذج XGBoost
params = {
    'objective': 'reg:squarederror',  # هدفنا هو التنبؤ بالقيم العددية
    'max_depth': 6,
    'learning_rate': 0.1,
    'colsample_bytree': 0.8,  # لتقليل التداخل بين الأعمدة
    'subsample': 0.8,  # لتقليل التداخل بين البيانات
}

# تدريب النموذج باستخدام DMatrix
evals_result = {}
model = xgb.train(params=params, 
                  dtrain=train_dmatrix, 
                  num_boost_round=1000, 
                  evals=[(test_dmatrix, 'validation')],
                  early_stopping_rounds=50, 
                  evals_result=evals_result, 
                  verbose_eval=50)

# التنبؤ بالقيم
y_pred = model.predict(test_dmatrix)

# حساب MSE و RMSE
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# طباعة النتيجة النهائية
print(f"Training completed. RMSE on test data: {rmse:.4f}")

# عرض تقدم التدريب (RMSE) بشكل تدريجي
training_progress = evals_result['validation']['rmse']
for i, progress in enumerate(training_progress):
    print(f"Training progress: {i + 1}% - RMSE: {progress:.4f}")

# حفظ النموذج المدرب لاستخدامه لاحقًا
model.save_model('xgboost_model.json')  # حفظ النموذج باستخدام تنسيق JSON
