import pandas as pd

# تحميل ملف train.csv الأصلي
original_file = 'data/train.csv'
# اسم الملف الجديد الذي سيتم إنشاؤه
new_file = 'data/train_small.csv'

# عدد الدورات التي نريد الاحتفاظ بها في الملف الجديد
num_rounds_to_keep = 100  # يمكنك تعديل هذا العدد

# قراءة الملف كاملاً
data = pd.read_csv(original_file)

# استخراج آخر القيم للدورات المحددة
last_voting_rounds = data['voting_round'].unique()[-num_rounds_to_keep:]
data_small = data[data['voting_round'].isin(last_voting_rounds)]

# حفظ البيانات في ملف جديد
data_small.to_csv(new_file, index=False)

print(f"تم إنشاء الملف {new_file} بنجاح، ويحتوي على آخر {num_rounds_to_keep} دورة من {original_file}.")
