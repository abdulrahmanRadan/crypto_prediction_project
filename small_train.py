import pandas as pd

# قراءة ملف CSV الأصلي
input_file = 'data/train.csv'
output_file = 'data/train_200_rounds.csv'

# تحميل البيانات مع قراءة عمود voting_round فقط لتقليل الاستهلاك
data = pd.read_csv(input_file, usecols=['voting_round'])

# الحصول على 200 قيمة فريدة فقط من voting_round
unique_rounds = data['voting_round'].unique()[:300]

# إعادة تحميل كامل البيانات وتصفية الدورات المختارة فقط
filtered_data = pd.read_csv(input_file)
filtered_data = filtered_data[filtered_data['voting_round'].isin(unique_rounds)]

# حفظ البيانات المفلترة في ملف CSV جديد
filtered_data.to_csv(output_file, index=False)

print("تم نسخ 200 دورة إلى الملف الجديد:", output_file)
