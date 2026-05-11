import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 1. تحميل الملفات
f1 = 'student_performance_dirty.csv'
f2 = 'synthetic_student_performance_hidden.csv'

def solve_everything(file_path):
    df = pd.read_csv(file_path)
    
    # تنظيف العواميد (مسح أي مسافات في الأسماء)
    df.columns = df.columns.str.strip()
    
    # ملء القيم المفقودة (الرقمية بالمتوسط، والنصية بالمنوال)
    for col in df.columns:
        if df[col].dtype in [np.float64, np.int64]:
            df[col] = df[col].fillna(df[col].mean()).abs() # معالجة السوالب بالمرة
        else:
            df[col] = df[col].fillna(df[col].mode()[0])
            
    # تحويل النصوص لأرقام
    df_final = pd.get_dummies(df)
    
    # البحث عن عمود الـ Exam_Score حتى لو اسمه متغير شوية
    target_col = [c for c in df_final.columns if 'Exam' in c][0]
    
    X = df_final.drop(target_col, axis=1)
    y = df_final[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression().fit(X_train, y_train)
    return mean_squared_error(y_test, model.predict(X_test))

# 2. تنفيذ وطباعة النتائج
print("\n" + "="*20)
try:
    print(f"Clean Data MSE: {solve_everything(f1):.4f}")
    print(f"Synthetic Data MSE: {solve_everything(f2):.4f}")
except Exception as e:
    print(f"Error logic: {e}")
print("="*20)