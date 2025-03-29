#!/usr/bin/env python
# coding: utf-8

# In[4]:


get_ipython().system('pip install streamlit')
get_ipython().system('pip install pyngrok')


# In[5]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import streamlit as st
import matplotlib.pyplot as plt

# عنوان التطبيق
st.title("Traffic Volume Prediction")

# تحميل البيانات من ملف CSV
uploaded_file = st.file_uploader("traffic_data.csv", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    # عرض أسماء الأعمدة للتحقق
    st.write("Columns in the dataset:", data.columns)

    # التأكد من وجود الأعمدة المطلوبة
    datetime_column = 'date_time'  # عمود التاريخ
    traffic_volume_column = 'traffic_volume'  # عمود حجم الحركة

    if datetime_column not in data.columns or traffic_volume_column not in data.columns:
        st.error(f"يجب أن يحتوي الملف على عمود '{datetime_column}' وعمود '{traffic_volume_column}'.")
    else:
        # تحويل عمود التاريخ إلى نوع datetime
        data[datetime_column] = pd.to_datetime(data[datetime_column])

        # استخراج ميزات من التاريخ
        data['hour'] = data[datetime_column].dt.hour
        data['day_of_week'] = data[datetime_column].dt.dayofweek
        data['month'] = data[datetime_column].dt.month

        # حذف العمود الأصلي للتاريخ
        data.drop(datetime_column, axis=1, inplace=True)

        # تحويل الأعمدة الفئوية إلى متغيرات عددية
        data = pd.get_dummies(data, drop_first=True)

        # تقسيم البيانات إلى ميزات وأهداف
        X = data.drop(traffic_volume_column, axis=1)  # الميزات
        y = data[traffic_volume_column]  # الهدف

        # تقسيم البيانات إلى مجموعة تدريب واختبار
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # تدريب النموذج
        model = RandomForestRegressor()
        model.fit(X_train, y_train)

        # تقييم النموذج
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # عرض النتائج كتابيًا
        st.write(f'Mean Absolute Error: {mae:.2f}')
        st.write(f'R² Score: {r2:.2f}')

        # رسم النتائج
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred)
        plt.xlabel('Actual Traffic Volume')
        plt.ylabel('Predicted Traffic Volume')
        plt.title('Actual vs Predicted Traffic Volume')
        plt.plot([y.min(), y.max()], [y.min(), y.max()], '--', color='red')  # خط المثالي
        st.pyplot(plt)


# In[ ]:




