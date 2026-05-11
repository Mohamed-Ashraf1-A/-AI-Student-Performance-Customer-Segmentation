import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1. تحميل وتجهيز الداتا
df = pd.read_csv('customer_segmentation_dirty.csv')
X = df[['Purchase_Frequency', 'Annual_Spending', 'Website_Visits', 'Customer_Score']]
X = X.fillna(X.mean()).abs()

# 2. عمل الـ Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. تطبيق الـ K-Means بـ 3 مجموعات (بناءً على الرسمة اللي طلعتلك)
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# 4. إضافة المجموعات للملف الأصلي
df['Cluster'] = clusters

# 5. طباعة متوسطات كل مجموعة عشان نفهمهم
print("\n" + "="*10 + " CUSTOMER SEGMENTS " + "="*10)
print(df.groupby('Cluster')[['Purchase_Frequency', 'Annual_Spending', 'Customer_Score']].mean())
print("="*39)

# 6. رسم المجموعات (Annual Spending vs Customer Score)
plt.scatter(X_scaled[clusters == 0, 1], X_scaled[clusters == 0, 3], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X_scaled[clusters == 1, 1], X_scaled[clusters == 1, 3], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X_scaled[clusters == 2, 1], X_scaled[clusters == 2, 3], s = 100, c = 'green', label = 'Cluster 3')
plt.title('Clusters of Customers')
plt.xlabel('Annual Spending (Scaled)')
plt.ylabel('Customer Score (Scaled)')
plt.legend()
plt.show()