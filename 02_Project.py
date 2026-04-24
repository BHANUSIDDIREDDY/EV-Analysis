import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import zscore
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# ================= LOAD DATA =================
df = pd.read_csv("ev.csv")   # make sure ev.csv is in same folder

print(df.head())
print("\nMissing values:\n", df.isna().sum())

# ================= DATA CLEANING =================
df = df.dropna().drop_duplicates()

for col in df.select_dtypes(include=['object', 'string']).columns:
    df[col] = df[col].str.strip()

df.reset_index(drop=True, inplace=True)

print("\nCleaned shape:", df.shape)

# ================= Z-SCORE (OUTLIERS) =================
df['Z_score'] = zscore(df['Electric Range'])

z_outliers = df[abs(df['Z_score']) > 3]

print("\nZ-score Outliers count:", len(z_outliers))

# ================= VISUALIZATION =================

# Line Plot
df.groupby("Model Year")["Electric Range"].mean().plot()
plt.title("Average Electric Range by Model Year")
plt.xlabel("Model Year")
plt.ylabel("Electric Range")
plt.grid()
plt.show()

# Bar Plot
sns.countplot(data=df, x="Electric Vehicle Type")
plt.title("EV Type Count")
plt.xticks(rotation=45)
plt.show()

# Histogram
sns.histplot(df["Electric Range"], bins=20, kde=True)
plt.title("Electric Range Distribution")
plt.show()

# Scatter Plot
sns.scatterplot(data=df, x="Base MSRP", y="Electric Range", hue="Electric Vehicle Type")
plt.title("Range vs Price")
plt.show()

# Heatmap
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Box Plot
sns.boxplot(data=df, x="Electric Vehicle Type", y="Electric Range")
plt.title("Range by EV Type")
plt.xticks(rotation=45)
plt.show()

# Pie Chart
counts = df["Electric Vehicle Type"].value_counts()
plt.pie(counts, labels=counts.index, autopct='%1.1f%%')
plt.title("EV Type Distribution")
plt.show()

# Count Plot
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x="Model Year", order=sorted(df["Model Year"].unique()))
plt.xticks(rotation=45)
plt.title("EV Count by Year")
plt.show()

# ================= REGRESSION =================

df_filtered = df[(df['Electric Range'] > 0) & df['Model Year'].notna()]

X = df_filtered[['Model Year']]
y = df_filtered['Electric Range']

model = LinearRegression()
model.fit(X, y)

x_vals = np.linspace(X.min().values[0], X.max().values[0], 100)
x_vals = pd.DataFrame(x_vals, columns=["Model Year"])

y_vals = model.predict(x_vals)

plt.scatter(X, y, alpha=0.5)
plt.plot(x_vals, y_vals, color='red')
plt.title("Regression: Model Year vs Range")
plt.xlabel("Model Year")
plt.ylabel("Electric Range")
plt.grid()
plt.show()

# ================= IQR OUTLIERS =================

Q1 = df_filtered['Electric Range'].quantile(0.25)
Q3 = df_filtered['Electric Range'].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

iqr_outliers = df_filtered[(df_filtered['Electric Range'] < lower) |
                           (df_filtered['Electric Range'] > upper)]

print("\nIQR Outliers count:", len(iqr_outliers))

sns.boxplot(x=df_filtered['Electric Range'])
plt.title("Outliers (IQR Method)")
plt.show()

# ================= MODEL EVALUATION =================

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("\nR2 Score:", r2_score(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))

plt.scatter(X_test, y_test, alpha=0.5, label="Actual")
plt.plot(X_test, y_pred, color='red', label="Predicted")
plt.legend()
plt.title("Model Evaluation")
plt.xlabel("Model Year")
plt.ylabel("Electric Range")
plt.grid()
plt.show()
