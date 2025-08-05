# titanic_eda.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("titanic.csv")

# Display first 5 rows
print("🔹 First 5 Rows:\n", df.head())

# Basic info
print("\n🔹 Data Info:")
print(df.info())

# Summary statistics
print("\n🔹 Summary Statistics:")
print(df.describe(include='all'))

# Check for missing values
print("\n🔹 Missing Values:\n", df.isnull().sum())

# Countplot: Survival count
sns.countplot(x='Survived', data=df)
plt.title("Survival Count")
plt.show()

# Gender vs Survival
sns.countplot(x='Sex', hue='Survived', data=df)
plt.title("Gender vs Survival")
plt.show()

# Histogram: Age distribution
df['Age'].hist(bins=30)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

# Boxplot: Age by Pclass
sns.boxplot(x='Pclass', y='Age', data=df)
plt.title("Age by Passenger Class")
plt.show()

# Boxplot: Age by Survived
sns.boxplot(x='Survived', y='Age', data=df)
plt.title("Age by Survival")
plt.show()

# Correlation matrix
corr_matrix = df.corr(numeric_only=True)
print("\n🔹 Correlation Matrix:\n", corr_matrix)

# Heatmap
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

# Pairplot of selected features
selected_features = ['Survived', 'Pclass', 'Age', 'Fare', 'SibSp', 'Parch']
sns.pairplot(df[selected_features].dropna())
plt.suptitle("Pairplot of Selected Features", y=1.02)
plt.show()

# Detecting skewness
print("\n🔹 Skewness in numerical features:\n", df.skew(numeric_only=True))

# Value counts of Embarked
print("\n🔹 Value counts of 'Embarked':\n", df['Embarked'].value_counts())
sns.countplot(x='Embarked', hue='Survived', data=df)
plt.title("Embarked vs Survival")
plt.show()
