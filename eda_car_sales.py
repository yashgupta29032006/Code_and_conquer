import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set aesthetic styling
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14

# 1. Data Loading & Overview
print("--- 1. Data Loading & Overview ---")
try:
    df = pd.read_csv('car_sales_data.csv')
    print(f"Dataset Loaded Successfully. Shape: {df.shape}")
except FileNotFoundError:
    print("Error: car_sales_data.csv not found.")
    exit()

print("\nFirst 5 rows:")
print(df.head())

print("\nData Types & Info:")
print(df.info())

print("\nSummary Statistics:")
print(df.describe())

# Comment on structure:
# The dataset contains columns for Manufacturer, Model, Engine Size, Fuel Type, 
# Year of Manufacture, Mileage, and Price. 
# Target variable is Price. Features consist of both categorical and numerical data.

# 2. Data Cleaning
print("\n--- 2. Data Cleaning ---")
# Check for missing values
missing_values = df.isnull().sum()
print(f"Missing values per column:\n{missing_values}")

# Handling duplicates
duplicates_count = df.duplicated().sum()
print(f"Duplicate rows found: {duplicates_count}")
if duplicates_count > 0:
    df = df.drop_duplicates()
    print("Duplicates removed.")

# Check for invalid values (e.g., negative prices or future years)
invalid_prices = df[df['Price'] < 0].shape[0]
invalid_years = df[df['Year of manufacture'] > 2024].shape[0]
print(f"Rows with negative prices: {invalid_prices}")
print(f"Rows with future years: {invalid_years}")

# 3. Univariate Analysis
print("\n--- 3. Univariate Analysis ---")
numerical_cols = ['Engine size', 'Year of manufacture', 'Mileage', 'Price']

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for i, col in enumerate(numerical_cols):
    sns.histplot(df[col], kde=True, ax=axes[i], color='teal')
    axes[i].set_title(f'Distribution of {col}')
    
    # Skewness calculation
    skewness = df[col].skew()
    print(f"{col} - Skewness: {skewness:.2f}")

plt.tight_layout()
plt.savefig('univariate_analysis.png')
plt.show()

# Observations:
# Price likely shows right skewness due to luxury car models.
# Mileage may have wide distribution with potential outliers at the high end.
# Year of manufacture distribution shows how old/new the fleet is.

# 4. Categorical Analysis
print("\n--- 4. Categorical Analysis ---")
categorical_cols = ['Manufacturer', 'Fuel type']

for col in categorical_cols:
    plt.figure(figsize=(10, 6))
    sns.countplot(y=df[col], order=df[col].value_counts().index, hue=df[col], palette='viridis', legend=False)
    plt.title(f'Count of Cars by {col}')
    plt.savefig(f'count_{col.lower().replace(" ", "_")}.png')
    plt.show()
    print(f"\nValue counts for {col}:\n{df[col].value_counts(normalize=True) * 100}")

# Dominant Manufacturers/Models
top_models = df['Model'].value_counts().head(10)
print(f"\nTop 10 Models:\n{top_models}")

# 5. Bivariate Analysis
print("\n--- 5. Bivariate Analysis ---")

# Mileage vs Price
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Mileage', y='Price', alpha=0.3, color='blue')
plt.title('Mileage vs Price')
plt.savefig('mileage_vs_price.png')
plt.show()

# Year vs Price
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Year of manufacture', y='Price', alpha=0.3, color='orange')
plt.title('Year of Manufacture vs Price')
plt.savefig('year_vs_price.png')
plt.show()

# Engine Size vs Price (Boxplot might be cleaner if engine sizes are discrete enough)
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Engine size', y='Price', palette='Set2')
plt.xticks(rotation=45)
plt.title('Engine Size vs Price')
plt.savefig('enginesize_vs_price.png')
plt.show()

# Manufacturer vs Average Price
plt.figure(figsize=(12, 6))
avg_price_man = df.groupby('Manufacturer')['Price'].mean().sort_values(ascending=False)
sns.barplot(x=avg_price_man.index, y=avg_price_man.values, palette='coolwarm')
plt.title('Average Price by Manufacturer')
plt.ylabel('Average Price')
plt.savefig('avg_price_manufacturer.png')
plt.show()

# 6. Correlation Analysis
print("\n--- 6. Correlation Analysis ---")
# Only numerical for heatmap
corr_matrix = df[numerical_cols].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0)
plt.title('Correlation Heatmap')
plt.savefig('correlation_heatmap.png')
plt.show()

print("\nStrongest correlations with Price:")
print(corr_matrix['Price'].sort_values(ascending=False))

# 7. Outlier Detection
print("\n--- 7. Outlier Detection ---")
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
key_vars = ['Price', 'Mileage', 'Engine size']

for i, var in enumerate(key_vars):
    sns.boxplot(y=df[var], ax=axes[i], color='salmon')
    axes[i].set_title(f'Boxplot of {var}')

plt.tight_layout()
plt.savefig('outlier_detection.png')
plt.show()

# Insight on outliers:
# Extreme Price outliers are likely high-end luxury Porsche or BMW M5 models.
# Very high mileage cars exist but they significantly drop in price.
# Engine sizes are standardized.

print("\n--- EDA Task Completed ---")
