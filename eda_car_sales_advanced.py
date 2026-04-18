import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set aesthetic styling for competition-level quality
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# 1. Data Loading
print("--- 1. Data Loading ---")
df = pd.read_csv('car_sales_data.csv')
df = df.drop_duplicates() # Clean duplicates early
print(f"Dataset Shape: {df.shape}")

# 2. Key Relationship Plots (CRITICAL)
print("\n--- 2. Key Relationship Plots ---")

# Mileage vs Price with Trendline
plt.figure(figsize=(10, 6))
sns.regplot(data=df, x='Mileage', y='Price', scatter_kws={'alpha':0.2, 'color':'teal'}, line_kws={'color':'red'})
plt.title('Mileage vs Price: The Depreciation Curve')
plt.xlabel('Mileage')
plt.ylabel('Price')
plt.savefig('mileage_vs_price_trend.png')

# Year vs Price
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Year of manufacture', y='Price', alpha=0.3, color='orange')
plt.title('Year of Manufacture vs Price: age-based Valuation')
plt.xlabel('Year of Manufacture')
plt.ylabel('Price')
plt.savefig('year_vs_price_scatter.png')

# Engine Size vs Price
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Engine size', y='Price', palette='viridis')
plt.title('Engine Size vs Price: Power & Luxury Correlation')
plt.xlabel('Engine Size (L)')
plt.ylabel('Price')
plt.savefig('engine_size_vs_price_box.png')

# Manufacturer vs Average Price (Sorted Bar Chart)
plt.figure(figsize=(12, 6))
avg_price_man = df.groupby('Manufacturer')['Price'].mean().sort_values(ascending=False)
sns.barplot(x=avg_price_man.index, y=avg_price_man.values, palette='RdYlGn_r')
plt.title('Manufacturer vs Average Price: Brand Premium Hierarchy')
plt.xlabel('Manufacturer')
plt.ylabel('Average Price')
plt.xticks(rotation=45)
plt.savefig('manufacturer_avg_price.png')

# 3. Grouped Analysis
print("\n--- 3. Grouped Analysis ---")

# Average price by Manufacturer
print("\nTop 5 Most Expensive Manufacturers:")
print(avg_price_man.head(5))
print("\nCheapest Manufacturers:")
print(avg_price_man.tail(5))

# Average price by Year (Last 10 years for focus)
recent_years = df[df['Year of manufacture'] >= 2010].groupby('Year of manufacture')['Price'].mean()
print("\nAverage Price Trend (Last 10 Years):")
print(recent_years)

# Engine Size Bins
df['Engine_Bin'] = pd.cut(df['Engine size'], bins=[0, 1.2, 1.6, 2.0, 3.0, 5.0], labels=['Eco (<1.2L)', 'Standard (1.2-1.6L)', 'Mid Range (1.6-2.0L)', 'Heavy (2.0-3.0L)', 'Performance (>3.0L)'])
price_by_bin = df.groupby('Engine_Bin', observed=False)['Price'].mean()
print("\nAverage Price by Engine Category:")
print(price_by_bin)

# 4. Correlation with Price
print("\n--- 4. Correlation Analysis ---")
numerical_cols = ['Engine size', 'Year of manufacture', 'Mileage', 'Price']
correlations = df[numerical_cols].corr()['Price'].sort_values(ascending=False)
print("Correlation values with Price:")
print(correlations)

# 5. Pattern Detection & Advanced Observations (Calculated)
print("\n--- 5. Advanced Observations ---")

# Threshold detection: Mileage where price drops sharply
# We can look at 50k intervals
df['Mileage_Bracket'] = (df['Mileage'] // 50000) * 50000
mileage_depreciation = df.groupby('Mileage_Bracket')['Price'].mean()
print("\nPrice Decay at 50k Mileage Intervals:")
print(mileage_depreciation)

# Cluster Detection: Economy vs Luxury
# Define Luxury as Top 25% of price
luxury_threshold = df['Price'].quantile(0.75)
print(f"Luxury Segment Threshold (Top 25%): {luxury_threshold}")

# 6. Final Summary Generation Logic
print("\nEDA Completed. Insights being formulated.")
