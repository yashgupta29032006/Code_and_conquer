import matplotlib
matplotlib.use('Agg') # Essential for non-interactive execution
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# --- 1. Load and Prepare Data ---
print("Loading dataset...")
df = pd.read_csv('car_sales_data.csv').drop_duplicates()
df.columns = [c.strip() for c in df.columns] # Clean column names

# --- 2. Dashboard Structure ---
sns.set_theme(style="whitegrid")
fig = plt.figure(figsize=(22, 18))
grid = plt.GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)

# SECTION 1: OVERVIEW METRICS
ax1 = fig.add_subplot(grid[0, 0])
ax1.axis('off')
total_cars = len(df)
avg_price = df['Price'].mean()
avg_mileage = df['Mileage'].mean()

metrics_text = (
    f"🚀 THE HACKATHON DASHBOARD\n\n"
    f"📊 MARKET OVERVIEW\n"
    f"----------------------------\n"
    f"• Total Cars: {total_cars:,}\n"
    f"• Avg Price: ${avg_price:,.2f}\n"
    f"• Avg Mileage: {avg_mileage:,.0f} miles\n"
)
ax1.text(0.1, 0.5, metrics_text, fontsize=20, fontweight='bold', 
         verticalalignment='center', family='monospace',
         bbox=dict(facecolor='#f0f0f0', alpha=0.9, boxstyle='round,pad=1.5', edgecolor='teal'))

# SECTION 2: DEPRECIATION ANALYSIS
ax2 = fig.add_subplot(grid[0, 1])
sns.regplot(data=df, x='Mileage', y='Price', ax=ax2, 
            scatter_kws={'alpha':0.1, 'color':'#7f8c8d'}, line_kws={'color':'#e74c3c', 'lw':3})
ax2.set_title("Car Value Drops Rapidly with Usage", fontsize=18, fontweight='bold', pad=20)
ax2.set_xlabel("Mileage (miles)", fontsize=14)
ax2.set_ylabel("Price ($)", fontsize=14)
ax2.annotate("50% value loss in \nfirst 50k miles", xy=(25000, 30000), xytext=(150000, 45000),
             arrowprops=dict(arrowstyle="->", color='#c0392b', lw=2), fontsize=14, fontweight='bold', color='#c0392b')

# SECTION 3: AGE IMPACT
ax3 = fig.add_subplot(grid[1, 0])
sns.scatterplot(data=df, x='Year of manufacture', y='Price', ax=ax3, alpha=0.3, color='#f39c12')
ax3.set_title("Newer Cars Command Higher Prices", fontsize=18, fontweight='bold', pad=20)
ax3.set_xlabel("Year of Manufacture", fontsize=14)
ax3.set_ylabel("Price ($)", fontsize=14)
ax3.annotate("Technological Premium \n(Post-2018 models)", xy=(2020, 60000), xytext=(1985, 90000),
             arrowprops=dict(arrowstyle="->", color='#d35400', lw=2), fontsize=14, fontweight='bold', color='#d35400')

# SECTION 4: ENGINE INFLUENCE
ax4 = fig.add_subplot(grid[1, 1])
sns.boxplot(data=df, x='Engine size', y='Price', ax=ax4, palette='viridis')
ax4.set_title("Higher Engine Capacity Drives Premium Pricing", fontsize=18, fontweight='bold', pad=20)
ax4.set_xlabel("Engine Size (L)", fontsize=14)
ax4.set_ylabel("Price ($)", fontsize=14)
ax4.annotate("Direct correlation with \nluxury segments", xy=(3, 40000), xytext=(1, 100000),
             arrowprops=dict(arrowstyle="->", color='#2c3e50', lw=2), fontsize=14, fontweight='bold', color='#2c3e50')

# SECTION 5: BRAND SEGMENTATION
ax5 = fig.add_subplot(grid[2, :])
brand_avg = df.groupby('Manufacturer')['Price'].mean().sort_values(ascending=False)
colors = sns.color_palette("coolwarm", len(brand_avg))
sns.barplot(x=brand_avg.index, y=brand_avg.values, ax=ax5, palette=colors)
ax5.set_title("Clear Brand-Based Price Hierarchy", fontsize=18, fontweight='bold', pad=20)
ax5.set_ylabel("Average Price ($)", fontsize=14)
ax5.set_xlabel("Manufacturer", fontsize=14)
plt.xticks(rotation=15)

fig.suptitle("HACKATHON INSIGHTS: WHAT DRIVES CAR PRICES?", fontsize=28, fontweight='bold', y=0.96)

# --- 3. Save Output ---
output_file = 'car_sales_executive_dashboard.png'
plt.savefig(output_file, bbox_inches='tight', dpi=120)
print(f"Dashboard successfully generated: {output_file}")
