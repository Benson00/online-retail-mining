import os
import glob
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define paths and create output directory
input_dir = "results_10core"
output_dir = "plots/results10core"
os.makedirs(output_dir, exist_ok=True)

# Find the JSON file in the results directory
json_files = glob.glob(os.path.join(input_dir, "*.json"))
if not json_files:
    raise FileNotFoundError(f"No JSON files found in the '{input_dir}' directory.")

# Use the first JSON file found
input_file = json_files[0]
print(f"Reading data from: {input_file}")

with open(input_file, 'r', encoding='utf-8') as f:
    raw_data = json.load(f)

# Extract metadata and thresholds dynamically
# Convert thresholds to strings as they act as dictionary keys in JSON
thresholds = [str(t) for t in raw_data['metadata']['thresholds']]
# Sort thresholds descending (e.g., from 0.03 down to 0.02)
thresholds = sorted(thresholds, key=float, reverse=True) 

sns.set_theme(style="whitegrid", font="sans-serif")
colors = ["#34495e", "#2ecc71", "#e74c3c"] # Apriori, FP-Growth, SON



# DATA EXTRACTION TO DATAFRAMES

# DF 1: Total Times
df_time = pd.DataFrame({
    'Support': thresholds * 3,
    'Algorithm': ['Apriori']*3 + ['FP-Growth']*3 + ['SON']*3,
    'Time (s)': [
        raw_data['apriori'][t]['total_time'] for t in thresholds
    ] + [
        raw_data['fpgrowth'][t]['total_time'] for t in thresholds
    ] + [
        raw_data['son'][t]['total_time'] for t in thresholds
    ]
})
df_time.to_csv(os.path.join(output_dir, "total_times.csv"), index=False)

# DF 2: Apriori Breakdown
df_apriori = pd.DataFrame({
    'Support': thresholds,
    'Level 1 (s)': [raw_data['apriori'][t]['levels'].get("1", {}).get('execution_time', 0) for t in thresholds],
    'Level 2 (s)': [raw_data['apriori'][t]['levels'].get("2", {}).get('execution_time', 0) for t in thresholds],
    'Level 3 (s)': [raw_data['apriori'][t]['levels'].get("3", {}).get('execution_time', 0) for t in thresholds]
}).set_index('Support')
df_apriori.to_csv(os.path.join(output_dir, "apriori_breakdown.csv"))

# DF 3: FP-Growth Breakdown
df_fpg = pd.DataFrame({
    'Support': thresholds,
    'Build Time (s)': [raw_data['fpgrowth'][t]['build_time'] for t in thresholds],
    'Mine Time (s)': [raw_data['fpgrowth'][t]['mine_time'] for t in thresholds]
}).set_index('Support')
df_fpg.to_csv(os.path.join(output_dir, "fpgrowth_breakdown.csv"))

# DF 4: SON Breakdown
df_son = pd.DataFrame({
    'Support': thresholds,
    'Phase 1 (s)': [raw_data['son'][t]['phase1_time'] for t in thresholds],
    'Phase 2 (s)': [raw_data['son'][t]['phase2_time'] for t in thresholds]
}).set_index('Support')
df_son.to_csv(os.path.join(output_dir, "son_breakdown.csv"))



# PLOT GENERATION

# Plot 1: Total Time Bar Chart
plt.figure(figsize=(9, 6))
sns.barplot(data=df_time, x='Support', y='Time (s)', hue='Algorithm', palette=colors)
plt.title('Total Execution Time by Algorithm', fontsize=14, pad=15)
plt.xlabel('Minimum Support', fontsize=12)
plt.ylabel('Total Time (seconds)', fontsize=12)
plt.legend(title='Algorithm')
plt.savefig(os.path.join(output_dir, "1_total_time_bar.png"), dpi=300, bbox_inches='tight')
plt.close()

# Plot 2: Scalability Line Chart
plt.figure(figsize=(9, 6))
sns.lineplot(data=df_time, x='Support', y='Time (s)', hue='Algorithm', marker='o', palette=colors, linewidth=2.5, markersize=8)
plt.gca().invert_xaxis() # Invert X-axis to show decreasing support
plt.title('Scalability as Support Decreases', fontsize=14, pad=15)
plt.xlabel('Minimum Support (Decreasing)', fontsize=12)
plt.ylabel('Total Time (seconds)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig(os.path.join(output_dir, "2_scalability_line.png"), dpi=300, bbox_inches='tight')
plt.close()

# Plot 3: Apriori Breakdown Stacked Bar
df_apriori.plot(kind='bar', stacked=True, figsize=(8, 6), color=['#95a5a6', '#e67e22', '#f1c40f'])
plt.title('Apriori: Time Spent per Level', fontsize=14, pad=15)
plt.xlabel('Minimum Support', fontsize=12)
plt.ylabel('Time (seconds)', fontsize=12)
plt.xticks(rotation=0)
plt.legend(title='Level (k)')
plt.savefig(os.path.join(output_dir, "3_apriori_breakdown.png"), dpi=300, bbox_inches='tight')
plt.close()

# Plot 4: FP-Growth Breakdown Stacked Bar
df_fpg.plot(kind='bar', stacked=True, figsize=(8, 6), color=['#2980b9', '#3498db'])
plt.title('FP-Growth: Tree Build vs Mining', fontsize=14, pad=15)
plt.xlabel('Minimum Support', fontsize=12)
plt.ylabel('Time (seconds)', fontsize=12)
plt.xticks(rotation=0)
plt.legend(title='Phase')
plt.savefig(os.path.join(output_dir, "4_fpgrowth_breakdown.png"), dpi=300, bbox_inches='tight')
plt.close()

# Plot 5: SON Breakdown Stacked Bar
df_son.plot(kind='bar', stacked=True, figsize=(8, 6), color=['#8e44ad', '#9b59b6'])
plt.title('SON Algorithm: Phase 1 vs Phase 2', fontsize=14, pad=15)
plt.xlabel('Minimum Support', fontsize=12)
plt.ylabel('Time (seconds)', fontsize=12)
plt.xticks(rotation=0)
plt.legend(title='MapReduce Phase')
plt.savefig(os.path.join(output_dir, "5_son_breakdown.png"), dpi=300, bbox_inches='tight')
plt.close()

print(f"Success! All plots and CSVs are saved in '{output_dir}'.")