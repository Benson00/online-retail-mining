import os
import glob
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define the exact folder names for each configuration
directories = {
    "20 Cores": "results1",
    "10 Cores": "results_10core", # Handles results_10core or results10core
    "4 Cores": "results4core"
}

output_dir = "plots/scalability"
os.makedirs(output_dir, exist_ok=True)

all_records = []

for core_label, folder in directories.items():
    folder_path = folder
    # Fallback check for alternate folder naming
    if not os.path.exists(folder_path):
        if folder == "results_10core" and os.path.exists("results10core"):
            folder_path = "results10core"
        elif folder == "results4core" and os.path.exists("results_4core"):
            folder_path = "results_4core"

    if not os.path.exists(folder_path):
        print(f"Warning: Directory '{folder}' not found. Skipping.")
        continue

    # Dynamically find and read all JSON files inside the folder
    json_files = glob.glob(os.path.join(folder_path, "*.json"))
    if not json_files:
        print(f"Warning: No JSON files found in '{folder_path}'.")
        continue

    for file_path in json_files:
        print(f"Reading file: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            for algo in ['apriori', 'fpgrowth', 'son']:
                if algo in data:
                    if algo == 'apriori': algo_name = 'Apriori'
                    elif algo == 'fpgrowth': algo_name = 'FP-Growth'
                    else: algo_name = 'SON'
                    
                    for support_val, metrics in data[algo].items():
                        all_records.append({
                            'Cores': core_label,
                            'Algorithm': algo_name,
                            'Support': float(support_val),
                            'Total Time (s)': metrics.get('total_time', 0)
                        })

df = pd.DataFrame(all_records)
if df.empty:
    raise ValueError("No data extracted from JSON files! Check your folder paths and contents.")

# Order cores as requested: 20 Cores -> 10 Cores -> 4 Cores
core_order = ["20 Cores", "10 Cores", "4 Cores"]
df['Cores'] = pd.Categorical(df['Cores'], categories=core_order, ordered=True)

# Save consolidated summary CSV
df.to_csv(os.path.join(output_dir, "scalability_summary.csv"), index=False)

# Plotting setup
sns.set_theme(style="whitegrid", font="sans-serif")
colors = {"Apriori": "#34495e", "FP-Growth": "#2ecc71", "SON": "#e74c3c"}

supports = sorted(df['Support'].unique(), reverse=True)
for supp in supports:
    df_supp = df[df['Support'] == supp]
    plt.figure(figsize=(8, 5))
    sns.lineplot(
        data=df_supp,
        x='Cores',
        y='Total Time (s)',
        hue='Algorithm',
        marker='o',
        palette=colors,
        linewidth=2.5,
        markersize=8
    )
    plt.title(f'Scalability Across Cores (Min Support = {supp})', fontsize=14, pad=15)
    plt.xlabel('Hardware Configuration', fontsize=12)
    plt.ylabel('Total Execution Time (seconds)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Algorithm')
    plt.savefig(os.path.join(output_dir, f"scalability_support_{supp}.png"), dpi=300, bbox_inches='tight')
    plt.close()

print(f"Success! All charts and CSV saved in '{output_dir}'.")