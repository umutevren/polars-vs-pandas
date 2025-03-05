import pandas as pd
import polars as pl
import seaborn as sns
import time
from typing import Callable
import matplotlib.pyplot as plt

def time_operation(func: Callable, name: str) -> float:
    """Measure execution time of an operation."""
    start_time = time.time()
    func()
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"{name}: {execution_time:.4f} seconds")
    return execution_time

# Load the tips dataset
tips_pd = sns.load_dataset('tips')
# Convert to Polars DataFrame
tips_pl = pl.from_pandas(tips_pd)

# Dictionary to store timing results
results = {
    'Operation': [],
    'Pandas': [],
    'Polars': []
}

# 1. Basic Filtering
print("\n=== Filtering Operations ===")
results['Operation'].append('Filtering')
results['Pandas'].append(
    time_operation(
        lambda: tips_pd[
            (tips_pd['total_bill'] > 20) & 
            (tips_pd['tip'] > 3)
        ],
        "Pandas filtering"
    )
)
results['Polars'].append(
    time_operation(
        lambda: tips_pl.filter(
            (pl.col('total_bill') > 20) & 
            (pl.col('tip') > 3)
        ),
        "Polars filtering"
    )
)

# 2. Grouping and Aggregation
print("\n=== Grouping and Aggregation ===")
results['Operation'].append('GroupBy')
results['Pandas'].append(
    time_operation(
        lambda: tips_pd.groupby('day').agg({
            'total_bill': ['mean', 'sum'],
            'tip': ['mean', 'sum']
        }),
        "Pandas groupby"
    )
)
results['Polars'].append(
    time_operation(
        lambda: tips_pl.groupby('day').agg([
            pl.col('total_bill').mean().alias('total_bill_mean'),
            pl.col('total_bill').sum().alias('total_bill_sum'),
            pl.col('tip').mean().alias('tip_mean'),
            pl.col('tip').sum().alias('tip_sum')
        ]),
        "Polars groupby"
    )
)

# 3. Sorting
print("\n=== Sorting Operations ===")
results['Operation'].append('Sorting')
results['Pandas'].append(
    time_operation(
        lambda: tips_pd.sort_values(['total_bill', 'tip'], ascending=[False, True]),
        "Pandas sorting"
    )
)
results['Polars'].append(
    time_operation(
        lambda: tips_pl.sort(['total_bill', 'tip'], descending=[True, False]),
        "Polars sorting"
    )
)

# 4. Column Manipulation
print("\n=== Column Manipulation ===")
results['Operation'].append('Column Calculation')
results['Pandas'].append(
    time_operation(
        lambda: tips_pd.assign(
            tip_percentage=(tips_pd['tip'] / tips_pd['total_bill']) * 100
        ),
        "Pandas column calculation"
    )
)
results['Polars'].append(
    time_operation(
        lambda: tips_pl.with_columns([
            (pl.col('tip') / pl.col('total_bill') * 100).alias('tip_percentage')
        ]),
        "Polars column calculation"
    )
)

# Create a bar plot comparing the performance
plt.figure(figsize=(12, 6))
x = range(len(results['Operation']))
width = 0.35

plt.bar([i - width/2 for i in x], results['Pandas'], width, label='Pandas', color='blue', alpha=0.6)
plt.bar([i + width/2 for i in x], results['Polars'], width, label='Polars', color='red', alpha=0.6)

plt.xlabel('Operation')
plt.ylabel('Time (seconds)')
plt.title('Performance Comparison: Pandas vs Polars')
plt.xticks(x, results['Operation'])
plt.legend()

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the plot
plt.savefig('performance_comparison.png')
print("\nPerformance comparison plot has been saved as 'performance_comparison.png'")

# Print summary statistics
print("\n=== Performance Summary ===")
total_pandas = sum(results['Pandas'])
total_polars = sum(results['Polars'])
print(f"Total execution time - Pandas: {total_pandas:.4f} seconds")
print(f"Total execution time - Polars: {total_polars:.4f} seconds")
print(f"Polars is {(total_pandas/total_polars):.2f}x faster overall") 