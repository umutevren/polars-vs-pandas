import pandas as pd
import polars as pl
import seaborn as sns
import time
import numpy as np
import matplotlib.pyplot as plt
import psutil
import gc
from typing import Callable, Dict, List, Any, Tuple

def time_operation(func: Callable, name: str) -> Tuple[float, float]:
    """
    Measure execution time and memory usage of an operation.
    
    Returns:
        Tuple containing (execution_time, peak_memory_usage)
    """
    # Force garbage collection before the operation
    gc.collect()
    process = psutil.Process()
    
    # Measure memory before operation
    memory_before = process.memory_info().rss / 1024 / 1024  # MB
    
    # Measure execution time
    start_time = time.time()
    result = func()
    end_time = time.time()
    
    # Measure memory after operation
    memory_after = process.memory_info().rss / 1024 / 1024  # MB
    
    execution_time = end_time - start_time
    memory_used = memory_after - memory_before
    
    print(f"{name}: {execution_time:.4f} seconds, Memory: {memory_used:.2f} MB")
    return execution_time, memory_used

def generate_large_dataset(rows: int = 100000) -> Tuple[pd.DataFrame, pl.DataFrame]:
    """Generate a synthetic large dataset for both Pandas and Polars."""
    print(f"Generating synthetic dataset with {rows} rows...")
    
    # Generate data
    np.random.seed(42)
    
    # Use integers instead of dates to avoid overflow
    data = {
        'id': np.arange(rows),
        'value_a': np.random.rand(rows) * 100,
        'value_b': np.random.rand(rows) * 200,
        'category': np.random.choice(['A', 'B', 'C', 'D'], size=rows),
        'year': np.random.randint(2000, 2023, size=rows),
        'month': np.random.randint(1, 13, size=rows),
        'text': np.random.choice(['apple', 'banana', 'cherry', 'date', 'elderberry'], size=rows)
    }
    
    # Create DataFrames
    df_pd = pd.DataFrame(data)
    df_pl = pl.DataFrame(data)
    
    return df_pd, df_pl

# Dictionary to store timing and memory results
results = {
    'Operation': [],
    'Pandas Time': [],
    'Polars Time': [],
    'Pandas Memory': [],
    'Polars Memory': []
}

# Load the tips dataset for basic operations
print("\n=== Loading Tips Dataset ===")
tips_pd = sns.load_dataset('tips')
# Convert to Polars DataFrame
tips_pl = pl.from_pandas(tips_pd)

# 1. Basic Filtering
print("\n=== Filtering Operations (Tips Dataset) ===")
results['Operation'].append('Basic Filtering')
pd_time, pd_mem = time_operation(
    lambda: tips_pd[
        (tips_pd['total_bill'] > 20) & 
        (tips_pd['tip'] > 3)
    ],
    "Pandas filtering"
)
results['Pandas Time'].append(pd_time)
results['Pandas Memory'].append(pd_mem)

pl_time, pl_mem = time_operation(
    lambda: tips_pl.filter(
        (pl.col('total_bill') > 20) & 
        (pl.col('tip') > 3)
    ),
    "Polars filtering"
)
results['Polars Time'].append(pl_time)
results['Polars Memory'].append(pl_mem)

# 2. Grouping and Aggregation
print("\n=== Grouping and Aggregation (Tips Dataset) ===")
results['Operation'].append('GroupBy')
pd_time, pd_mem = time_operation(
    lambda: tips_pd.groupby('day').agg({
        'total_bill': ['mean', 'sum'],
        'tip': ['mean', 'sum']
    }),
    "Pandas groupby"
)
results['Pandas Time'].append(pd_time)
results['Pandas Memory'].append(pd_mem)

pl_time, pl_mem = time_operation(
    lambda: tips_pl.group_by('day').agg([
        pl.col('total_bill').mean().alias('total_bill_mean'),
        pl.col('total_bill').sum().alias('total_bill_sum'),
        pl.col('tip').mean().alias('tip_mean'),
        pl.col('tip').sum().alias('tip_sum')
    ]),
    "Polars groupby"
)
results['Polars Time'].append(pl_time)
results['Polars Memory'].append(pl_mem)

# 3. Sorting
print("\n=== Sorting Operations (Tips Dataset) ===")
results['Operation'].append('Sorting')
pd_time, pd_mem = time_operation(
    lambda: tips_pd.sort_values(['total_bill', 'tip'], ascending=[False, True]),
    "Pandas sorting"
)
results['Pandas Time'].append(pd_time)
results['Pandas Memory'].append(pd_mem)

pl_time, pl_mem = time_operation(
    lambda: tips_pl.sort(['total_bill', 'tip'], descending=[True, False]),
    "Polars sorting"
)
results['Polars Time'].append(pl_time)
results['Polars Memory'].append(pl_mem)

# 4. Column Manipulation
print("\n=== Column Manipulation (Tips Dataset) ===")
results['Operation'].append('Column Calculation')
pd_time, pd_mem = time_operation(
    lambda: tips_pd.assign(
        tip_percentage=(tips_pd['tip'] / tips_pd['total_bill']) * 100
    ),
    "Pandas column calculation"
)
results['Pandas Time'].append(pd_time)
results['Pandas Memory'].append(pd_mem)

pl_time, pl_mem = time_operation(
    lambda: tips_pl.with_columns([
        (pl.col('tip') / pl.col('total_bill') * 100).alias('tip_percentage')
    ]),
    "Polars column calculation"
)
results['Polars Time'].append(pl_time)
results['Polars Memory'].append(pl_mem)

# 5. Create larger dataset for more demanding comparisons
print("\n=== Advanced Comparisons with Larger Dataset ===")
large_pd, large_pl = generate_large_dataset(500000)

# 6. Joins/Merges
print("\n=== Join Operations ===")
# Create smaller dataframes to join with
keys_pd = pd.DataFrame({
    'category': ['A', 'B', 'C', 'D'],
    'category_value': [10, 20, 30, 40]
})
keys_pl = pl.DataFrame({
    'category': ['A', 'B', 'C', 'D'],
    'category_value': [10, 20, 30, 40]
})

results['Operation'].append('Join Operation')
pd_time, pd_mem = time_operation(
    lambda: pd.merge(large_pd, keys_pd, on='category', how='left'),
    "Pandas join"
)
results['Pandas Time'].append(pd_time)
results['Pandas Memory'].append(pd_mem)

pl_time, pl_mem = time_operation(
    lambda: large_pl.join(keys_pl, on='category', how='left'),
    "Polars join"
)
results['Polars Time'].append(pl_time)
results['Polars Memory'].append(pl_mem)

# 7. String Operations
print("\n=== String Operations ===")
results['Operation'].append('String Operations')
pd_time, pd_mem = time_operation(
    lambda: large_pd['text'].str.upper() + "_" + large_pd['category'].str.lower(),
    "Pandas string operations"
)
results['Pandas Time'].append(pd_time)
results['Pandas Memory'].append(pd_mem)

pl_time, pl_mem = time_operation(
    lambda: large_pl.select(
        pl.col('text').str.to_uppercase() + "_" + pl.col('category').str.to_lowercase()
    ),
    "Polars string operations"
)
results['Polars Time'].append(pl_time)
results['Polars Memory'].append(pl_mem)

# 8. Window Functions
print("\n=== Window Functions ===")
results['Operation'].append('Window Functions')
pd_time, pd_mem = time_operation(
    lambda: large_pd.assign(
        rolling_mean=large_pd.groupby('category')['value_a'].transform(
            lambda x: x.rolling(window=10, min_periods=1).mean()
        )
    ),
    "Pandas window functions"
)
results['Pandas Time'].append(pd_time)
results['Pandas Memory'].append(pd_mem)

pl_time, pl_mem = time_operation(
    lambda: large_pl.select([
        pl.col('*'),
        pl.col('value_a').rolling_mean(window_size=10).over('category').alias('rolling_mean')
    ]),
    "Polars window functions"
)
results['Polars Time'].append(pl_time)
results['Polars Memory'].append(pl_mem)

# 9. Lazy vs Eager Evaluation (Polars specific)
print("\n=== Lazy vs Eager Evaluation (Polars) ===")
results['Operation'].append('Lazy Evaluation')
# Eager mode already measured in window functions
results['Pandas Time'].append(pd_time)  # Reuse previous pandas time for comparison
results['Pandas Memory'].append(pd_mem)  # Reuse previous pandas memory

pl_time, pl_mem = time_operation(
    lambda: large_pl.lazy().select([
        pl.col('*'),
        pl.col('value_a').rolling_mean(window_size=10).over('category').alias('rolling_mean')
    ]).collect(),
    "Polars lazy evaluation"
)
results['Polars Time'].append(pl_time)
results['Polars Memory'].append(pl_mem)

# 10. Complex aggregation with multiple operations
print("\n=== Complex Aggregation ===")
results['Operation'].append('Complex Aggregation')
pd_time, pd_mem = time_operation(
    lambda: large_pd.groupby('category').agg({
        'value_a': ['min', 'max', 'mean', 'std'],
        'value_b': ['min', 'max', 'mean', 'std'],
        'text': lambda x: ','.join(x.unique()[:5])
    }),
    "Pandas complex aggregation"
)
results['Pandas Time'].append(pd_time)
results['Pandas Memory'].append(pd_mem)

pl_time, pl_mem = time_operation(
    lambda: large_pl.group_by('category').agg([
        pl.col('value_a').min().alias('value_a_min'),
        pl.col('value_a').max().alias('value_a_max'),
        pl.col('value_a').mean().alias('value_a_mean'),
        pl.col('value_a').std().alias('value_a_std'),
        pl.col('value_b').min().alias('value_b_min'),
        pl.col('value_b').max().alias('value_b_max'),
        pl.col('value_b').mean().alias('value_b_mean'),
        pl.col('value_b').std().alias('value_b_std'),
        pl.col('text').count().alias('text_count')
    ]),
    "Polars complex aggregation"
)
results['Polars Time'].append(pl_time)
results['Polars Memory'].append(pl_mem)

# Create a function to plot the results
def plot_comparison(results: Dict[str, List], metric: str, ylabel: str, filename: str):
    """Create a bar plot comparing the performance metrics."""
    plt.figure(figsize=(15, 8))
    x = range(len(results['Operation']))
    width = 0.35
    
    if metric == 'Time':
        pandas_values = results['Pandas Time']
        polars_values = results['Polars Time']
    else:  # Memory
        pandas_values = results['Pandas Memory']
        polars_values = results['Polars Memory']
    
    plt.bar([i - width/2 for i in x], pandas_values, width, label='Pandas', color='blue', alpha=0.6)
    plt.bar([i + width/2 for i in x], polars_values, width, label='Polars', color='red', alpha=0.6)
    
    plt.xlabel('Operation')
    plt.ylabel(ylabel)
    plt.title(f'{metric} Comparison: Pandas vs Polars')
    plt.xticks(x, results['Operation'])
    plt.legend()
    
    # Add text labels for the height of each bar
    for i, v in enumerate(pandas_values):
        plt.text(i - width/2, v + 0.01, f'{v:.3f}', ha='center', fontsize=8)
    
    for i, v in enumerate(polars_values):
        plt.text(i + width/2, v + 0.01, f'{v:.3f}', ha='center', fontsize=8)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(filename)
    print(f"\n{metric} comparison plot has been saved as '{filename}'")

# Plot time comparison
plot_comparison(results, 'Time', 'Time (seconds)', 'time_comparison.png')

# Plot memory comparison
plot_comparison(results, 'Memory', 'Memory Usage (MB)', 'memory_comparison.png')

# Print summary statistics
print("\n=== Performance Summary ===")
total_pandas_time = sum(results['Pandas Time'])
total_polars_time = sum(results['Polars Time'])
print(f"Total execution time - Pandas: {total_pandas_time:.4f} seconds")
print(f"Total execution time - Polars: {total_polars_time:.4f} seconds")
print(f"Polars is {(total_pandas_time/total_polars_time):.2f}x faster overall")

total_pandas_memory = sum(results['Pandas Memory'])
total_polars_memory = sum(results['Polars Memory'])
print(f"Total memory usage - Pandas: {total_pandas_memory:.2f} MB")
print(f"Total memory usage - Polars: {total_polars_memory:.2f} MB")
print(f"Polars uses {(total_pandas_memory/total_polars_memory):.2f}x less memory overall")

# Create a summary dataframe for easy comparison
print("\n=== Detailed Operation Comparison ===")
comparison_df = pd.DataFrame({
    'Operation': results['Operation'],
    'Pandas Time (s)': results['Pandas Time'],
    'Polars Time (s)': results['Polars Time'],
    'Time Difference': [p/q if q > 0 else float('inf') for p, q in zip(results['Pandas Time'], results['Polars Time'])],
    'Pandas Memory (MB)': results['Pandas Memory'],
    'Polars Memory (MB)': results['Polars Memory'],
    'Memory Difference': [p/q if q > 0 else float('inf') for p, q in zip(results['Pandas Memory'], results['Polars Memory'])]
})
print(comparison_df)

# Save results to CSV
comparison_df.to_csv('comparison_results.csv', index=False)
print("\nDetailed results have been saved to 'comparison_results.csv'") 