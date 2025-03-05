# Polars vs Pandas Performance Comparison

This project demonstrates a performance comparison between Polars and Pandas libraries using the seaborn's tips dataset. Various data operations are performed and timed to showcase the performance differences between these two powerful data manipulation libraries.

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the comparison:
```bash
python comparison.py
```

## Operations Compared

The following operations are compared between Polars and Pandas:
- Data loading
- Basic filtering
- Grouping and aggregation
- Sorting
- Column manipulation
- Joins/Merges

Each operation is timed to measure performance differences between the two libraries. 