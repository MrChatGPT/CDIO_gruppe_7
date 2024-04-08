import numpy as np
import pandas as pd

# Creating an array of your numbers
numbers = np.array([945.5, 947.5, 810.5, 952.5, 774.5, 834.5, 762.0, 759.5, 840.5, 785.0, 534.0])

# Calculating sum, median, and mean
total_sum = np.sum(numbers)
median = np.median(numbers)
mean = np.mean(numbers)

print(f"sum={total_sum}")
print(f"median={median} mean={mean}")


# Creating a Series from your numbers
numbers_series = pd.Series([945.5, 947.5, 810.5, 952.5, 774.5, 834.5, 762.0, 759.5, 840.5, 785.0, 534.0])

# Using .describe() to get descriptive statistics
desc = numbers_series.describe()

print(f"sum={numbers_series.sum()}")
print(f"median={numbers_series.median()} mean={numbers_series.mean()}")
print(f"desc=\n{desc}")

