import pandas as pd
import numpy as np

data = pd.read_excel(r"C:\Users\heman\OneDrive\Documents\Sem 4\ML\Lab\3\Lab_Session1_Data.xlsx", sheet_name="Purchase_Data")

try:
    data["Date"] = pd.to_datetime(data["Date"], format="%m/%d/%Y")
except ValueError:
    # If formatting cannot be determined, raise an informative error
    raise ValueError("Unable to interpret the 'Date' column format. Please provide the correct format or convert it before reading.")

# Extract relevant columns and create matrices
price_matrix = data[["Price", "Open", "High", "Low"]].to_numpy()
cost_vector = data["Cost"].to_numpy()

if price_matrix.shape[1] != cost_vector.shape[0]:
    raise ValueError("Number of features in price_matrix does not match number of elements in cost_vector.")

# Calculate pseudo-inverse and product
pseudoinverse = np.linalg.pinv(price_matrix)
estimated_costs = np.dot(pseudoinverse, cost_vector)

# Handle potential singularity warnings from pinv
if np.linalg.cond(price_matrix) > 1 / np.finfo(float).eps:
    print("Warning: The price matrix is nearly singular, so the pseudo-inverse is not well-conditioned. The estimated costs may be inaccurate.")

# Display estimated costs
for i, product in enumerate(data["Product"].unique()):
    print(f"Estimated cost for product {product}: {estimated_costs[i]}")
