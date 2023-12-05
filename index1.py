#Import necessary libraries
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import pandas as pd

# Sample transaction data (replace this with your dataset)
data = {'TransactionID': [1, 1, 2, 2, 2, 3, 3, 4, 4, 4],
        'Item': ['A', 'B', 'A', 'B', 'C', 'B', 'C', 'A', 'B', 'C']}
df = pd.DataFrame(data)

# Convert data to one-hot encoded format
basket = pd.crosstab(df['TransactionID'], df['Item'])

# Apply Apriori algorithm
frequent_itemsets = apriori(basket, min_support=0.2, use_colnames=True)

# Generate association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

# Display the frequent itemsets and association rules
print("Frequent Itemsets:")
print(frequent_itemsets)

print("\nAssociation Rules:")
print(rules)
