#Import necessary libraries
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import pandas as pd

# Sample transaction data 
data = {'TransactionID': [1, 1, 2, 2, 2, 3, 3, 4, 4, 4],
        'Item': ['A', 'B', 'A', 'B', 'C', 'B', 'C', 'A', 'B', 'C']}
df = pd.DataFrame(data)

# Convert data to one-hot encoded format
basket = pd.crosstab(df['TransactionID'], df['Item'])

# Apriori algorithm
frequent_itemsets = apriori(basket, min_support=0.2, use_colnames=True)

# association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

# the frequent itemsets and association rules
print("Frequent Itemsets:")
print(frequent_itemsets)

print("\nAssociation Rules:")
print(rules)
OUTPUT:
Frequent Itemsets:
   support   itemsets
0     0.75        (A)
1     1.00        (B)
2     0.75        (C)
3     0.75     (B, A)
4     0.50     (C, A)
5     0.75     (B, C)
6     0.50  (B, A, C)

Association Rules:
  antecedents consequents  antecedent support  consequent support  support  \
0         (B)         (A)                1.00                0.75     0.75   
1         (A)         (B)                0.75                1.00     0.75   
2         (B)         (C)                1.00                0.75     0.75   
3         (C)         (B)                0.75                1.00     0.75   
4      (C, A)         (B)                0.50                1.00     0.50   

   confidence  lift  leverage  conviction  zhangs_metric  
0        0.75   1.0       0.0         1.0            0.0  
1        1.00   1.0       0.0         inf            0.0  
2        0.75   1.0       0.0         1.0            0.0  
3        1.00   1.0       0.0         inf            0.0  
4        1.00   1.0       0.0         inf            0.0  
/usr/local/lib/python3.10/dist-packages/ipykernel/ipkernel.py:283: DeprecationWarning: `should_run_async` will not call `transform_cell` automatically in the future. Please pass the result to `transformed_cell` argument and any exception that happen during thetransform in `preprocessing_exc_tuple` in IPython 7.17 and above.
  and should_run_async(code)
/usr/local/lib/python3.10/dist-packages/mlxtend/frequent_patterns/fpcommon.py:110: DeprecationWarning: DataFrames with non-bool types result in worse computationalperformance and their support might be discontinued in the future.Please use a DataFrame with bool type
  warnings.warn(
