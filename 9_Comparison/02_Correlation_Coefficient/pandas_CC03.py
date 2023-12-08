import pandas as pd
df = pd.DataFrame({'x': [1, 2, 3, 4, 5], 'y': [5, 4, 3, 2, 1]})
coef = df.corr()
print(coef)

