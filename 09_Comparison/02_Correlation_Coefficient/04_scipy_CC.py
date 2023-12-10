from scipy.stats import pearsonr
x = [1, 2, 3, 4, 5]
y = [5, 4, 3, 2, 1]
coef, p_value = pearsonr(x, y)
print(coef)
