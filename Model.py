import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns 
import matplotlib.pyplot as plt 

file_path = 'C:\\Users\\Taras\\Downloads\\DirectMarketing.csv.csv'
data = pd.read_csv(file_path)

print(data.isnull().sum())

data = pd.get_dummies (data, columns=['Age', 'Gender', 'OwnHome', 'Married', 'Location', 'History'], drop_first=True)

scaler = StandardScaler()
data[['Salary', 'Children', 'Catalogs']] = scaler.fit_transform(data[['Salary', 'Children', 'Catalogs']])

X = data.drop('AmountSpent', axis = 1)
y = data['AmountSpent']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LinearRegression()

lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print('Linear Regression Performance:')
print(f'Mean Absolute Error (MAE): {mae}')
print(f'Mean Squared Error (MSE): {mse}')
print(f'R-squared (R^2): {r2}')

print('\nModel Coefficients:')
for feature, coef in zip(X.columns, lr.coef_):
    print(f'{feature}: {coef}')
    
plt.scatter(y_test, y_pred, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', lw = 2)
plt.title('Actual vs Predicted Amount Spent')
plt.xlabel('Actual Amount Spent')
plt.ylabel('Predicted Amount Spent')
plt.show()

