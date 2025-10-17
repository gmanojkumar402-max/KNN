import pandas as pd
import numpy as np
from KNN import KNearestNeighbors
import matplotlib.pyplot as plt
# Set random seed for reproducibility
np.random.seed(42)
# Number of samples
n = 400
# Create columns
user_id = np.arange(1, n+1)
# Randomly assign genders
gender = np.random.choice(['Male', 'Female'], size=n)
# Random ages (between 18 and 60)
age = np.random.randint(18, 60, size=n)
# Estimated salaries (in USD, between 20k and 150k)
estimated_salary = np.random.randint(20000, 150000, size=n)
# Define a simple rule to simulate purchase behavior
# e.g., older users with higher salary are more likely to purchase
purchased = ((age > 30) & (estimated_salary > 60000)).astype(int)
# Combine into a DataFrame
df = pd.DataFrame({
    'User ID': user_id,
    'Gender': gender,
    'Age': age,
    'EstimatedSalary': estimated_salary,
    'Purchased': purchased
})
X = df.iloc[:,2:4].values
y = df.iloc[:,-1].values
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)
from sklearn.preprocessing import StandardScaler
standardscaler =  StandardScaler()
X_train = standardscaler.fit_transform(X_train)
X_test = standardscaler.transform(X_test)
# object creation
knn = KNearestNeighbors(k = 5) # assume k = 3 initially
knn.fit(X_train,y_train)

knn.predict(np.array([60,100000]).reshape(1,2))


def predict_new():
    age = int(input("Enter the age"))
    salary = int(input("Enter the salary"))
    X_new = np.array([[age], [salary]]).reshape(1, 2)

    X_new = standardscaler.transform(X_new)
    result = knn.predict(X_new)
    if result == 0:
        print("will not purchase")
    else:
        print("will purchase")
predict_new()