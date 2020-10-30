# Simple Linear Regression

# Importing liberaries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



# Importing Dataset
dataset = pd.read_csv("Salary_Data.csv") 



# seperating dependent and independent variable

# Independent variable
x = dataset.iloc[:,:-1]
# Dependent vairiable
y = dataset.iloc[:,-1]




# Splitting the dataset into Training and Testing sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 1/3,random_state = 0)


# NOTE-: In simple Linear Regression we dont need to take care of feature scaling it is automatically taken care by the liberaries in simple linear Regression



# Fitting Simple Linear Regression to the training set
from sklearn.linear_model import LinearRegression
# Instanciating Linear Regression class
regressor = LinearRegression()
# Fitting our Training model in the regressor by passing the indipendent and dependent vairiable
regressor.fit(x_train,y_train)
# In the above code we have made a Linear regression machine called "regeressor" which has learned the trend from the training set and now we can use to predict the values of the test dataset




# Predicting the Test set result
y_pred = regressor.predict(x_test)




# Visualising the Training set  (Graph Plotting)
plt.scatter(x_train,y_train,color = "red")
plt.plot(x_train,regressor.predict(x_train), color = "blue")
plt.title('Salary Vs Experience [Training set]')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()



# Visualising the Testing set  (Graph Plotting)
plt.scatter(x_test,y_test,color = "red")
plt.plot(x_train,regressor.predict(x_train), color = "blue")
plt.title('Salary Vs Experience [Testing set]')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()




