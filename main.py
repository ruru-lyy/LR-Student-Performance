import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.utils import shuffle

data = pd.read_csv('student\\student-mat.csv',sep=";")


data = data[['school','sex','age','internet','famrel','freetime','goout','health','Dalc','Walc','studytime','failures','schoolsup','paid',
             'activities','absences','G1','G2','G3']]

print(data.head())

predict = 'G3' # this is the label of the dataset , the variable that we are goin to predict

columns_to_encode = ['school','sex','internet','schoolsup','paid','activities']

label_encoder = LabelEncoder()

for col in columns_to_encode:
    data[col] = label_encoder.fit_transform(data[col])

print(data.head())

X = data.drop(predict,axis = 1)
y = data[predict]

'''The following code will save your best performing model'''
best = 0 
for _ in range(30):
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= 0.2)
    ## LINEAR REGRESSION ##
'''
    linear = linear_model.LinearRegression()

    linear.fit(X_train,y_train)
    acc = linear.score(X_test,y_test) #accuracy
    print(f"The accuracy of the linear model is {acc}")
    if acc > best:
        best = acc
        ## SAVING MODELS AND VISUALZING DATA ##

        #As we see our model accuracy fluctuates everytime we train it, we want to save the model with the highest accuracy

        # saving the model
        with open("studentmodel.pickle","wb") as f:
            pickle.dump(linear,f) '''

# load the model
with open("studentmodel.pickle","rb") as f:
    saved_model = pickle.load(f)

# print model coefficients and intercept
print(f"Coefficient {saved_model.coef_}")
print(f"Intercept {saved_model.intercept_}")

## PREDICTED VS ACTUAL VALUES
'''This part prints the predicted value of the target label against each row and the actual value '''
predictions = saved_model.predict(X_test)

results_df = pd.DataFrame(X_test, columns=X_test.columns)  
results_df['Prediction'] = predictions
results_df['True Value'] = y_test

# Display DataFrame
print(results_df)


# we are checking how each variable 'p' of a student affects their Final Grade "G3" 

key_variables = ['absences', 'studytime', 'failures', 'G1', 'G2']

for p in key_variables:
    plt.figure(figsize=(8, 6))
    style.use("ggplot")

    # Scatter plot for the key variable vs Final Grade
    plt.scatter(data[p], data[predict], color='blue', label='Data Points')
    plt.xlabel(f"{p}")
    plt.ylabel("Final grade")


## VISUALIZING THE MODEL PERFORMANCE
'''comparing the predicted values (predictions) of the final grades (G3) against their true values (y_test), 
with a red diagonal line representing the ideal case where predictions perfectly match true values, helping visualize the accuracy of the model.

'''

plt.figure(figsize=(10, 6))
plt.scatter(predictions, y_test, color='blue', alpha=0.6, label='Predicted vs True')
plt.plot([min(predictions), max(predictions)], [min(predictions), max(predictions)], color='red', linewidth=2, label='Ideal Line')
plt.xlabel("Predicted Final Grade (G3)")
plt.ylabel("True Final Grade (G3)")
plt.title("Model Predictions vs True Values")
plt.legend()
plt.grid(True)
plt.show()
