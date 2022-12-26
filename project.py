#pip install pandas, scikit-learn, sklearn, numpy, flask
import pandas as pd 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#reading the dataset(.csv), df=>dataframe
df = pd.read_csv("areaprice.csv")
print(df)
#print(df.head())#5 rows
# print(df.head(3))
# print(df.tail())
# print(df.tail(3))


#To split the dataset into X and Y axis
#X axis(input/independent variable) and Y axis(output/Dependent variable)
#iloc=>indexlocation 
#iloc[rows,columns]
x = df.iloc[:,:-1]#area
y = df.iloc[:,-1]#price

# print(x)#to print x axis
# print(y)#to print y axis


#Split the dataset into traing and testing
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.1)

#Call the model
model = LinearRegression()


#train the dataset
model.fit(x_train,y_train)


#test/predict the output
pred = model.predict([[4000]])
print("The predicted output of the model is: ",pred)
