from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

X,y=make_classification(
    n_samples=500,
    n_features=2,
    n_redundant=0,
    n_informative=2,
    random_state=42
)

X_train,X_test,y_train,y_test=train_test_split(
    X,y,test_size=0.2,random_state=42
)

scaler=StandardScaler()

X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

model=LogisticRegression()

model.fit(X_train,y_train)

# from user input=

age=int(input("Enter Customer Age: "))
salary=float(input("Enter customer Salary: "))

user_data=[[age,salary]]

prediction=model.predict(user_data)

probability=model.predict_proba(user_data)

if prediction[0] ==1:
    print("\nCustomer Will Buy Product!")

else:
    print("\nCustomer Will Not Buy Product!")

print("Prediction Probability: \n",probability)