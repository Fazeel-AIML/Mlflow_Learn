from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")

wine = load_wine()
X = wine.data
y = wine.target

x_train, x_test , y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

max_depth=10
n_estimator = 10
with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=n_estimator,max_depth=max_depth)
    model.fit(x_train,y_train)
    
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_pred=y_pred,y_true=y_test)
    
    mlflow.log_metric("accuracy",accuracy)
    mlflow.log_param("max_depth",max_depth)
    mlflow.log_param("n_estimator",n_estimator)
    
    cm = confusion_matrix(y_pred=y_pred,y_true=y_test)
    plt.figure(figsize=(10,6))
    sns.heatmap(cm,annot=True, cmap="Blues",xticklabels=wine.target_names,yticklabels=wine.target_names)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    
    plt.savefig("Confusion-Matrix.png")
    
    mlflow.log_artifact("Confusion-Matrix.png")
    mlflow.log_artifact(__file__)
    print(accuracy)
    
