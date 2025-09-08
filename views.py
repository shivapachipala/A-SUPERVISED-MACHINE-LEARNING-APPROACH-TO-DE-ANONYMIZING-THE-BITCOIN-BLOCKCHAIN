from django.shortcuts import render
from django.template import RequestContext
from django.contrib import messages
from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage
import os
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import io
import base64

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

global uname
global X_train, X_test, y_train, y_test
accuracy, precision, recall, fscore = [], [], [], []
global dataset, ranges, classes, X, detected

def calculateMetrics(algorithm, predict, y_test):
    global accuracy, precision, recall, fscore
    a = accuracy_score(y_test,predict)*100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)

dataset = pd.read_csv("Dataset/Bitcoin.csv")
dataset.fillna(0, inplace = True)
labels = np.unique(dataset['class'])

le = LabelEncoder()
dataset['class'] = pd.Series(le.fit_transform(dataset['class'].astype(str)))#encode all str columns to numeric
dataset.fillna(0, inplace = True)            
dataset = dataset.values

X = dataset[:,0:dataset.shape[1]-1]
Y = dataset[:,dataset.shape[1]-1]
print(X)
print(Y)

indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
Y = Y[indices]

sc = StandardScaler()
X = sc.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2) #split dataset into train and test

dt_cls = DecisionTreeClassifier()
dt_cls.fit(X_train, y_train)
predict = dt_cls.predict(X_test)
calculateMetrics("Decision Tree", predict, y_test)

bc_cls = LogisticRegression()
bc_cls.fit(X_train, y_train)
predict = bc_cls.predict(X_test)
calculateMetrics("Logistic Regression", predict, y_test)

ada_cls = AdaBoostClassifier()
ada_cls.fit(X_train, y_train)
predict = ada_cls.predict(X_test)
calculateMetrics("AdaBoost Classifier", predict, y_test)

gb_cls = GradientBoostingClassifier(n_estimators=50)
gb_cls.fit(X_train, y_train)
predict = gb_cls.predict(X_test)
calculateMetrics("Gradient Boosting Classifier", predict, y_test)

knn_cls = KNeighborsClassifier(n_neighbors=2)
knn_cls.fit(X_train, y_train)
predict = knn_cls.predict(X_test)
calculateMetrics("KNN", predict, y_test)

rf_cls = RandomForestClassifier()
rf_cls.fit(X_train, y_train)
predict = rf_cls.predict(X_test)
calculateMetrics("Random Forest", predict, y_test)

def PredictAction(request):
    if request.method == 'POST':
        global dataset, labels, sc, rf_cls
        myfile = request.FILES['t1'].read()
        fname = request.FILES['t1'].name
        if os.path.exists("IllegalApp/static/"+fname):
            os.remove("IllegalApp/static/"+fname)
        with open("IllegalApp/static/"+fname, "wb") as file:
            file.write(myfile)
        file.close()
        dataset = pd.read_csv("IllegalApp/static/"+fname)
        dataset.fillna(0, inplace = True)
        dataset = dataset.values
        temp = dataset
        dataset = sc.fit_transform(dataset)
        pred = rf_cls.predict(dataset)
        output = ''
        output+='<table border=1 align=center width=100%><tr><th><font size="" color="black">Test Transaction Values</th><th><font size="" color="black">De-Anonymize</th>'
        output+='</tr>'
        for i in range(len(pred)):
            value = int(pred[i])
            output+='<td><font size="" color="black">'+str(temp[i])+'</td><td><font size="" color="black">'+labels[value]+'</td></tr>'
        context= {'data':output}
        return render(request, 'ViewResult.html', context)
        

def Predict(request):
    if request.method == 'GET':
       return render(request, 'Predict.html', {})

def TrainML(request):
    if request.method == 'GET':
        global accuracy, precision, recall, fscore
        output = ''
        output+='<table border=1 align=center width=100%><tr><th><font size="" color="black">Algorithm Name</th><th><font size="" color="black">Accuracy</th><th><font size="" color="black">Precision</th>'
        output+='<th><font size="" color="black">Recall</th><th><font size="" color="black">FSCORE</th></tr>'
        global accuracy, precision, recall, fscore
        algorithms = ['Decision Tree', 'LogisticRegression', 'AdaBoost', 'Gradient Boosting', 'KNN', 'CNN']
        for i in range(len(algorithms)):
            output+='<td><font size="" color="black">'+algorithms[i]+'</td><td><font size="" color="black">'+str(accuracy[i])+'</td><td><font size="" color="black">'+str(precision[i])+'</td><td><font size="" color="black">'+str(recall[i])+'</td><td><font size="" color="black">'+str(fscore[i])+'</td></tr>'
        output+= "</table></br>"
        df = pd.DataFrame([['Decision Tree','Precision',precision[0]],['Decision Tree','Recall',recall[0]],['Decision Tree','F1 Score',fscore[0]],['Decision Tree','Accuracy',accuracy[0]],
                           ['LogisticRegression','Precision',precision[1]],['LogisticRegression','Recall',recall[1]],['LogisticRegression','F1 Score',fscore[1]],['LogisticRegression','Accuracy',accuracy[1]],
                           ['AdaBoost','Precision',precision[2]],['AdaBoost','Recall',recall[2]],['AdaBoost','F1 Score',fscore[2]],['AdaBoost','Accuracy',accuracy[2]],
                           ['Gradient Boosting','Precision',precision[3]],['Gradient Boosting','Recall',recall[3]],['Gradient Boosting','F1 Score',fscore[3]],['Gradient Boosting','Accuracy',accuracy[3]],
                           ['KNN','Precision',precision[4]],['KNN','Recall',recall[4]],['KNN','F1 Score',fscore[4]],['KNN','Accuracy',accuracy[4]],
                           ['CNN','Precision',precision[5]],['CNN','Recall',recall[5]],['CNN','F1 Score',fscore[5]],['CNN','Accuracy',accuracy[5]],
                          ],columns=['Algorithms','Metrics','Value'])
        df.pivot_table(index="Algorithms", columns="Metrics", values="Value").plot(kind='bar', figsize=(8, 4))
        plt.title("All Algorithms Performance Graph")
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        img_b64 = base64.b64encode(buf.getvalue()).decode()    
        context= {'data':output, 'img': img_b64}
        return render(request, 'ViewResult.html', context)

def AdminLogin(request):
    if request.method == 'GET':
       return render(request, 'AdminLogin.html', {})

def index(request):
    if request.method == 'GET':
       return render(request, 'index.html', {})

def Aboutus(request):
    if request.method == 'GET':
       return render(request, 'Aboutus.html', {})

def LoadDataset(request):
    if request.method == 'GET':
       return render(request, 'LoadDataset.html', {})    

def AdminLoginAction(request):
    if request.method == 'POST':
        global uname
        username = request.POST.get('username', False)
        password = request.POST.get('password', False)
        if username == "admin" and password == "admin":
            context= {'data':'welcome '+username}
            return render(request, 'AdminScreen.html', context)
        else:
            context= {'data':'Invalid login details'}
            return render(request, 'AdminLogin.html', context)          

def LoadDatasetAction(request):
    if request.method == 'POST':
        global dataset
        myfile = request.FILES['t1'].read()
        fname = request.FILES['t1'].name
        if os.path.exists("IllegalApp/static/"+fname):
            os.remove("IllegalApp/static/"+fname)
        with open("IllegalApp/static/"+fname, "wb") as file:
            file.write(myfile)
        file.close()
        dataset = pd.read_csv("IllegalApp/static/"+fname,nrows=1000)
        dataset.fillna(0, inplace = True)
        columns = dataset.columns
        datasets = dataset.values
        output='<table border=1 align=center width=100%><tr>'
        for i in range(0, 10):
            output += '<th><font size="" color="black">'+columns[i]+'</th>'
        output += '</tr>'
        for i in range(len(datasets)):
            output += '<tr>'
            for j in range(0, 10):
                output += '<td><font size="" color="black">'+str(datasets[i,j])+'</td>'
            output += '</tr>'
        output+= "</table></br></br></br></br>"
        #print(output)
        context= {'data':output}
        return render(request, 'ViewResult.html', context)    







        
