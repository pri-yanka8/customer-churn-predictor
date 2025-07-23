from django.shortcuts import render
from django.template import RequestContext
from django.contrib import messages
from django.http import HttpResponse
from django.conf import settings
import os
from django.core.files.storage import FileSystemStorage
import os
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv("Dataset/Customer_Data.csv")
dataset.drop(['Customer_ID'], axis = 1,inplace=True)
labels = np.unique(dataset['Customer_Status'])
#Pre-process the data by converting categorical variables to numerical variables, and replacing missing values with mean.
label_encoder = []
columns = dataset.columns
types = dataset.dtypes.values
for i in range(len(types)):
    name = types[i]
    if name == 'object': #finding column with object type
        le = LabelEncoder()
        dataset[columns[i]] = pd.Series(le.fit_transform(dataset[columns[i]].astype(str)))#encode all str columns to numeric
        label_encoder.append([columns[i], le])
dataset.fillna(dataset.mean(), inplace = True)
#dataset normalizing using standard Scaler
Y = dataset['Customer_Status'].ravel()
dataset.drop(['Customer_Status'], axis = 1,inplace=True)
X = dataset.values
sc = StandardScaler()
X = sc.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
rf_cls = RandomForestClassifier()
rf_cls.fit(X_train, y_train)

def Predict(request):
    if request.method == 'GET':
        return render(request, 'Predict.html', {})

def PredictAction(request):
    if request.method == 'POST':
        global dataset, rf_cls, label_encoder, sc, labels
        myfile = request.FILES['t1'].read()
        fname = request.FILES['t1'].name
        if os.path.exists("ChurnApp/static/"+fname):
            os.remove("ChurnApp/static/"+fname)
        with open("ChurnApp/static/"+fname, "wb") as file:
            file.write(myfile)
        file.close()

        testdata = pd.read_csv("ChurnApp/static/"+fname)
        temp = testdata.values
        for i in range(len(label_encoder)):
            le = label_encoder[i]
            if le[0] != 'Customer_Status':
                testdata[le[0]] = pd.Series(le[1].transform(testdata[le[0]].astype(str)))#encode all str columns to numeric                
        testdata.fillna(dataset.mean(), inplace = True)#replace missing values        
        testdata = testdata.values
        testdata = sc.transform(testdata)
        predict = rf_cls.predict(testdata)
        predict = predict.ravel()
        print(predict)
        output = '<table border=1 align=center width=100%><tr><th><font size="3" color="black">Test Data</th>'
        output += '<th><font size="3" color="black">Customer Churn Prediction</th></tr>'
        for i in range(len(predict)):
            output += '<tr><td><font size="3" color="black">'+str(temp[i])+'</td>'
            if predict[i] == 0:
                output += '<td><font size="3" color="red"><b>'+str(labels[predict[i]])+'</b></font></td></tr>'
            elif predict[i] == 1:
                output += '<td><font size="3" color="cyan"><b>'+str(labels[predict[i]])+'</b></font></td></tr>'
            elif predict[i] == 2:
                output += '<td><font size="3" color="green"><b>'+str(labels[predict[i]])+'</b></font></td></tr>'    
        output+= "</table></br></br></br></br>"    
        context= {'data':output}
        return render(request, 'Graph.html', context)   

def Visualization(request):
    if request.method == 'GET':
        return render(request, 'Visualization.html', {})

def VisualizationAction(request):
    if request.method == 'POST':
        column = request.POST.get('t1', False)
        data = pd.read_csv("Dataset/Customer_Data.csv")
        gender_churn = data.groupby([column,'Customer_Status']).size().reset_index()
        gender_churn = gender_churn.rename(columns={0: 'Count'})
        if column == "Gender":
            plt.figure(figsize=(4, 3))
        if column == "Churn_Category":
            plt.figure(figsize=(8, 3))
            plt.xticks(rotation=70)
        if column == "State" or column == "Age":
            plt.figure(figsize=(16, 3))
            plt.xticks(rotation=70)
        sns.barplot(x=column,y='Count',hue='Customer_Status',data=gender_churn)
        plt.title(column+" Based Churned Graph")
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        plt.clf()
        plt.cla()
        context= {'data':column+" Based Churned Graph", 'img': img_b64}
        return render(request, 'Graph.html', context)    

def index(request):
    if request.method == 'GET':
       return render(request, 'index.html', {})

