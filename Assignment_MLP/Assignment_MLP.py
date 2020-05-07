#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ## Perceptron Class

# In[2]:


class Perceptron:
    def __init__(self, features, epochs=1000, learning_rate=0.01):
        self.epochs = epochs
        self.features = features
        self.learning_rate = learning_rate
        self.weights = [0 for i in range(features+1)]
        
        print(f"Epochs\t\t: {epochs}\nFeatures\t: {features}\nLearning Rate\t: {learning_rate}")
        
    def predict(self, inputs):
        summation = self.weights[0]
        summation += sum([inputs[j]*self.weights[j+1] for j in range(self.features)])
        return 1 if summation>0 else 0

    def train(self, tinputs, labels):
        for epoch in range(self.epochs):
            for inputs, label in zip(tinputs, labels):
                prediction = self.predict(inputs)
                self.weights[1:] += self.learning_rate * (label - prediction) * inputs
                self.weights[0] += self.learning_rate * (label - prediction)
        
        print("Trained Weight:\n"+"\n".join(map(str,self.weights)))
        
    def DoPredict(self, testData):
        predicted = list()
        testLabel = list(map(int, testData[:,-1]))

        print("Labels\tPredicted")

        for i in range(len(testLabel)):
            predicted.append(self.predict(test[i,:-1]))
            print(f'{testLabel[i]}\t{predicted[-1]}')

        return testLabel, predicted


# ## Other Functions

# In[3]:


def GenerateConfusionMatrix(testLabel, predicted):
    confusion = {'TP':0, 'TN':0, 'FP':0, 'FN':0}
    for i in range(len(predicted)):
        if testLabel[i] == 1 and predicted[i] == 1:
            confusion['TP']+=1
        elif testLabel[i] == 1 and predicted[i] == 0:
            confusion['FN']+=1
        elif testLabel[i] == 0 and predicted[i] == 1:
            confusion['FP']+=1
        else:
            confusion['TN']+=1
    return confusion


# In[4]:


def ShowConfusionMatrix(testLabel, predicted):
    data = {'Actual Label': testLabel, 'Predicted Label': predicted}
    df = pd.DataFrame(data, columns=data.keys())
    confusion_matrix = pd.crosstab(df['Actual Label'], df['Predicted Label'], rownames=['Actual'], colnames=['Predicted'])
    sns.heatmap(confusion_matrix, annot=True)


# In[5]:


def CalculateMetrics(confusion):
    accuracy = (confusion['TP']+confusion['TN'])/sum(confusion.values())
    error_rate = 1-accuracy
    precision = confusion['TP']/(confusion['TP']+confusion['FP'])
    recall = confusion['TP']/(confusion['TP']+confusion['FN'])
    f1_score = 2*precision*recall/(precision+recall)
    
    print(f"Accuracy \t: {round(accuracy,4)}\nError Rate \t: {round(error_rate,4)}\nPrecision \t: {round(precision,4)}\nRecall  \t: {round(recall,4)}\nF1 Score \t: {round(f1_score,4)}")


# ## Creating Random DataSet

# In[6]:


# Creating RandomDataSet.csv
data = np.zeros((100,5), dtype=float)

np.random.seed(42)
data[:50,4] = 0
data[50:,4] = 1

for i in range(data.shape[1]-1):
    data[:50, i] = np.random.random((1,50))*10
    data[50:, i] = np.random.random((1,50))*50

df = pd.DataFrame(data, columns= ['Feature'+str(i) for i in range(1,5)]+['label'])
df.to_csv("RandomDataSet.csv", index=False)


# ## Single Layer Perceptron on RandomDataSet

# ### Reading RandomDataSet.csv

# In[7]:


# reading the created RandomDataSet.csv
df = pd.read_csv('RandomDataSet.csv')
df.head()


# In[8]:


plt.scatter(df[ df['label']==0.0 ]['Feature1'], df[ df['label']==0.0 ]['Feature4'], marker='o', Label=0)
plt.scatter(df[ df['label']==1.0 ]['Feature1'], df[ df['label']==1.0 ]['Feature4'], marker='x', Label=1)

plt.xlabel('Feature1')
plt.ylabel('Feature4')

plt.show()


# ### Splitting DataSet into Train and Test Data

# In[9]:


data = df.to_numpy()

# splitting dataset into test data and train data
test = np.vstack((data[:20], data[50:70]))
train = np.vstack((data[20:50], data[70:]))

features = train[:,:-1]
labels = train[:,-1]


# ### Perceptron Object, Training, Testing

# In[10]:


p = Perceptron(features.shape[1], 1000, 0.01)


# In[11]:


p.train(features, labels)


# In[12]:


testLabel, predicted = p.DoPredict(test)


# ### Confusion Metrix and Metrics

# In[13]:


confusion = GenerateConfusionMatrix(testLabel, predicted)
print(confusion)


# In[14]:


ShowConfusionMatrix(testLabel, predicted)


# In[15]:


CalculateMetrics(confusion)


# ## Single Layer Perceptron on Sonar DataSet

# ### Reading Sonar DataSet

# In[16]:


# using sonar dataset
df = pd.read_csv('sonar.all-data.csv')

df['R'] = df['R'].map({'R': 1, 'M': 0})
df.head()


# In[17]:


plt.scatter(df[ df['R']==0.0 ]['0.0200'], df[ df['R']==0.0 ]['0.2111'], marker='o', Label=0)
plt.scatter(df[ df['R']==1.0 ]['0.0200'], df[ df['R']==1.0 ]['0.2111'], marker='x', Label=1)

plt.xlabel('Feature1')
plt.ylabel('Feature2')

plt.show()


# ### Splitting DataSet into Train and Test Data

# In[18]:


# converting dataframe into numpy array
data = df.to_numpy()

# splitting dataset into test data and train data
test = np.vstack((data[:20], data[180:]))
train = data[20:180]

features = train[:,:-1]
labels = train[:,-1]


# ### Perceptron Object, Training, Testing

# In[19]:


p = Perceptron(features.shape[1])


# In[20]:


p.train(features, labels)


# In[21]:


testLabel, predicted = p.DoPredict(test)


# ### Confusion Metrix and Metrics

# In[22]:


confusion = GenerateConfusionMatrix(testLabel, predicted)
print(confusion)


# In[23]:


ShowConfusionMatrix(testLabel, predicted)


# In[24]:


CalculateMetrics(confusion)

