#!/usr/bin/env python
# coding: utf-8

# In[5]:



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pprint
get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


df = pd.read_csv("data.csv")
print("="*50)
print("First Five Rows ","\n")
print(df.head(2),"\n")


# In[7]:


print("="*50)
print("Information About Dataset","\n")
print(df.info(),"\n")


# In[ ]:





# In[8]:


df.head()


# In[ ]:





# In[9]:


df.columns


# In[133]:





# In[10]:


df.head()


# In[12]:


dataset = df


# In[13]:


dataset = df
dataset["Month"] = pd.to_datetime(df["Date"]).dt.month
dataset["Year"] = pd.to_datetime(df["Date"]).dt.year
dataset["Date"] = pd.to_datetime(df["Date"]).dt.date
dataset["Time"] = pd.to_datetime(df["Date"]).dt.time
dataset["Week"] = pd.to_datetime(df["Date"]).dt.week
dataset["Day"] = pd.to_datetime(df["Date"]).dt.day_name()
dataset = df.set_index("Date")
dataset.index = pd.to_datetime(dataset.index)
dataset.head(1)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[14]:


from matplotlib import style

fig = plt.figure()
ax1 = plt.subplot2grid((1,1), (0,0))

style.use('ggplot')

sns.lineplot(x=dataset["Day"], y=dataset["Total_consumption"], data=df)
sns.set(rc={'figure.figsize':(15,6)})

plt.title("Energy consumptionnin Year 2010")
plt.xlabel("Day")
plt.ylabel("Energy in KW")
plt.grid(True)
plt.legend()

for label in ax1.xaxis.get_ticklabels():
    label.set_rotation(90)


plt.title("Energy Consumption According to Month")


# In[18]:


from matplotlib import style


fig = plt.figure()

ax1= fig.add_subplot(311)



style.use('ggplot')

y_2010  = dataset["2010"]["Total_consumption"].to_list()
x_2010 = dataset["2010"]["Day"].to_list()
ax1.plot(x_2010,y_2010, color="red", linewidth=1.7)




plt.rcParams["figure.figsize"] = (18,8)
plt.title("Energy consumption")
plt.xlabel("Date")
plt.ylabel("Energy in KW")
plt.grid(True, alpha=1)
plt.legend()

for label in ax1.xaxis.get_ticklabels():
    label.set_rotation(90)


# In[19]:


sns.distplot(dataset["Total_consumption"])
plt.title("Ennergy Distribution")


# In[20]:


fig = plt.figure()
ax1= fig.add_subplot(111)

sns.lineplot(x=dataset["Month"],y=dataset["Total_consumption"], data=df)
plt.title("Energy Consumption vs Days ")
plt.xlabel("Months")
plt.grid(True, alpha=1)
plt.legend()

for label in ax1.xaxis.get_ticklabels():
    label.set_rotation(90)


# In[91]:


TestData= dataset.tail(100)
Training_set= dataset.iloc[:,0:1]


# In[92]:


Training_set= Training_set[:-60]
print("Training Set Shape ", Training_set.shape)
print("Test Set Shape ", TestData.shape)


# In[93]:


import sklearn
from sklearn import preprocessing


# In[94]:


sc = sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1))
Train = sc.fit_transform(Training_set)


# In[95]:


X_Train = []
Y_Train = []

# Range should be fromm 60 Values to END 
for i in range(60, Train.shape[0]):
    
    # X_Train 0-59 
    X_Train.append(Train[i-60:i])
    
    # Y Would be 60 th Value based on past 60 Values 
    Y_Train.append(Train[i])

# Convert into Numpy Array
X_Train = np.array(X_Train)
Y_Train = np.array(Y_Train)

print(X_Train.shape)
print(Y_Train.shape)


# In[96]:


# Shape should be Number of [Datapoints , Steps , 1 )
# we convert into 3-d Vector or #rd Dimesnsion
X_Train = np.reshape(X_Train, newshape=(X_Train.shape[0], X_Train.shape[1], 1))
X_Train.shape


# In[97]:


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 10, return_sequences = True, input_shape = (X_Train.shape[1], 1)))
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 10, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 10, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 10))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')


# In[98]:


regressor.fit(X_Train, Y_Train, epochs = 10, batch_size = 32)


# In[99]:


TestData.head(2)


# In[100]:


TestData.shape


# In[101]:


Training_set.shape


# In[102]:


Df_Total = pd.concat((Training_set[["Total_consumption"]], TestData[["Total_consumption"]]), axis=0)


# In[103]:


Df_Total.shape


# In[104]:


inputs = Df_Total[len(Df_Total) - len(TestData) - 60:].values
inputs.shape


# In[105]:


inputs = Df_Total[len(Df_Total) - len(TestData) - 60:].values

# We need to Reshape
inputs = inputs.reshape(-1,1)

# Normalize the Dataset
inputs = sc.transform(inputs)

X_test = []
for i in range(60, 160):
    X_test.append(inputs[i-60:i])
    
# Convert into Numpy Array
X_test = np.array(X_test)

# Reshape before Passing to Network
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Pass to Model 
predicted_stock_price = regressor.predict(X_test)

# Do inverse Transformation to get Values 
predicted_stock_price = sc.inverse_transform(predicted_stock_price)


# In[106]:


True_MegaWatt = TestData["Total_consumption"].to_list()
Predicted_MegaWatt  = predicted_stock_price
dates = TestData.index.to_list()


# In[107]:


Machine_Df = pd.DataFrame(data={
    "Date":dates,
    "TrueMegaWatt": True_MegaWatt,
    "PredictedMeagWatt":[x[0] for x in Predicted_MegaWatt ]
})


# In[108]:


Machine_Df.head(30)


# In[126]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




