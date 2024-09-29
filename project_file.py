import pandas as pd

#-------import data & delete unnecessary column - STEP1

DATA=pd.read_csv('Churn_Modelling.csv')
DATA=DATA.drop(['RowNumber','CustomerId','Surname'],axis=1)

#--------now binary conversion of columns gender & geography & preparing the data

from sklearn.preprocessing import OneHotEncoder,LabelEncoder
LE_Gender=LabelEncoder()
LE_Gender_List=LE_Gender.fit_transform(DATA['Gender'])
DATA['Gender']=LE_Gender_List
OHE_Geography=OneHotEncoder()
OHE_Geography_List=OHE_Geography.fit_transform(DATA[['Geography']]).toarray()
OHE_Geography_List_DF=pd.DataFrame(OHE_Geography_List,columns=OHE_Geography.get_feature_names_out(['Geography']))
DATA=pd.concat([DATA.drop(['Geography'],axis=1),OHE_Geography_List_DF],axis=1)

#------------now Independent & Dependent data set created x and y then they are scaled and then stored in decilized format

Y=DATA['Exited'] #Dependent variable
X=DATA.drop(['Exited'],axis=1) #Independent variable

from sklearn.model_selection import train_test_split
X_Train,X_Test,Y_Train,Y_Test=train_test_split(X,Y,test_size=0.3,random_state=42)
from sklearn.preprocessing import StandardScaler
SScaler=StandardScaler()
X_Train=SScaler.fit_transform(X_Train)
X_Test=SScaler.fit_transform(X_Test)

import pickle

with open('LE_Gender.pkl','wb') as file:
    pickle.dump(LE_Gender,file)
with open('OHE_Geography.pkl','wb') as file:
    pickle.dump(OHE_Geography,file)
with open('SScaler.pkl','wb') as file:
    pickle.dump(SScaler,file)
    

#now ANN Model-------------------------------------
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

My_Model=Sequential([
    Dense(64,activation='relu',input_shape=(X_Train.shape[1],)),
    Dense(32,activation='relu'),
    Dense(1,activation='sigmoid')
    
    
])

My_Model.summary()

#----------------------------------create optimizers losses and compile

import tensorflow as tf
from tensorflow.keras import optimizers,losses
Opt=optimizers.Adam(learning_rate=0.02)
Loss=losses.BinaryCrossentropy()
My_Model.compile(optimizer=Opt,loss=Loss,metrics=['accuracy'])


#create log and callbacks
from tensorflow.keras.callbacks import EarlyStopping,TensorBoard
import datetime

LogDir='LOG/FIT' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
TensorBoard_CallBack=TensorBoard(log_dir=LogDir,histogram_freq=1)
EarlyStopping_CallBack=EarlyStopping(monitor='val_loss',patience=10,restore_best_weights=True)

My_Model.fit(
    X_Train,Y_Train,validation_data=(X_Test,Y_Test),epochs=10,
    callbacks=[EarlyStopping_CallBack,TensorBoard_CallBack]
    
    
)

My_Model.save('My_Model.h5')








