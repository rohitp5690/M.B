import streamlit as st
import pandas as pd
import pickle
from tensorflow.keras.models import load_model


My_Model=load_model('My_Model.h5')
with open('LE_Gender.pkl','rb') as file:
    LE_Gender=pickle.load(file)
with open('OHE_Geography.pkl','rb') as file:
    OHE_Geography=pickle.load(file)

with open('SScaler.pkl','rb') as file:
    SScaler=pickle.load(file)

st.title('Churn Probability Detector')
CreditScore=st.number_input('Credit Score',step=50)
Geography=st.selectbox("Geography",OHE_Geography.categories_[0])
Gender=st.selectbox('Gender',LE_Gender.classes_)
Age=st.slider('Age',15,90)
Tenure=st.slider('Tenure',1,10)
Balance=st.number_input('Balance')
NumOfProducts=st.slider("No. of Products",1,4)
HasCrCard=st.selectbox('Has Credit Card',[0,1])
is_active_member=st.selectbox("Is active member? ",[0,1])
EstimatedSalary=st.number_input('Estimated Salary')

input_data = {
    'CreditScore': CreditScore,
    'Geography': Geography,
    'Gender': Gender,
    'Age': Age,
    'Tenure': Tenure,
    'Balance': Balance,
    'NumOfProducts': NumOfProducts,
    'HasCrCard': HasCrCard,
    'IsActiveMember': is_active_member,
    'EstimatedSalary': EstimatedSalary
}

Test_Data=pd.DataFrame([input_data])


Test_Data['Gender']=LE_Gender.transform([Test_Data['Gender']])
OHE_Geography_List=OHE_Geography.transform([Test_Data['Geography']]).toarray()
OHE_Geography_List_DF=pd.DataFrame(OHE_Geography_List,columns=OHE_Geography.get_feature_names_out(['Geography']))
Test_Data=pd.concat([Test_Data.drop(['Geography'],axis=1),OHE_Geography_List_DF],axis=1)

Test_Data=SScaler.transform(Test_Data)

Prediction=My_Model.predict(Test_Data)
if Prediction[0][0] >= 0.50:
    st.write(f'The Customer will Churn with Churn Proabibility of {round(Prediction[0][0]*100,2)}%')
else:
    st.write(f'The Customer will not Churn with Churn Proabibility of {round(Prediction[0][0]*100,2)}%')
    
