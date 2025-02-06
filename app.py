from pickle import load
import streamlit as st 
import numpy as np
import time 


st.set_page_config(page_title= "Diabete App",page_icon='🚑')
st.markdown("""
            <style>
            .stAppToolbar {
    display:none
    }</style>""",unsafe_allow_html=True )


@st.cache_data
def get_model():
    with open("model.pickle","rb") as file:
        model=load(file)
    return model

model =get_model()
def make_prediction(X):
    X=np.array(X).reshape((1,-1))
    prediction=model.predict(X)
    probability=model.predict_proba(X)*100
   
    if prediction[0] == 0 :
        return "diabete",probability[0][0]
    return "No diabete",probability[0][1]

    
    
    
    
def main():
    
    st.header("Diabete Prediction")
    st.subheader("A machine learning-based web app for early diabetes prediction")
    
    with st.form ("form-data"):
        
        col1,col2 = st.columns(2)
        
        with col1:
            Pregnancies=st.number_input("Pregnancies" ,min_value=0.0,step=1.0)
            
            BloodPressure=st.number_input("Blood Pressure Value",min_value=0.0,step=1.0)
            SkinThickness=st.number_input("Skin Thickness Value",min_value=0.0,step=1.0)
            BMI=st.number_input("BMI",min_value=0.0)
        
        with col2 :
            Glucose=st.number_input("Glucose Value",min_value=0.0,step=1.0,max_value=400.0)
            Insulin=st.number_input("Insulin Level",min_value=0.0,step=1.0)
            DiabetesPedigreeFunction=st.number_input("Diabetes Pedigree Function Level",min_value=0.0)
            Age=st.number_input("Age of Person",min_value=3.0,step=1.0,max_value=100.0)
    
        btn = st.form_submit_button("Predict")
        if btn:
            with st.spinner("Wait..."):
                time.sleep(1)
            input_data=[int(Pregnancies),int(Glucose),int(BloodPressure),int(SkinThickness),int(Insulin),BMI,DiabetesPedigreeFunction,int(Age)]
            result = make_prediction(input_data)
            
            if result[0] =="diabete":
                st.error(f"A Probability of {result[1] } % is diabete")
            else :
                st.success(f"A Probability of {result[1] } % is not diabete")

if __name__ =="__main__":
    main()


