import pandas as pd
import datetime
import xgboost as xgb
import streamlit as st
import numpy as np
import joblib
from PIL import Image

# Load the current date and time
date_time = datetime.datetime.now()

# Load the pre-trained model
model = xgb.XGBRegressor()
model.load_model('xgb_model.json')  # Ensure the path is correct

def main():

    img = Image.open("car_bg.jpg")
    st.image(img, use_column_width=True)
    
    # Transparent overlay for content
    st.markdown("""
    <style>
    .main .block-container {
        background: rgba(255, 255, 255, 0.7);
        padding: 20px;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)
   
    # HTML for the header section
    html_temp = """
    <div style="background-color:lightblue;padding:16px">
    <h2 style="color:black; text-align:center;">Car Price Prediction Using ML</h2>
    </div>
    """
    
    st.markdown(html_temp, unsafe_allow_html=True)
    
    st.write("")
    
    st.markdown("\n##### Are you planning to sell your car!?\n ##### This model will help you calculate the price of your car based on the current market.")

    # Input fields for user data
    i1 = st.number_input("Current ex-showroom price of the car (In lakhs)", min_value=100000, max_value=10000000, step=100000)
        
    i2 = st.number_input("What is the distance covered by the car in kilometres?", min_value=100, max_value=5000000, step=1000)
    
    s1 = st.selectbox('What is the Fuel Type?', ('Petrol', 'Diesel', 'CNG'))
    
    if s1 == 'Petrol':
        s1 = 0
    elif s1 == 'Diesel':
        s1 = 1
    else:
        s1 = 2
        
    s2 = st.selectbox('Are you a Dealer or Individual?', ('Individual', 'Dealer'))
    if s2 == 'Dealer':
        s2 = 0
    else:
        s2 = 1
        
    s3 = st.selectbox('What is the Transmission Type?', ('Manual', 'Automatic'))
    if s3 == 'Manual':
        s3 = 0
    else:
        s3 = 1
    
    i6 = st.slider('Number of Previous Owners', 0, 3)
    
    years = st.number_input('Which Year was the Car Purchased?', 1992, date_time.year, step=1)
    p7 = date_time.year - years
    
    # Prepare the input data for the model
    data_new = pd.DataFrame({
        'Present_Price': i1,
        'Kms_Driven': i2,
        'Fuel_Type': s1,
        'Seller_Type': s2,
        'Transmission': s3,
        'Owner': i6,
        'Age': p7
    }, index=[0])
    
    try:
        if st.button('Predict Price'):
            prediction = model.predict(data_new)
            if prediction > 0:
                st.balloons()
                st.success('You can sell the car for {:.2f} lakhs'.format(prediction[0]))
            else:
                st.warning('You will not be able to sell this car! Sorry.')
    except Exception as e:
        st.warning(f'Something went wrong, please try again. Error: {str(e)}')
        
if __name__ == '__main__':
    main()
