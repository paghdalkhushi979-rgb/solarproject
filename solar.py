import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import joblib

st.set_page_config(page_title="Solar Power Prediction",
                   page_icon="☀️",
                   layout="wide")

st.title("☀️ Solar Power Prediction System")
st.write("## Machine Learning Project using Polynomial Linear Regression")



st.markdown("----")

st.sidebar.title("Navigation")

st.sidebar.success(" ☀️ Solar Power Prediction")

option = st.sidebar.radio("Select page",
                          (
        "🏠 Home",
        "📂 Dataset",
        "📊 Dataset Information",
        "📈 Data Visualization",
        "🤖 Model Training",
        "🔮 Prediction",
        "📖 Feature Description",
        "ℹ️ About Project",
        "📑 Model Information",))
        


df = pd.read_csv("SolarData.csv")

st.subheader("Missing Values Before Cleaning")

st.write(df.isnull().sum())

df.dropna(inplace=True)

st.success("✅ Missing Values Removed Successfully!")

st.write("Dataset Shape:",df.shape)

if option == "🏠 Home":
    st.header("Welcome 👋")
    st.image("solar.banner.png",use_container_width=True)


    st.write("""
This project predicts **Solar Power Output**
using **Polynomial linear Regression**

## Features
- - 📂 Dataset Preview
- 📊 Dataset Information
- 🤖 Model Training
- ⚡ Solar Power Prediction
- 📈 Performance Evaluation
""")
    
    col1,col2,col3 = st.columns(3)
    with col1:
        st.metric("📋 Total Rows",df.shape[0])
    with col2:
        st.metric("🧩 Total Columns",df.shape[1])
    with col3:
        st.metric("🤖 Algorithm","Polynomial Regression")
    st.subheader("🔄 Project Workflow")

    st.subheader("🚀 Project Workflow")

    st.success("📂 Load Dataset")
    st.write("⬇️")

    st.success("🧹 Remove Missing Values")
    st.write("⬇️")

    st.success("📊 Data Visualization")
    st.write("⬇️")

    st.success("🤖 Train Polynomial Regression Model")
    st.write("⬇️")

    st.success("🔮 Predict Solar Power")
    st.write("⬇️")

    st.success("📥 Download Prediction Report")

    st.success("☀️ Harness the power of the Sun with Machine Learning!")

elif option == "📂 Dataset":
    st.header("📂 Solar Dataset")

    st.dataframe(df)

    st.download_button(" 📥 Download Dataset",
                   df.to_csv(index = False),
                   "SolarData.csv","text/csv")
elif option == "📊 Dataset Information":
    st.header("📊 Dataset Information")

    col1,col2 = st.columns(2)

    with col1:
        st.metric("Rows",df.shape[0])
    with col2:
        st.metric("Columns",df.shape[1])

    st.subheader("Columns")

    st.write(df.columns.tolist())

    st.subheader("Missing Values")

    st.write(df.isnull().sum())

    st.subheader("Data Types")

    st.write(df.dtypes)

    st.subheader("Statistical Summary")

    st.dataframe(df.describe())    

elif option == "📈 Data Visualization":
    st.header("📈 Data Visualization Dashboard")

    graph =  st.selectbox("Select Graph",
                          ("Histogram (Seaborn)",
                           "Scatter Plot (Seaborn)",
                           "Box Plot (Plotly)",
                           "Correlation Heatmap (Seaborn)"))
    if graph == "Histogram (Seaborn)":
        column = st.selectbox("Select Column",df.columns)

        fig, ax = plt.subplots(figsize = (8,5))
        sns.histplot(data = df , x = column , bins = 20 , kde = True , color = "purple" , ax = ax)

        

        ax.set_title(f"{column} Distribution")

        ax.set_xlabel(column)

        ax.set_ylabel("Frequency")

        st.pyplot(fig)

    elif graph == "Scatter Plot (Seaborn)":
        x = st.selectbox("Select X-axis",df.columns,index = 0)

        y = st.selectbox("Select Y-axis",df.columns , index = 5)

        fig, ax = plt.subplots(figsize = (8,5))
        sns.scatterplot(data = df , x = x , y = y , color = "teal",s = 80,ax = ax)

        

        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(f"{x} vs {y}")
        st.pyplot(fig)

    elif graph == "Box Plot (Plotly)":
        column = st.selectbox("Select Column",df.columns)

        fig = px.box(df,y = column , title = f"{column} Box Plot",
                     color_discrete_sequence=["gold"])
        
        st.plotly_chart(fig)

    elif graph == "Correlation Heatmap (Seaborn)":
        fig, ax = plt.subplots(figsize=(8,7))

        sns.heatmap(df.corr(numeric_only=True),annot = True , cmap = "Blues",linewidths=0.5,ax = ax)
        st.pyplot(fig)           

elif option == "🤖 Model Training":
    st.header("🤖 Train PolyNomial Linear Regression Model")
    st.write("Click the button below to train the model.")

    x = df.drop("Power_Output",axis = 1)
    y = df["Power_Output"]

    x_train,x_test,y_train,y_test = train_test_split(
        x,y,test_size=0.2,random_state=42 )
    
    if st.button("🚀 Train Model"):
        poly = PolynomialFeatures(degree=3)
        x_poly_train = poly.fit_transform(x_train)
        x_poly_test = poly.transform(x_test)
        model = LinearRegression()

        
        model.fit(x_poly_train,y_train)
        pre = model.predict(x_poly_test)

        joblib.dump(model, "solar_model.pkl")
        joblib.dump(poly, "poly.pkl")

        st.success("✅ Model Saved Successfully!")

        acc = r2_score(y_test,pre)
        mse = mean_squared_error(y_test,pre)

        inter = model.intercept_
        coa = model.coef_

        st.success("✅ Model Trained Successfully!")
        st.success(f"⭐ Model Accuracy:{round(acc*100,2)}%")

        col1,col2 = st.columns(2)

        with col1:
            st.metric("Accuracy",round(acc,4))
            st.metric("MSE",round(mse,4))

        with col2:
            st.metric("Intercept",round(model.intercept_,4)) 

        feature_names = poly.get_feature_names_out(x.columns)

        coef_df = pd.DataFrame({
                    "Feature": feature_names,
                    "Coefficient": model.coef_})

        st.subheader("📊 Feature Coefficients")
        st.dataframe(coef_df)     

elif option == "🔮 Prediction" :
    st.header("🔮 Solar Power Prediction")  
    st.warning("⚠️ Please enter realstic weather values for better prediction.")  
    st.write("Enter the values below to predict Solar Power Output")

    model = joblib.load("solar_model.pkl")
    poly = joblib.load("poly.pkl")



    irradiance = st.number_input("☀️ Solar Irradiance",min_value=0.0,value=800.0)       

    temperature = st.number_input("🌡️ Temperature",value = 30.0)

    humidity = st.number_input("💧 Humidity",value=60.0)

    wind_speed = st.number_input( "🌬️ Wind Speed",value = 10.0)

    cloud_cover = st.number_input("☁️ Cloud Cover",value = 20.0)

    st.subheader("📋 Input Summary")

    col1,col2 = st.columns(2)
    with col1:
        st.write("☀️ Irradiance :", irradiance)
        st.write("🌡️ Temperature :", temperature)

    with col2:
        st.write("💧 Humidity :", humidity)
        st.write("🌬️ Wind Speed :", wind_speed)
        st.write("☁️ Cloud Cover :", cloud_cover)



    if st.button("⚡ Predict Solar Power"):
        
        input_data = [[irradiance,temperature,humidity,wind_speed,
                    cloud_cover]]
        input_poly = poly.transform(input_data)
        with st.spinner("Predicting....."):

            prediction = model.predict(input_poly)

        st.success("✅ Prediction Completed")

        st.metric("⚡ Predicted Solar Power Output",
            f"{prediction[0]:.2f}")
        
        st.balloons()
        
        if prediction[0] > 500:
            st.success(" 🟢Excellent Solar Power Generation")
        elif prediction[0]>300:
            st.info(" 🟡 Moderate Solar Power Generation")
        else:
            st.error(" 🔴 Low Solar Power Generation")

        st.progress(100)
        st.success("⭐ Prediction Confidence : High")    

        result = pd.DataFrame({
            "Solar Irradiance":[irradiance],
            "Temperature":[temperature],
            "Humidity":[humidity],
            "Wind Speed":[wind_speed],
            "Cloud Cover":[cloud_cover],
            "Predicted Power Output":[prediction[0]]})

elif option == "📖 Feature Description":
    st.header("📖 Feature Description")
    data = {"Feature":["Solar Irradiance","Temperature","Humidity",
    "Wind Speed","Cloud Cover","Power Output"],
    "Description": [
            "Amount of sunlight received.",
            "Atmospheric temperature.",
            "Amount of moisture in air.",
            "Wind speed in m/s.",
            "Percentage of cloud coverage.",
            "Generated solar power."
        ]}
    st.table(pd.DataFrame(data))
        
        
        
elif option == "ℹ️ About Project":
    st.header("ℹ️ About Project")

    st.write(""" ### ☀️ Solar Power Prediction System 
    This project predicts solar power output using 
    Polynomial Linear Regression.
            
    ### Technologies Used
     - 🐍 Python
    - 📊 Streamlit
    - 🤖 Scikit-Learn
    - 📈 Plotly
    - 🐼 Pandas
    - 🎨 Seaborn

    ### Machine Learning Algorithm
    Polynomial Linear Regression(Degree = 3) """)

elif option == "📑 Model Information":
    st.header("📑 Model Information")

    st.info(""" Algorithm Used:
✔ Polynomial Regression
Degree :3
            
Traget Variable :

✔ Power_Output
            
Evalution Metrics :

✔ R² Score

✔ MSE

✔ Intercept

✔ Coefficients""")
    
    st.info("""Dataset Contains Weather Parameters used for predicting solar Power
            Output using Polynomial Regression.""")
        

   

 
