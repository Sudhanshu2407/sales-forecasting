#Importing the important Libraries.

import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

#------------------------------------------
#We have to import dataset.

sales_df=pd.read_csv(r"C:\sudhanshu_projects\project-task-training-course\advertising.csv")

#Here we load the model using pickle library.

model=pickle.load(open("C:\sudhanshu_projects\project-task-training-course\Sales-forecasting.pkl","rb"))


#-----------------------------------------
#Here we create the title of the streamlit app.
st.title("Sales_forecasting_app")


#----------------------------------------
#opening the image

image = Image.open("C:\sudhanshu_projects\project-task-training-course\sales_forecasting_app_image.jpg")

st.image(image,caption="Sales-forecasting-app",width=500)

#-----------------------------------------
#Now show the dataframe on the streamlit app.

st.header("The given dataset is: ")

st.dataframe(sales_df)


#-----------------------------------------
#Here we create the header of the streamlit app.
st.header("Please enter the values that spend on Tv,Newspaper,Radio")

#------------------------------------------
#Here we create the slider of the streamlit app.
#Tv min=0.7,max=296.4

#Radio min=0.0,max=49.6

#Nespaper min=0.3    max=114.0

tv_value=st.slider(label="enter the amount of TV Advertisment",min_value=0.7,max_value=296.4)
st.write("tv value is",tv_value)

radio_value=st.slider(label="enter the amount of Radio Advertisment",min_value=0.0,max_value=49.6)
st.write("radio value is",radio_value)

newspaper_value=st.slider(label="enter the amount of Newspaper Advertisment",min_value=0.3,max_value=114.0)
st.write("newspaper value is",newspaper_value)

#--------------------------------------------
#Now we predict the salary on the basis of different value.

x_test=[tv_value,radio_value,newspaper_value]

result=model.predict([x_test])

st.header("The result of the model is: ")

st.write("Salary is: ",result[0])

#--------------------------------------------
#Here we create the matplotlib plot in streamlit app.

st.write("The visualization of the dataset are: ")

fig=plt.figure()

#-------------------------------------------
#Here we create scatter plot.
plt.scatter(sales_df["TV"],sales_df["Sales"])
plt.xlabel("Tv Price")
plt.ylabel("Sales Price")
st.pyplot(fig)

#-------------------------------------------
#Here we create heatmap.

st.write("Correlation Between the features are: ")

plot = sns.heatmap(sales_df.corr(), annot=True)
 
# Display the plot in Streamlit
st.pyplot(plot.get_figure())

#--------------------------------------------
#Here we create Pairplot.

st.write("Pairplot of all features are: ")

plot= sns.pairplot(sales_df)

st.pyplot(plot.fig)


