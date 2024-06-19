#Importing the Important Library.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pickle


warnings.filterwarnings("ignore")

#----------------------------------------
#Read the sales csv file.

sales_df=pd.read_csv(r"C:\sudhanshu_projects\project-task-training-course\advertising.csv")

#----------------------------------------
#Check there is any null value.

sales_df.isnull().sum()

#----------------------------------------
#Now check the data type of all columns.

sales_df.dtypes

#----------------------------------------
#Now we decide Independent and dependent feature.

x=sales_df.iloc[:,0:3].values #Independent Variable.

y=sales_df.iloc[:,3].values #Dependent Variable.

#----------------------------------------
#Now we have to split the dataset into train and test.

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

#----------------------------------------
#Now we have to do the scaling.

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

x_train_sc=sc.fit_transform(x_train)

x_test_sc=sc.transform(x_test)

#-----------------------------------------
#Now we select the best linear regression model.

from sklearn.linear_model import LinearRegression

lr=LinearRegression()

lr.fit(x_train,y_train)

y_pred=lr.predict(x_test)

#------------------------------------------
#Here we train and test the model on scaled data.

lr1=LinearRegression()

lr1.fit(x_train_sc,y_train)

y_pred1=lr1.predict(x_test_sc)


#-------------------------------------------
#Now we find the score of different models.

lr.score(x_test,y_test)

lr1.score(x_test_sc,y_test)

#-------------------------------------------
#Here we do the normalization of the data.

from sklearn.preprocessing import Normalizer

nz=Normalizer()

x_train_nz=nz.fit_transform(x_train)

x_test_nz=nz.transform(x_test)

#-------------------------------------------
#Now we again train and test the model on normalized data.

lr2=LinearRegression()

lr2.fit(x_train_nz,y_train)

y_pred2=lr2.predict(x_test_nz)

#-------------------------------------------
#Now we find the score of regression model on normalized data.

lr2.score(x_test_nz,y_test)

#--------------------------------------------
#Here we find the coefficient and intercept of the model.

#For model lr.
m_coef=lr.coef_
c_intercept=lr.intercept_
print("Coefficient and intercept of model lr are: ",m_coef,c_intercept)

#For model lr1.
m_coef1=lr.coef_
c_intercept1=lr.intercept_
print("Coefficient and intercept of model lr1 are: ",m_coef1,c_intercept1)


#For model lr2.
m_coef2=lr.coef_
c_intercept2=lr.intercept_
print("Coefficient and intercept of model lr2 are: ",m_coef2,c_intercept2)

#So the equation of predicting the values are given as:
    
y_pred=m_coef[0]*x[0]+m_coef[1]*x[1]+m_coef[2]*x[2]+c_intercept

#--------------------------------------------
#Now we use the regularization techniques.

from sklearn.linear_model import Lasso,Ridge

#Now we use the lasso regularization.

#As lasso also known as l1 regularization and this is also known as feature selection.

# LassoR=loss+alpha*||w||,where ||w||=w1+w2+......+wn.

lasso_lr=Lasso()

lasso_lr.fit(x_train,y_train)

y_pred_lasso=lasso_lr.predict(x_test)

#--------------------------------------------
#Now we find the score of lasso model.

lasso_lr.score(x_test,y_test)

#--------------------------------------------
#Now we find the coefficient and intercept of the lasso model.

m_coef_lasso=lasso_lr.coef_

c_intercept_lasso=lasso_lr.intercept_

print("The coefficient and intercepts of lasso model are: ",m_coef_lasso,c_intercept_lasso)

#--------------------------------------------
#Now we use the ridge regulae=rzation.

#As ridge also known as l2 regularization and this is also known as feature generalization.

# RidgeR=loss+alpha*||w^2||,where ||w^2||=w1^2+w2^2+......+wn^2.

ridge_lr = Ridge()

ridge_lr.fit(x_train,y_train)

y_pred_ridge=ridge_lr.predict(x_test)

#------------------------------------------
#Now we find the score of ridge model.

ridge_lr.score(x_test,y_test)

#------------------------------------------
#Now we find the coefficient and intercept of the ridge model.

m_coef_ridge=ridge_lr.coef_

c_intercept_ridge=ridge_lr.intercept_

print("The coefficient and intercepts of lasso model are: ",m_coef_ridge,c_intercept_ridge)


#------------------------------------------
#Now we use the OLS(ordinary least square) method of statsmodel.api .

import statsmodels.api as sm

x_opt=x[:,[0,1,2]]

Ols_reg=sm.OLS(endog=y,exog=x_opt).fit()

Ols_reg.summary()

#Conclusion Here we check the P-value in the summary table.
# As for all feature p-value is 0,It means no need to 
#eliminate any feature from the x.

#------------------------------------------
#Now Here we use the statistical method to check the accuracy of the model.

from sklearn.metrics import r2_score,mean_squared_error

#Now we find the r2_score

R2_score=r2_score(y_test, y_pred_lasso)
print("R2_score value is: ",R2_score)
#Conclusion: It shows that that accuracy of the model result is 86.62%.

mse=mean_squared_error(y_test, y_pred_lasso)
print("The value of mean_squared_error is: ",mse)
#Conclusion: It shows that the mean squared error for the model is 4.46.

rmse=np.sqrt(mse)
print("The value of root mean squared error is: ",rmse)
#Conclusion: It shows that the root mean squared error for the model is 2.11.


#-------------------------------------------
#Now do the visualization

plt.plot(sales_df["Sales"]) 
#Conclusion: Here we see the sales are going ups and down continously. 

#Here we find the correlation between the features.

sales_df.corr()

#We can also show the correlation using heatmap.

sns.heatmap(sales_df.corr(),annot=True)
#Conclusion: Here we see that the sales are Highly correlated with Tv.It means
# more the tv advertisement, more the sales.

sns.lineplot(data=sales_df,x="TV",y="Sales")
#Conclusion: Here we see that on increasing the tv advertisement ,the sales of the product increases.

#Here we create regression plot.
sns.regplot(data=sales_df,x="TV",y="Sales")

#Here we create the scatter plot.
plt.scatter(sales_df["Sales"],sales_df["TV"])

#Here we create the box plot.
plt.boxplot(sales_df["Sales"])
#Conclusion: Here we see that there is no outlier in the sales dataset.

#Now we create the bar chart.
plt.bar(sales_df["Sales"],height=10.0)

sns.barplot(data=sales_df,x="Sales")

#Now we create the hist plot.
plt.hist(sales_df["Sales"])

#Now we create the pairplot.
sns.pairplot(data=sales_df)
#Conclusion: Here we see that only Tv vs Sales plot has some meaning.

#-------------------------------------------------
#Now we have to save the model for using them in the front end.
pickle.dump(lasso_lr,open("C:\sudhanshu_projects\project-task-training-course\Sales-forecasting.pkl","wb"))
#Here we save the sales-forecasting.pkl file in the same folder using dump function.

#--------------------------------------------------
#Now we have to load the pkl file.
model=pickle.load(open("C:\sudhanshu_projects\project-task-training-course\Sales-forecasting.pkl","rb"))
#Here we get the model using the load function of pickle.

y_pred_pkl=model.predict(x_test)
#Here we predict the result using this loaded model.









