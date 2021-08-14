import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
train_data=pd.read_excel('Data_Train.xlsx')
train_data.head()

#DEAL WITH MISSING VALUES.

train_data.isna().sum()
train_data.shape
train_data.dropna(inplace=True)
train_data.isna().sum()

#DATA CLEANING TO MAKE OUR DATA  READY FOR THE ANALYSIS AS WELL AS MODELLING.

train_data.dtypes

def change_into_datetime(col):
    train_data[col]=pd.to_datetime(train_data[col])
    
train_data.columns
for i in ['Date_of_Journey','Dep_Time', 'Arrival_Time',]:
    change_into_datetime(i)
train_data.dtypes

train_data['Journey_day']=train_data['Date_of_Journey'].dt.day
train_data['Journey_month']=train_data['Date_of_Journey'].dt.month
train_data['Journey_year']=train_data['Date_of_Journey'].dt.year

train_data.head()
train_data.drop('Date_of_Journey',axis=1,inplace=True)

#EXTRACT DERIVED FEATURES FROM DATA.

def extract_hour(df,col):#now we want to extract or gain hour and minute of colum arrival time and departure time separetly so we will apply logic on it and after accessing hour 
    df[col+'_hour']=df[col].dt.hour#and minute we will drop arrival time and dept time df[col+_hour] will give col name as dept_time_hour and arrival_time_hour.
    
def extract_minute(df,col):
    df[col+'_minute']=df[col].dt.minute
    
def drop_column(df,col):
    df.drop(col,axis=1,inplace=True)
    
extract_hour(train_data,'Dep_Time')
extract_minute(train_data,'Dep_Time')
drop_column(train_data,'Dep_Time')

extract_hour(train_data,'Arrival_Time')
extract_minute(train_data,'Arrival_Time')
drop_column(train_data,'Arrival_Time')

train_data.head()

#now there is a duration columns in which duration is giving in the form of hours and minutes. but somewhere it is only giving in minutes and hours so to convert whole duration
#column into hours and minutes we have to apply logic for eg
#if hour and minutes both are there then length =2 and if only hours or minutes will be there for eg 19 hours then it will convert 19 hours 0 minutes and if 30 minutes then 
#0 hours and 30 minutes. and if hours and minutes both are given then length =2 and if hour or minute is given then lenth!=0.

duration=list(train_data['Duration'])#it create a list in which whole data of column duration will store in it.
x='2h 50m'
len(x.split(' '))
for i in range(len(duration)):
    if len(duration[i].split(' '))==2:
        pass
    else:
        if 'h' in duration[i]:
            duration[i]=duration[i] + '0m'
            ##duration[i] + ' ' + '0m'
        
        else:
            duration[i]='0h' + duration[i]
            ##duration[i]='0h' + ' ' + duration[i] 
            
train_data['Duration']=duration
train_data.head()

#PERFORM DATA PREPROCESSING:-
'2h 50m'.split(' ')
'2h 50m'.split(' ')[0]
'2h 50m'.split(' ')[1][0:-1]

def hour(x):
    return x.split(' ')[0][0:-1]

def minute(x):
    return x.split(' ')[1][0:-1]
    
train_data['Duration_hour']=train_data['Duration'].apply(hour)
train_data['Duration_mins']=train_data['Duration'].apply(minute)

train_data.head()
drop_column(train_data,'Duration')
train_data.dtypes

train_data.drop('Duration',axis=1,inplace=True)
train_data.head()
train_data.dtypes
train_data['Duration_hours']=train_data['Duration_hours'].astype(int)
train_data['Duration_mins']=train_data['Duration_mins'].astype(int)

train_data.dtypes
train_data.head()
train_data.dtypes
cat_col=[col for col in train_data.columns if train_data[col].dtype=='O']#categorigal data.
cat_col
cont_col=[col for col in train_data.columns if train_data[col].dtype!='O']#continuous data or columns or numerical column or data.

#HANDLE CATEROGICAL DATA
#perfirm  FEATURE ENCODING data on data.

***categorical data are of two types:-
***nominal data:-dont have any order or  hirarchy like country and perform onehot encoding
***ordinal data:-have hierarcy like good better best. and perform label encoding.
cont_col
categorical=train_data[cat_col]
categorical.head()
categorical['Airline'].value_counts()
plt.figure(figsize=(15,5))
sns.boxplot(y='Price',x='Airline',data=train_data.sort_values('Price',ascending=False))
plt.figure(figsize=(15,5))
sns.boxplot(y='Price',x='Total_Stops',data=train_data.sort_values('Price',ascending=False))
len(categorical['Airline'].unique())
# As Airline is Nominal Categorical data we will perform OneHotEncoding
Airline=pd.get_dummies(categorical['Airline'], drop_first=True)#now we wan to deal with string data beacuse ml dont understand string.After applying this,all our integer featuress will show.
Airline.head()
categorical['Source'].value_counts()
# Source vs Price

plt.figure(figsize=(15,5))
sns.catplot(y='Price',x='Source',data=train_data.sort_values('Price',ascending=False),kind='boxen')
# As Source is Nominal Categorical data we will perform OneHotEncoding


Source=pd.get_dummies(categorical['Source'], drop_first=True)
Source.head()
categorical['Destination'].value_counts()
# As Destination is Nominal Categorical data we will perform OneHotEncoding

#PERFORM LABEL ENCODING ON DATASET.
Destination=pd.get_dummies(categorical['Destination'], drop_first=True)
Destination.head()
categorical['Route']
categorical['Route_1']=categorical['Route'].str.split('→').str[0]#it split my root column of route 0.-->
categorical['Route_2']=categorical['Route'].str.split('→').str[1]
categorical['Route_3']=categorical['Route'].str.split('→').str[2]
categorical['Route_4']=categorical['Route'].str.split('→').str[3]
categorical['Route_5']=categorical['Route'].str.split('→').str[4]

categorical.head()
import warnings 
from warnings import filterwarnings
filterwarnings('ignore')
categorical['Route_1'].fillna('None',inplace=True)
categorical['Route_2'].fillna('None',inplace=True)
categorical['Route_3'].fillna('None',inplace=True)
categorical['Route_4'].fillna('None',inplace=True)
categorical['Route_5'].fillna('None',inplace=True)
categorical.head()
#now extract how many categories in each cat_feature
for feature in categorical.columns:
    print('{} has total {} categories \n'.format(feature,len(categorical[feature].value_counts())))
    ### as we will see we have lots of features in Route , one hot encoding will not be a better option lets appply Label Encoding
    
from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
categorical.columnsfor i in ['Route_1', 'Route_2', 'Route_3', 'Route_4','Route_5']:
    categorical[i]=encoder.fit_transform(categorical[i])
categorical.head()#all text of categorical data will convert into integer.

# Additional_Info contains almost 80% no_info,so we can drop this column
# we can drop Route as well as we have pre-process that column
    
drop_column(categorical,'Route')
drop_column(categorical,'Additional_Info')#it is not providing any information so drop it.
categorical.head()
categorical['Total_Stops'].value_counts()
categorical['Total_Stops'].unique()
# As this is case of Ordinal Categorical type we perform LabelEncoder
# Here Values are assigned with corresponding key

dict={'non-stop':0, '2 stops':2, '1 stop':1, '3 stops':3, '4 stops':4}#now in categorical data there was 1 stop,2 stop,3stop so this was easily replaced by 1,2,3 by using dictionary.
categorical['Total_Stops']=categorical['Total_Stops'].map(dict)
categorical.head()
train_data[cont_col]
# Concatenate dataframe --> categorical + Airline + Source + Destination

data_train=pd.concat([categorical,Airline,Source,Destination,train_data[cont_col]],axis=1)
data_train.head()
drop_column(data_train,'Airline')
drop_column(data_train,'Source')
drop_column(data_train,'Destination')
data_train.head()
pd.set_option('display.max_columns',35)#all columns are not displaying so we increase the limit of pandas as 35 columns.so all columns will display.
data_train.head()

#OUTLIER DETECTION  & oulier imputation if available IN DATA.

data_train.columns
def plot(df,col):
    fig,(ax1,ax2)=plt.subplots(2,1)
    sns.distplot(df[col],ax=ax1)#we plotted displot and box plot so we can find our outlier.
    sns.boxplot(df[col],ax=ax2)
plt.figure(figsize=(30,20))
plot(data_train,'Price')
data_train['Price']=np.where(data_train['Price']>=40000,data_train['Price'].median(),data_train['Price'])#now we found that our outlier is present on>40,000 then we apply median to handle the outlier.
plt.figure(figsize=(30,20))
plot(data_train,'Price')
### separate your independent & dependent data #SEPRATE OUT YOUR DEPENDENT AND INDEPENDENT FEATURE.SELECT BEST FEATURE USING FEATURE SELECTION TECHNIQUES.

X=data_train.drop('Price',axis=1)
X.head()
y=data_train['Price']
y
##type(X)
##type(y)
##X.isnull().sum()
##y.isnull().sum()
#### as now we dont have any missing value in data, we can definitely go ahead with Feature Selection
###np.array(X)
##np.array(y)
from sklearn.feature_selection import mutual_info_classif
mutual_info_classif()
###mutual_info_classif(np.array(X),np.array(y))

#Apply feature selction of data.

X.dtypes
mutual_info_classif(X,y)#X is dependent and y is independent.It return some type of pririty or importance of data.
imp=pd.DataFrame(mutual_info_classif(X,y),index=X.columns)#create dataframe.
imp
imp.columns=['importance']
imp.sort_values(by='importance',ascending=False)

#APPLYING MACHINE LEARNING ON YOUR DATA  AND AUTOMATE YOUR PREDICTION.

***so for applying ml first we split our data into train_data and test_data then we will do prediction of my ml model so we use fit function to find relationship between 
different features.Then we will find the training score of ml_model.after that we will do prediction by using test_data and store it in   array or list.
then aafter that i want to check the performance of my model so for this we use r square matrix,for this there is a class in sklearn like r2_score and we also perform 
mean square error(MAE),underoot of mean square error(RMSE) to check performance of model..

from sklearn.model_selection import train_test_split#split data in training and testing.
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)#test_size=0.2 is nothing but 20% of data is for evaluation.
from sklearn import metrics
##dump your model using pickle so that we will re-use
import pickle
def predict(ml_model,dump):
    model=ml_model.fit(X_train,y_train)#X_train is nothing but data given for evaluation or for training to ml .y_train is nothing but my actual data of training. 
    print('Training score : {}'.format(model.score(X_train,y_train)))
    y_prediction=model.predict(X_test)#X_test is nothing but the data which is given to ml for evaluation or for prediction of data by using ml.it mean in X_test data,data is given for evaluation or to find the prediction or to find coclusion of data.
    print('predictions are: \n {}'.format(y_prediction))#X_test is nothing but the data given for evaluation to find prediction.predict data is nothing but the data which was predict by ml algorithm by applying X_data
    print('\n')
    
    r2_score=metrics.r2_score(y_test,y_prediction)#y_test is nothing but the real test data  or we can say that model should give y_test data after prediction. let us take one example:-we have data like 5,6 then it is X_data which is gien for prediction After that we apply multiplication by using calculator and assume that calculator is ml machine so it will give 30 as answer so it is y_test data but calculator predict or give as 29 then it is prediction data. 
    print('r2 score: {}'.format(r2_score))#so we will find difference between X_test data and pred_data to find difference or error.
    print('MAE:',metrics.mean_absolute_error(y_test,y_prediction))
    print('MSE:',metrics.mean_squared_error(y_test,y_prediction))
    print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test,y_prediction)))
    sns.distplot(y_test-y_prediction)
    
    ****LINEAR REGRESSION model:-in this, how can we know which line is our best fit line so which has less error from datapoints consider as a best fit line less error is
    nothing but the distance of datapoints from line should be minimum. for eg my weight is 88 kg but best fit line tellls that my weigtht=85 then error=3.
    so we will find mean square error (MSE)to find the error ie. 1/total datapoints[summation of i=1 to n (yi(actual value)-yyi(prediction value)]^2.

        If there is 1000-2000 best fit lines then it will difficult to find mse to get our best fit line so we use gradeint descent approach  in this case,in this we take
        different different slope and apply it into y=mx+c and we plot a graph of slop and mse in which whe take one global minima on x-axis(m or slope) where our slop or mse=0.
        and plot a graph then v shape graph will plot ie.gradient descent.So to reduce the mse or from mse=high to 0 we have one theorm ie.convergence theorm.
        
        convergence theorm=mnew=mold-(delta m/derivative of m)8*learning rate(alpha)
        
         
    
    if dump==1:
        ##dump your model using pickle so that we will re-use
        file=open('E:\End-2-end Projects\Flight_Price/model.pkl','wb')
        pickle.dump(model,file)
from sklearn.ensemble import RandomForestRegressor
predict(RandomForestRegressor(),1)

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

predict(DecisionTreeRegressor(),0)
predict(LinearRegression(),0)

#HYPERTUNE OR CROSS VALIDATE YOR MODEL.

from sklearn.model_selection import RandomizedSearchCV

# Number of trees in random forest
n_estimators=[int(x) for x in np.linspace(start=100,stop=1200,num=6)]

# Number of features to consider at every split
max_features=['auto','sqrt']

# Maximum number of levels in tree
max_depth=[int(x) for x in np.linspace(5,30,num=4)]

# Minimum number of samples required to split a node
min_samples_split=[5,10,15,100]

# Create the random grid

random_grid={
    'n_estimators':n_estimators,
    'max_features':max_features,
'max_depth':max_depth,
    'min_samples_split':min_samples_split
}

random_grid

# Random search of parameters, using 3 fold cross validation

rf_random=RandomizedSearchCV(estimator=reg_rf,param_distributions=random_grid,cv=3,verbose=2,n_jobs=-1)
rf_random.fit(X_train,y_train)
rf_random.best_params_
prediction=rf_random.predict(X_test)
sns.distplot(y_test-prediction)
metrics.r2_score(y_test,prediction)
print('MAE',metrics.mean_absolute_error(y_test,prediction))
print('MSE',metrics.mean_squared_error(y_test,prediction))
print('RMSE',np.sqrt(metrics.mean_squared_error(y_test,prediction)))
!pip install pickle
import pickle
# open a file, where you want to store the data
file=open('rf_random.pkl','wb')
# dump information to that file
pickle.dump(rf_random,file)
model=open('rf_random.pkl','rb')
forest=pickle.load(model)
y_prediction=forest.predict(X_test)
y_prediction
metrics.r2_score(y_test,y_prediction)
    