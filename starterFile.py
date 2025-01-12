import numpy as np
from matplotlib import pyplot as plt
# %matplotlib inline
import matplotlib
matplotlib.rcParams['figure.figsize'] = (20.0, 10.0)
import pandas as pd


df1 = pd.read_csv('Bengaluru_House_Data.csv')
df1.shape

df1.groupby('area_type')['area_type'].agg('count')  # to aggregate the data based on area_type


# to drop few column that is not affecting the price of the house
df2 = df1.drop(['area_type','society','balcony','availability'],axis='columns')


# DATA CLEANING PROCESS

# show the number of rows having null values
df2.isnull().sum()  

# drop the rows having null values
df3 = df2.dropna()

# In the size column there are some values in BHK and some in Bedroom. So we need to convert all to BHK
# to see how many unique columns are there
df3['size'].unique()

# making a new column BHK 
df3['bhk'] = df3['size'].apply(lambda x: int(x.split(' ')[0]))

# the total sq feet column is also in range so we need to convert it to a single value replacing it with average value
# making a function just to check whether it is a float or not , if not then carry the average of it 
def is_float(x):
    try:
        float(x)
    except:
        return False
    return True

df3[~df3['total_sqft'].apply(is_float)].head(10)  # came to know along with range it has sq.meter , perch , etc

# for range 
def convert_sqft_to_num(x):
    tokens = x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0])+float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None
    
# for rest kind i m ignoring them 
df4 = df3.copy()  # making a deep copy of the dataframe
df4['total_sqft'] = df4['total_sqft'].apply(convert_sqft_to_num)
df4.loc[30]  # to check the 30th row of the dataframe


# FEATURE ENGINEERING AND DIMENSIONALITY REDUCTION

# doing price per sq_feet
df5 = df4.copy()
df5['price_per_sqft'] = df5['price']*100000/df5['total_sqft']


# working on location points 
len(df5.location.unique())  # to check the unique values of location i.e. 1305 means with one hot encoding it will have 1305 columns so dropping few
df5.location  = df5.location.apply(lambda x:x.strip())  # it will strip the name from left and right side
location_stats = df5['location'].value_counts(ascending = False)  # sort the data into descending order 
location_less_than_10 = location_stats[location_stats<=10]  # taking the location which has less than 10 data points
df5.location = df5.location.apply(lambda x: 'other' if x in location_less_than_10 else x)  # replacing the location with other if it has less than 10 data points
len(df5.location.unique())  # 242


# OUTLIER DETECTION
# size with the square_ft -> so less than 300 will be rejected 
df6 = df5[~(df5.total_sqft/df5.bhk<300)]  # removing the data points which has less than 300 sqft per bhk

# to remove the extreme cases of price per sqft like very high and very low price .
# That can be done by grouping the dataframe and the calculate the mean and std for each location and upto 1 std we can take it .
def remove_pps_outlier(df):
    df_out = pd.DataFrame()  # it is made to return a dataframe that satisfy the condition
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft>(m-st)) & (subdf.price_per_sqft<=(m+st))] # keep upto 1 std
        df_out = pd.concat([df_out,reduced_df],ignore_index=True) # add it with df_out dataframe
    return df_out

df7 = remove_pps_outlier(df6)



# Now the issue is the 2BHK has more price than 3BHK in some cases so we need to remove those outliers
# we can do that by plotting the scatter plot of 2BHK and 3BHK and then remove the outliers
def plot_scatter_chart(df,location):
    bhk2 = df[(df.location==location) & (df.bhk==2)]
    bhk3 = df[(df.location==location) & (df.bhk==3)]
    matplotlib.rcParams['figure.figsize'] = (15,10)
    plt.scatter(bhk2.total_sqft,bhk2.price,color='blue',label='2 BHK', s=50)
    plt.scatter(bhk3.total_sqft,bhk3.price,marker='+', color='green',label='3 BHK', s=50)
    plt.xlabel("Total Square Feet Area")
    plt.ylabel("Price (Lakh Indian Rupees)")
    plt.title(location)
    plt.legend()

# now for every location  , we will remove the outliers -> 2BHK mean should be greater than 1BHK mean 
def remove_bhk_outlier(df):
    exclude_indices = np.array([]) # to store the indices of the data points that are to be removed
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        # the below for loop will give the mean and std of the price per sqft for each bhk
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }

        # it will add if the mean value is less than estimate mean
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)

    return df.drop(exclude_indices,axis='index')


df8 = remove_bhk_outlier(df7)



# to remove the bathrooms like 2 bedrooms and 4 bathrooms 
df9 = df8[ df8.bath < df8.bhk+2 ]

# now droping the extra columns that are made for price estimation or for cleanung purpose
df9 = df9.drop(['price_per_sqft'] , axis= 'columns')















# For categorial data , like location doing one hot encoding
dummies = pd.get_dummies(df9.location)

df10 = pd.concat([df9,dummies],axis='columns')
# to avoid dummy variable trap
df11 = df10.drop(['other'],axis='columns')
# since dummies are made so we can drop the location column
df11 = df10.drop(['location'],axis='columns')



# Model making
X = df11.drop(['price', 'size'], axis='columns')
y = df11.price

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=10)

from sklearn.linear_model import LinearRegression
lr_clf = LinearRegression()
lr_clf.fit(X_train,y_train)
lr_clf.score(X_test,y_test)

# we will use k fold cross validation to measure the accuracy of our LinearRegression model
from sklearn.model_selection import ShuffleSplit      # it will basically divide the data randomly into parts that made for cross_validation
from sklearn.model_selection import cross_val_score   

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

cross_val_score(LinearRegression(), X, y, cv=cv)



# now we will try other models like Lasso and DecisionTreeRegressor by using GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import ShuffleSplit, GridSearchCV
import pandas as pd

def find_best_model_using_gridsearchcv(X, y):
    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Define hyperparameter search space
    algos = {
        'linear_regression': {
            'model': LinearRegression(),
            'params': {
                # Removed 'normalize', as it's not valid anymore
                'fit_intercept': [True, False],
                'positive': [True, False]
            }
        },
        'lasso': {
            'model': Lasso(),
            'params': {
                'alpha': [1, 2],
                'selection': ['random', 'cyclic']
            }
        },
        'decision_tree': {
            'model': DecisionTreeRegressor(),
            'params': {
                'criterion': ['squared_error', 'friedman_mse'],
                'splitter': ['best', 'random']
            }
        }
    }

    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

    for algo_name, config in algos.items():
        gs = GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
        gs.fit(X_scaled, y)
        scores.append({
            'model': algo_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })

    return pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])


# Example usage
best_model_results = find_best_model_using_gridsearchcv(X, y)
print(best_model_results)


# Assume that the linear Regression is giving u the best result
def predict_price(location,sqft,bath,bhk):    
    loc_index = np.where(X.columns==location)[0][0]

    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return lr_clf.predict([x])[0]


predict_price('1st Phase JP Nagar',1000, 2, 2)  
predict_price('1st Phase JP Nagar',1000, 3, 3)
predict_price('Indira Nagar',1000, 2, 2)




# Export the tested model to a pickle file and it will be used by python flask server
import pickle
with open('banglore_home_prices_model.pickle','wb') as f:
    pickle.dump(lr_clf,f)  # .dump( model , file)


# for storing the columns that are used in the model
import json
columns = {
    'data_columns' : [col.lower() for col in X.columns]  # to make all the columns to be in lower order
}
with open("columns.json","w") as f:
    f.write(json.dumps(columns))  # .dumps( data , file)










# Now main task is to write a python flask server to write the http request made from the UI and then predict the price of the house
#  



