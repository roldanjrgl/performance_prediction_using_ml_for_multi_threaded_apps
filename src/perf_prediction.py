#Multicore project - Performance prediction

from matplotlib import pyplot as plt
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Ridge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import math
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
import scipy.stats

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


# Reading the input data in the form of csv file
def read_csv_data(path):
    print("Reaading the data now...")
    df = pd.read_csv(path)
    column_names =   ['package', 'benchmark', 'threads', 'input_size_name', 'branch_instructions',
                      'branch_instructions_rate', 'branch_misses', 'branch_miss_percentage', 'l3_cache_misses', 'l3_cache_miss_percentage',
                      'l3_cache_references', 'cpu_cycles', 'total_instructions', 'ipc', 'cpu_clock', 
                      'page_faults', 'l1_data_cache_loads', 'l1_instruction_cache_load_misses' , 'llc_load_misses','exe_time', 
                      'speedup']
    df.columns = column_names
    print("Data read completed")
    return df, column_names


data, col_names = read_csv_data("data.csv")

# Dropping the categorical columns that do not contribute significantly to the speed-up pattern.
data = data.drop(['package', 'benchmark', 'input_size_name', 'Exe_time'], axis=1, errors='ignore')
data = data[ data.threads.isin([1, 2, 4, 8, 16, 32, 64, 128])]

# Plotting the histogram for Speed-up to detect anamolies in the measurements
pd.DataFrame.hist(data, column='speedup')

# Thus we can see that most of the speed ups are less than 200, so removing the anomolies in the data for better fit on the training data
data = data[data['speedup'] < 200]

#Defining a function to store plots as images during the prediction
def plotGraph(y_test,y_pred, model, threads):
    if max(y_test) >= max(y_pred):
        my_range = int(max(y_test))
    else:
        my_range = int(max(y_pred))
    plt.clf()
    plt.scatter(range(len(y_test)), y_test, color='blue', label="Actual speed-up")
    plt.scatter(range(len(y_pred)), y_pred, color='red', label="Predicted speed-up")
    plt.legend(loc="best")
    plt.xlabel("Test point")
    plt.ylabel("Speedup")
    plt.title("   Actual & Predicted speed-up \n with " + model + " & thread count: " + str(threads))
    plt.plot()
    plt.savefig("results/plot_images/"+str(model)+str(threads)+".jpeg" , dpi=300)
    return


#Defining different models for the supervised learning regression task
SVR_MODEL = SVR(epsilon = 0.01)
svm_params = [{'kernel': ['linear', 'rbf', 'poly'], 'gamma': [1e-4, 1e-3, 0.01, 0.1, 0.2, 0.5, 0.6, 0.9],'C': [1, 10, 100, 1000, 10000]}]

LINEAR_REG_RIDGE_MODEL = Ridge(alpha=1.0)
linear_reg_params = [{'alpha': [1.0, 5.0, 8.0, 10.0, 15.0]}]

kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
GAUSSIAN_PROCESS_MODEL = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
guassian_model_params = [{'n_restarts_optimizer': [2, 4, 9]}]

RANDOM_FOREST_REGRESSION_MODEL = RandomForestRegressor(max_depth=2, random_state=0)
random_forest_model_params = [{'max_depth': [1, 2, 4, 10, 20], 'n_estimators' : [1, 2, 10, 15,  20]}]

KNN_REGRESSION_MODEL = KNeighborsRegressor(n_neighbors=2)
knn_regression_params = [{'n_neighbors': [2, 3, 5, 10], 'weights' : ['uniform', 'distance']}]


models_map = {'SVR':SVR_MODEL, 'Linear regressor': LINEAR_REG_RIDGE_MODEL, 
              'Guassian Process regressor': GAUSSIAN_PROCESS_MODEL, 'Random forest' : RANDOM_FOREST_REGRESSION_MODEL,
             'KNeighbors regressor' : KNN_REGRESSION_MODEL
             }
params_map = {'SVR':svm_params, 'Linear regressor': linear_reg_params, 
              'Guassian Process regressor': guassian_model_params, 'Random forest' : random_forest_model_params,
               'KNeighbors regressor' : knn_regression_params
             }


#Running each model for all the thread configurations to finally find Mean Absolute Error and Pearson's coefficient.
mean_abs_error_df = [[]]
coeff_of_determination = [[]]
j = -1
for name, model in models_map.items():
    j = j +1
    kk = 0
    mean_abs_error_df.append([])
    coeff_of_determination.append([])
    print("==================================================================")
    print("Now running for ", name)
    for i, g in data.groupby(['threads']):
        if i in [1, 2, 4, 8, 16, 32, 64, 128]:
            print("  Thread count ", i)
            scorer = make_scorer(mean_absolute_error, greater_is_better=False)
            g = g[g['speedup'] < 20]
            X = g.iloc[:, :-1].values.astype(float)
            y = g.iloc[:, -1].values.astype(float)
			#Normalization step
            X = StandardScaler().fit_transform(X)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
            X_train_fs, X_test_fs =  X_train, X_test
			#Finding the best hyperparameters for the model
            regr = GridSearchCV(model, params_map[name], cv = 3, scoring=scorer)
            regr.fit(X_train_fs, y_train)
            y_pred = regr.predict(X_test_fs)
			#calculating the mean absolute error
            error_obtained = mean_absolute_error(y_pred, y_test)
            print("   Mean abs error is ",error_obtained )
            mean_abs_error_df[j].append(error_obtained)
			#computing the Pearson's coefficient
            coeff_of_determination[j].append(scipy.stats.pearsonr(y_test, y_pred)[0])
            kk = kk + 1
            plotGraph(y_test,y_pred, name, i)
            print("------------------")

# Writing output csv files in result directory
error_df = pd.DataFrame(mean_abs_error_df)
print("Mean absolute errors")
print(error_df)
error_df.to_csv("results/errors.csv")

coeff_of_r2_df = pd.DataFrame(coeff_of_determination)
print("Pearson's coefficients")
print(coeff_of_r2_df)
coeff_of_r2_df.to_csv("results/pearsons_coeff.csv")

# Everything successfully finished
print("Sucessfully completed")


