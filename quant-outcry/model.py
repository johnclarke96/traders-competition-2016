import pandas as pd
import numpy as np
import math
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.externals import joblib
from copy import deepcopy
#------------------DATA PARTITIONING------------------#
# 50,000 rows of data, partition into 7/10 training, 1.5/10 validation, 1.5/10 test
fname = 'outcry_data.csv'
indicators = ['GDP','CPI','TAMIT','RS','HS','PPI','MTIS','U','MS','PI']
data = pd.read_csv(fname, names=indicators, dtype=float, skiprows=[0])
data = np.array(data.values.tolist())
length = len(data)
partitions = (8./10, 1./10, 1./10)
part_vals = (int(round(partitions[0]*length)), int(round((partitions[0]+partitions[1])*length)))
data_dict = {'crossval':[], 'test':[]}
label_dict = {'crossval': [], 'test':[]}
lookback = 1
N = 3
#------------------DATA PARTITIONING------------------#
[1600.0, -0.1, 1000.0, 450.0, 1.09, -0.4, 1279.0, 6.0, 997.0, 275.0, 1672.89, -0.09, 446.66, 1.09, -0.4, 1289.09, 6.2, 964.31, 263.5]

#------------------FEATURE ENGINEERING------------------#
# pass in a set of features, returns 
# def polynomial(features, n):
# 	poly = PolynomialFeatures(n)
# 	poly = poly.fit_transform([features])
# 	return poly

# we create the feature vectors
# try log approx later, issue with negative values (+ |max_neg_value| approach)
data_dict['crossval'] = []
label_dict['crossval'] = []
for i in range(lookback,part_vals[1]):
	x = np.array([])

	val = data[i][2]
	d1 = deepcopy(data[i-1])
	d0 = np.delete(deepcopy(data[i]), [2])
	x = np.append(x, d0)
	x = np.append(x, d1)
	label_dict['crossval'].append(val)
	data_dict['crossval'].append(x)

# apply polynomial features here

data_dict['test'] = []
label_dict['test'] = []
for i in range(part_vals[1],len(data)):
	x = np.array([])

	val = data[i][2]
	d1 = deepcopy(data[i-1])
	d0 = np.delete(deepcopy(data[i]), [2])
	x = np.append(x, d0)
	x = np.append(x, d1)
	label_dict['crossval'].append(val)
	data_dict['crossval'].append(x)

	label_dict['test'].append(val)
	data_dict['test'].append(x)



#------------------FEATURE ENGINEERING------------------#

#------------------TESTING------------------#
def print_test_results(model):
	for i in range(len(data_dict['test'])):
   		print str(i) + ' actual: ', label_dict['test'][i]
   		print 'prediction: ', model.predict([data_dict['test'][i]])[0]

def avg_test_sse(model):
	predictions = model.predict(data_dict['test'])
	sse = abs(sum((label_dict['test']-predictions)**1) / len(predictions))
	print sse

def get_params(model):
	print 'coeff: ', model.coef_
	print 'alpha: ', model.alpha_
#------------------TESTING------------------#


#------------------TRAINING------------------#
# model 1: polynomial basis, lasso, log-diff of variables
def lasso_model():
    lasso = LassoCV()
    lasso.fit(data_dict['crossval'], label_dict['crossval'])
    print lasso.coef_
    joblib.dump(lasso, 'poly2_lasso.pkl') 
    # get_params(lasso)
    print_test_results(lasso)
    avg_test_sse(lasso)

def lasso_model_comp():
	lasso = joblib.load('poly2_lasso.pkl') 
	return lasso

def ridge_model_comp():
	ridge = joblib.load('poly2_ridge.pkl')
	return ridge

def ridge_model():
	ridge = RidgeCV()
	ridge.fit(data_dict['crossval'], label_dict['crossval'])
	joblib.dump(ridge, 'poly2_ridge.pkl')
	# get_params(ridge)
	print_test_results(ridge)
	avg_test_sse(ridge)
#------------------TRAINING------------------#
def main():
	print 'lasso: '
	#lasso_model()
	#print 'ridge: '
	ridge_model()

if __name__ == "__main__":
	main()





