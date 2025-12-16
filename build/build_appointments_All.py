import pickle
import xgboost as xgb
import gc
import pandas as pd
import numpy as np
from TextMiningMachine.io import write_data
from TextMiningMachine.xgboost_optimizer import XgboostOptimizer
from sklearn.model_selection import train_test_split

gc.collect()


# set target col
target_col = 'NonCompletedAppointments'
# load raw data
data = pd.read_pickle('data/raw_data_all.p')
data[target_col] = [1 if data.loc[i,'CompletedAppointments']==0 else 0 for i in range(data.shape[0])]
#data = data[np.isfinite(data[target_col])]
#data = data.reset_index()

# # transform data into features
# features = trans.transform(data)
# ## save the transform
#file = 'data/features.p'
#pickle.dump(features, open(file, 'wb'))

# load transform
with open('models/text_cat_transformer_all.p', 'rb') as f:
    trans = pickle.load(f)
# load transformed data
with open('data/features.p', 'rb') as f:
    features = pickle.load(f)


import random
from scipy import sparse
random.seed(2018)


target_vals = data[target_col]
del data
gc.collect()
# split the data
X_train, X_test, y_train, y_test = train_test_split(features, target_vals, test_size=.5, random_state=0)
# split the data
X_test, X_holdout, y_test, y_holdout = train_test_split(X_test, y_test, test_size=.5, random_state=0)


# format the training and test sets
train = xgb.DMatrix(X_train, label=y_train, feature_names=trans.feature_names_clean)
test = xgb.DMatrix(X_test, label=y_test, feature_names=trans.feature_names_clean)
holdout = xgb.DMatrix(X_holdout, label=y_holdout, feature_names=trans.feature_names_clean)
#w Build XGboost models
x = XgboostOptimizer()

## Perform a iterative random grid search, where each iteration samples
## values for the parameters between the LHS and RHS specified

# update features related to random search algorithm
x.update_params({'num_boost_rounds': 400,
                 'max_over_fit':.035,
                 'max_minutes_total':900,
                 'num_rand_samples': 30,
                 'early_stopping_rounds': 3,
                 'objective': 'binary:logistic',
                 'eval_metric': ['logloss','auc']})
x.overfit_metric='auc'
#call random search algorithm
x.fit_random_search(dtrain = train, evals=[(train, 'Train'), (test, "Test")])
best_rand_score = x.best_score
#Extract the best parameters from the random search and set best params
# as starting values for the sequential search method
best_params = x.xgb_params_best
x.set_initial_xgb_params(best_params)
model = x.best_model

x.update_params({'num_boost_rounds': 500,
                 'build_past': 2,
                 'max_over_fit': .03,
                 'eta': np.arange(0.01, .4, .01),
                 'min_child_weight': range(10, 200, 10),
                 'max_minutes': 450,
                 'lambda': np.arange(1, 0, -.01),
                 'alpha': np.arange(0, 1, .01)
                 })

x.fit_seq_search(train, evals=[(train, 'Train'), (test, "Test")])
print('hold : ', x.base_score)
print('best : ', x.best_score)
print('num models built : ', x.n_models_tried)
# extract the best model
model = x.best_model



importances = model.get_score(importance_type='gain')
importance_frame = pd.DataFrame({'Importance': list(importances.values()), 'Feature': list(importances.keys())})
importance_frame.sort_values(by = 'Importance', inplace=True)
importance_frame = importance_frame.tail(100)
importance_frame.plot(x ='Feature', y='Importance' ,kind='barh',legend=False )


recurr_feat_names = importance_frame['Feature'].values
recurr_feat_inds = [trans.feature_names_clean.index(feat) for feat in importance_frame['Feature']]

X_test_recurr = sparse.csc_matrix(X_test)
X_train_recurr = sparse.csc_matrix(X_train)

X_test_recurr = X_test_recurr[:,recurr_feat_inds]
X_train_recurr = X_train_recurr[:,recurr_feat_inds]


# format the training and test sets
train_recurr = xgb.DMatrix(X_train_recurr, label=y_train, feature_names=[trans.feature_names_clean[i] for i in recurr_feat_inds])
test_recurr = xgb.DMatrix(X_test_recurr, label=y_test, feature_names=[trans.feature_names_clean[i] for i in recurr_feat_inds])


x= XgboostOptimizer()
# update features related to random search algorithm
# update features related to random search algorithm
x.update_params({'num_boost_rounds': 600,
                 'max_over_fit':.03,
                 'max_minutes_total':1200,
                 'num_rand_samples': 30,
                 'early_stopping_rounds': 2,
                 'objective': 'binary:logistic',
                 'eval_metric': ['logloss','auc']})
x.overfit_metric='auc'

x.set_random_param_range({'max_depth':(1,12),
                          'min_child_weight':(5,1000),
                          'eta':(.03,.15)})

#call random search algorithm
x.fit_random_search(dtrain = train_recurr, evals=[(train_recurr, 'Train'), (test_recurr, "Test")])

best_rand_score = x.best_score
#Extract the best parameters from the random search and set best params
# as starting values for the sequential search method
best_params = x.xgb_params_best
x.set_initial_xgb_params(best_params)
model = x.best_model

# save the optimal params
file = 'models/optimal_appointments_params_all_rand.p'
pickle.dump(x.xgb_params_best, open(file, 'wb'))

# save the best model
file = 'models/appointments_model_all_rand.p'
pickle.dump(model, open(file, 'wb'))

x.update_params({'num_boost_rounds': 600,
                 'build_past': 2,
                 'max_over_fit': .03,
                 'eta': np.arange(0.01, .4, .01),
                 'min_child_weight': range(10, 200, 10),
                 'max_minutes': 400,
                 'lambda': np.arange(1, 0, -.01),
                 'alpha': np.arange(0, 1, .01)
                 })

x.fit_seq_search(train_recurr, evals=[(train_recurr, 'Train'), (test_recurr, "Test")])
print('base : ', x.base_score)
print('best : ', x.best_score)
print('num models built : ', x.n_models_tried)

model = x.best_model

# save the optimal params
file = 'models/optimal_appointments_params_all.p'
pickle.dump(x.xgb_params_best, open(file, 'wb'))

# save the best model
file = 'models/NoShow_AppointmentLevel_All.p'
pickle.dump(model, open(file, 'wb'))

# write to table in statmods
####################################################

features = sparse.csc_matrix(features)
features = features[:,recurr_feat_inds]
features_xgb = xgb.DMatrix(features,label = data[target_col],feature_names=[trans.feature_names_clean[i] for i in recurr_feat_inds])
pred_vals = model.predict(features_xgb)
shap_vals = model.predict(features_xgb,pred_contribs=True)

dsn = 'statmods'
key = 'PatEncCSNID'
write_table = 'Appointment_Predictions'
date_col = 'AppointmentDateTime'


#output = np.hstack((np.transpose(np.stack(np.array(data[['HospitalAccountID','PatientMRN']]), pred_vals)), shap_vals))
output = data[[key,date_col,target_col]].values
output=np.concatenate((output,pred_vals[:, None],shap_vals),axis=1)
output = pd.DataFrame(output)
output.columns = [key,date_col,target_col,target_col+'_Preds'] + [trans.feature_names_clean[i] for i in recurr_feat_inds] + ['bias']
output.columns = [output.columns[i].replace('.','_') for i in range(len(output.columns))]
output[date_col] = [str(d) for d in data[date_col]]

# Write the predictions to the output table
write_data('statmods', output, write_table, key=key, check=False, safe_write=False)


