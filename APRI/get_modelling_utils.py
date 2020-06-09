'''
This script contains methods to compute several modelling steps:

get_features_selection(): trains a random_forest classifier and select the features that contribute significantly to the model outputs



get_dataframe_split(): to split source dataframe into train, test and validation.
Different parameters allow for customize splits


'''
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score,confusion_matrix, roc_auc_score, f1_score
import xgboost as xgb
from sklearn.pipeline import Pipeline





def get_feature_selection(y_train,x_train,y_test,x_test,y_val,x_val,fs_gridsearch,fs_threshold,df_test):
    cls = RandomForestClassifier()
    if fs_gridsearch:
        grid_params_rf= [{'n_estimators': [500],
                   'max_depth': [32],
                   'max_features': ["auto"]}]
        gs_fs = GridSearchCV(estimator=cls, param_grid=grid_params_rf, scoring='accuracy', cv=5, verbose=10,
                             n_jobs=-1)
        gs_fs.fit(x_train, y_train)
        # Best params
        print('Best params: %s' % gs_fs.best_params_)
        # Best training data r2
        print('Best training accuracy: %.3f' % gs_fs.best_score_)
        print(cls.get_params())
        cls.set_params(**gs_fs.best_params_)
    else:
        grid_params_rf= {'n_estimators': 500,
                   'max_depth': 16,
                   'max_features': "auto"}
        cls.set_params(**grid_params_rf)
    sel = SelectFromModel(cls,threshold=fs_threshold)
    sel.fit(x_train, y_train)
    col_rem=sel.get_support()
    x_train_red=sel.transform(x_train)
    x_test_red=sel.transform(x_test)
    cls.fit(x_train,y_train)
    y_pred=cls.predict(x_test)
    print(accuracy_score(y_test, y_pred))
    cls.fit(x_train_red,y_train)
    y_pred2=cls.predict(x_test_red)
    print(accuracy_score(y_test, y_pred2))
    if(abs(accuracy_score(y_test, y_pred)-accuracy_score(y_test, y_pred2)))<0.02:
        print("Done feature selection. Lost accuracy: ",(accuracy_score(y_test, y_pred)-accuracy_score(y_test, y_pred2)))
        x_train=x_train_red
        print(x_train.shape)
        x_test=x_test_red
        print(x_test.shape)
        x_val=sel.transform(x_val)
        print(x_val.shape)
        df_test2=df_test.drop(['target'],axis=1)
        columns=df_test2.columns[col_rem]
    else:
        print("Features selection rejected for input threshold. Lost accuracy: ",(accuracy_score(y_test, y_pred)-accuracy_score(y_test, y_pred2)))
        columns=df_test.columns
    return x_train,x_test,x_val,columns,cls.get_params()


def get_feature_selection_theme(x_train,x_test,x_val,features_dict):
    remove_variables=dict()
    remove_variables['lowlevel.barkbands_kurtosis_mean']=True
    remove_variables['lowlevel.barkbands_kurtosis_var',
    remove_variables['lowlevel.barkbands_skewness_mean',
    remove_variables['lowlevel.barkbands_skewness_var',
    remove_variables['lowlevel.barkbands_spread_mean',
    remove_variables['lowlevel.barkbands_spread_var', 
    remove_variables['lowlevel.hfc_mean',
    remove_variables['lowlevel.hfc_var', 
    remove_variables['lowlevel.pitch_salience_mean',
    remove_variables['lowlevel.pitch_salience_var', 
    remove_variables['lowlevel.silence_rate_20dB_mean1',
    remove_variables['lowlevel.silence_rate_20dB_var1',
    remove_variables['lowlevel.silence_rate_30dB_mean1',
    remove_variables['lowlevel.silence_rate_30dB_var1',
    remove_variables['lowlevel.silence_rate_60dB_mean1',
    remove_variables['lowlevel.silence_rate_60dB_var1',
    remove_variables['lowlevel.spectral_centroid_mean1',
    remove_variables['lowlevel.spectral_centroid_var1',
    remove_variables['lowlevel.spectral_complexity_mean1',
    remove_variables['lowlevel.spectral_crest_mean1', 
    remove_variables['lowlevel.spectral_crest_var1',
    remove_variables['lowlevel.spectral_decrease_mean1',
    remove_variables['lowlevel.spectral_decrease_var1',
    remove_variables['lowlevel.spectral_energy_mean1',
    remove_variables['lowlevel.spectral_energy_var1',
    remove_variables['lowlevel.spectral_energyband_high_mean1',
    remove_variables['lowlevel.spectral_energyband_high_var1',
    remove_variables['lowlevel.spectral_energyband_low_mean1',
    remove_variables['lowlevel.spectral_energyband_low_var1',
    remove_variables['lowlevel.spectral_energyband_middle_high_mean1',
    'lowlevel.spectral_energyband_middle_high_var1',
    'lowlevel.spectral_energyband_middle_low_mean1',
    'lowlevel.spectral_energyband_middle_low_var1',
    'lowlevel.spectral_flatness_db_mean1',
    'lowlevel.spectral_flatness_db_var1',
    'lowlevel.spectral_rms_mean1', 
    'lowlevel.spectral_rms_var1',
    'lowlevel.spectral_rolloff_mean1',
    'lowlevel.spectral_rolloff_var1',
    'lowlevel.spectral_strongpeak_mean1',
    'lowlevel.spectral_strongpeakh_var1',
    'lowlevel.sspectral_complexity_var1',
    'lowlevel.zerocrossingrate_mean1',
    'lowlevel.zerocrossingrate_var1',
    'tonal.tuning_diatonic_strength1',
    'tonal.tuning_equal_tempered_deviation1',
    'tonal.tuning_frequency1',
    'tonal.tuning_nontempered_energy_ratio1',
    'tonal.tuning_nontempered_peaks_energy_ratio1'
    
    
    
    'lowlevel.barkbands_mean1' #27
    'lowlevel.barkbands_var1' #27
    'lowlevel.erbbands_mean1' #40
    'lowlevel.erbbands_var1' #40
    'lowlevel.gfcc_mean1' #40
    'lowlevel.gfcc_var1' #40
    'lowlevel.melbands_mean1', #40
    'lowlevel.melbands_var1', #40
    'lowlevel.mfcc_mean1', #13
    'lowlevel.mfcc_var1'#13
    'lowlevel.spectral_contrast_mean1' #12
    'lowlevel.spectral_contrast_var1', #12
    'lowlevel.temporal_lpc_mean1' #11,
    'lowlevel.temporal_lpc_var1' #11
    'tonal.hpcp_mean1', #36
    'tonal.hpcp_var1', #36
    'tonal.thpcp1', #36













def train_xgb(x_train,y_train,x_test,y_test,x_val,y_val,xgb_gridsearch):
    print('Training model XGB...')
    dtrain = xgb.DMatrix(x_train, label=y_train)
    dtest = xgb.DMatrix(x_test, label=y_test)
    dval = xgb.DMatrix(x_val, label=y_val)

#Params
    params = {
    'max_depth':3,
    'min_child_weight': 1,
    'eta':0.03,
    'subsample': 1,
    'num_class':14,
    'colsample_bytree': 0.5,
    'objective':'multi:softmax',
    }
    params['eval_metric'] = "mlogloss"
    num_boost_round = 500
    if xgb_gridsearch:
        print('Gridsearch..')
        # Parameters max_depth and min_child_weight
        gridsearch_params = [
        (max_depth, min_child_weight)
        for max_depth in [3,5]
        for min_child_weight in [1,3]
    ]
        min_metric = float("Inf")
        best_params = None
        for max_depth, min_child_weight in gridsearch_params:
            print("CV with max_depth={}, min_child_weight={}".format(max_depth,min_child_weight))  # Update our parameters
            params['max_depth'] = max_depth
            params['min_child_weight'] = min_child_weight  # Run CV
            cv_results = xgb.cv(
                params,
                dtrain,
                num_boost_round=num_boost_round,
                seed=42,
                nfold=2,
                metrics={'mlogloss'},
                early_stopping_rounds=2,
                verbose_eval=5
            )
            # Update best mlogloss
            mean_metric = cv_results['test-mlogloss-mean'].min()
            boost_rounds = cv_results['test-mlogloss-mean'].argmin()
            print("\tMAE {} for {} rounds".format(mean_metric, boost_rounds))
            if mean_metric < min_metric:
                min_metric = mean_metric
                best_params = (max_depth, min_child_weight)
                print("Best params: {}, {}, MAE: {}".format(best_params[0], best_params[1], min_metric))
            params['max_depth'] = best_params[0]
            params['min_child_weight'] = best_params[1]
        # Parameters sumbsample and colsample
        gridsearch_params = [
        (subsample, colsample)
        for subsample in [1]
        for colsample in [0.1]
    ]
        min_mae = float("Inf")
        best_params = None  # We start by the largest values and go down to the smallest
        for subsample, colsample in reversed(gridsearch_params):
            print("CV with subsample={}, colsample={}".format(subsample,colsample))  # We update our parameters
            params['subsample'] = subsample
            params['colsample_bytree'] = colsample  # Run CV
            cv_results = xgb.cv(
                params,
                dtrain,
                num_boost_round=num_boost_round,
                seed=42,
                nfold=2,
                metrics={'mlogloss'},
                early_stopping_rounds=5,
                verbose_eval=5
            )
            # Update best score
            mean_mae = cv_results['test-mlogloss-mean'].min()
            boost_rounds = cv_results['test-mlogloss-mean'].argmin()
            print("\tMAE {} for {} rounds".format(mean_mae, boost_rounds))
            if mean_mae < min_mae:
                min_mae = mean_mae
                best_params = (subsample, colsample)
                print("Best params: {}, {}, MAE: {}".format(best_params[0], best_params[1], min_mae))
            params['subsample'] = best_params[0]
            params['colsample_bytree'] = best_params[1]
        # Parameter eta
        min_mae = float("Inf")
        for eta in [.1, .05]:
            print("CV with eta={}".format(eta))  # We update our parameters
            params['eta'] = eta  # Run and time CV
            cv_results = xgb.cv(
                params,
                dtrain,
                num_boost_round=num_boost_round,
                seed=42,
                nfold=2,
                metrics=['mlogloss'],
                early_stopping_rounds=5,
                verbose_eval=5
            )  # Update best score
            mean_mae = cv_results['test-mlogloss-mean'].min()
            boost_rounds = cv_results['test-mlogloss-mean'].argmin()
            print("\tMAE {} for {} rounds\n".format(mean_mae, boost_rounds))
            if mean_mae < min_mae:
                min_mae = mean_mae
                best_params = eta
                print("Best params: {}, MAE: {}".format(best_params, min_mae))
            params['eta'] = best_params
    num_boost_round = 2000
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtrain,"Train"),(dtest, "Test")],
        early_stopping_rounds=50
    )
    print(params)
    print('Test predictions with trained mode...')
    y_pred = model.predict(dtest)
    print('Train predictions with trained mode...')
    y_pred_t = model.predict(dtrain)
    print('Validation predictions with trained mode...')
    y_pred_val = model.predict(dval)
    print('Confussion matrix test:')
    print(confusion_matrix(y_test, y_pred))
    print('Confussion matrix validation:')
    print(confusion_matrix(y_val, y_pred_val))
    print('Prediction accuracy for test: %.3f ' % accuracy_score(y_test, y_pred))
    print('Prediction accuracy for train: %.3f ' % accuracy_score(y_train, y_pred_t))
    print('Prediction accuracy for validation: %.3f ' % accuracy_score(y_val, y_pred_val))
    return model

def train_rf(x_train,y_train,x_test,y_test,x_val,y_val,rf_gridsearch):
    print('Training model random forest...')
    cls = RandomForestClassifier()
    if rf_gridsearch:
        print('Tuning parameters...')
        grid_params_rf = [{'bootstrap':[False],
                           'n_estimators': [500,1000],
                           'max_depth': [8,16],
                           'max_features': ["auto"]}]
        gs_fs = GridSearchCV(estimator=cls, param_grid=grid_params_rf, scoring='f1_weighted', cv=5, verbose=10,
                             n_jobs=-1)
        gs_fs.fit(x_train, y_train)
        # Best params
        print('Best params: %s' % gs_fs.best_params_)
        # Best training data r2
        print('Best training accuracy: %.3f' % gs_fs.best_score_)
        cls.set_params(**gs_fs.best_params_)
        model = cls.fit(x_train, y_train)
    else:
        params_rf = {'bootstrap': False,
                          'n_estimators': 500,
                          'max_depth': 16,
                          'max_features': "auto"}
        cls.set_params(**params_rf)
        model=cls.fit(x_train, y_train)
    print(print(cls.get_params()))
    print('Test predictions with trained mode...')
    y_pred = model.predict(x_test)
    print('Train predictions with trained mode...')
    y_pred_t = model.predict(x_train)
    print('Validation predictions with trained mode...')
    y_pred_val = model.predict(x_val)
    print('Confussion matrix test:')
    print(confusion_matrix(y_test, y_pred))
    print('Confussion matrix validation:')
    print(confusion_matrix(y_val, y_pred_val))
    print('Prediction accuracy for test: %.3f ' % accuracy_score(y_test, y_pred))
    print('Prediction accuracy for train: %.3f ' % accuracy_score(y_train, y_pred_t))
    print('Prediction accuracy for validation: %.3f ' % accuracy_score(y_val, y_pred_val))
    return model

def train_svc(x_train,y_train,x_test,y_test,x_val,y_val,svc_gridsearch):
    print('Training model svc...')


    pipe_svc = Pipeline([('scl', StandardScaler()), ('cls', SVC())])
    if svc_gridsearch:
        print('Tuning parameters...')
        grid_params_svc = [{'cls__kernel': ['linear','rbf'],
                            'cls__gamma': [0.001],
                            'cls__C': [1,100,10000]}]
        gs_svc = GridSearchCV(estimator=pipe_svc, param_grid=grid_params_svc, scoring='f1_weighted', cv=5, verbose=10,
                             n_jobs=-1)
        gs_svc.fit(x_train, y_train)
        # Best params
        print('Best params: %s' % gs_svc.best_params_)
        # Best training data r2
        print('Best training accuracy: %.3f' % gs_svc.best_score_)
        #pipe_svc.steps[1][1].set_params(**gs_svc.best_params_)
        model=gs_svc.best_estimator_
    else:
        grid_params_svc = {'kernel': 'rbf',
                            'gamma': 0.001,
                            'C': 1}
        pipe_svc.steps[1][1].set_params(**grid_params_svc)
        model=pipe_svc.fit(x_train, y_train)
    print(model.steps[1][1].get_params())
    print('Test predictions with trained mode...')
    y_pred = model.predict(x_test)
    print('Train predictions with trained mode...')
    y_pred_t = model.predict(x_train)
    print('Validation predictions with trained mode...')
    y_pred_val = model.predict(x_val)
    print('Confussion matrix test:')
    print(confusion_matrix(y_test, y_pred))
    print('Confussion matrix validation:')
    print(confusion_matrix(y_val, y_pred_val))
    print('Prediction accuracy for test: %.3f ' % accuracy_score(y_test, y_pred))
    print('Prediction accuracy for train: %.3f ' % accuracy_score(y_train, y_pred_t))
    print('Prediction accuracy for validation: %.3f ' % accuracy_score(y_val, y_pred_val))
    return model


def train_gb(x_train,y_train,x_test,y_test,x_val,y_val,gb_gridsearch):
    print('Training model gradient boosting with sklearn...')
    cls = GradientBoostingClassifier()
    if gb_gridsearch:
        print('Tuning parameters...')
        grid_params_gb = [{'learning_rate':[0.05,0.1],
                           'n_estimators': [500],
                           'max_depth': [3,6],
                           'subsample': [1],
                           'min_samples_split': [2,4],
                           'min_samples_leaf': [1,2],
                           'max_features':['sqrt', 'log2'],
                           'verbose':[1]
                           }]
        gs_gb = GridSearchCV(estimator=cls, param_grid=grid_params_gb, scoring='f1_weighted', cv=5, verbose=10,
                             n_jobs=-1)
        gs_gb.fit(x_train, y_train)
        # Best params
        print('Best params: %s' % gs_gb.best_params_)
        # Best training data r2
        print('Best training accuracy: %.3f' % gs_gb.best_score_)
        cls.set_params(**gs_gb.best_params_)
        model = cls.fit(x_train, y_train)
    else:
        params_gb = {'learning_rate':0.05,
                           'n_estimators': 500,
                           'max_depth': 3,
                           'subsample': 1,
                           'min_samples_split': 2,
                           'min_samples_leaf': 1,
                           'max_features':'sqrt',
                           'verbose':2}
        cls.set_params(**params_gb)
        model=cls.fit(x_train, y_train)
    print(print(cls.get_params()))
    print('Test predictions with trained mode...')
    y_pred = model.predict(x_test)
    print('Train predictions with trained mode...')
    y_pred_t = model.predict(x_train)
    print('Validation predictions with trained mode...')
    y_pred_val = model.predict(x_val)
    print('Confussion matrix test:')
    print(confusion_matrix(y_test, y_pred))
    print('Confussion matrix validation:')
    print(confusion_matrix(y_val, y_pred_val))
    print('Prediction accuracy for test: %.3f ' % accuracy_score(y_test, y_pred))
    print('Prediction accuracy for train: %.3f ' % accuracy_score(y_train, y_pred_t))
    print('Prediction accuracy for validation: %.3f ' % accuracy_score(y_val, y_pred_val))
    return model
