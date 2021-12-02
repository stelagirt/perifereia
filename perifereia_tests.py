from datetime import date, timedelta
from pandas import read_csv
import geopandas as gpd
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.metrics import classification_report
from pandas import DataFrame
import pandas as pd
from logitboost import LogitBoost
from sklearn.tree import DecisionTreeRegressor

def indexdict(dfcol):
    lu = list(dfcol.unique())
    lu_dict = {x:lu.index(x)+1 for x in lu}
    return lu_dict

def normalized_values(y,dfmax, dfmin, dfmean, dfstd, t = None):
    if not t:
        a = (y- dfmin) / (dfmax - dfmin)
        return(a)
    elif t=='std':
        a = (y - dfmean) / dfstd
        return(a)
    elif t=='no':
        return y

def index_string_values(X_unnorm, str_classes):
    indexdicts = {}
    for str_class in str_classes:
        indexdicts[str_class]=indexdict(X_unnorm[str_class])
    X_unnorm_int = X_unnorm.copy()
    for c in str_classes:
        print(c)
        X_unnorm_int[c] = X_unnorm.apply(lambda x: convtoindex(x[c],indexdicts[c]),axis=1)
    return X_unnorm_int

def convtoindex(y, lu_dict):
    return(lu_dict[y])

def normalize_dataset(X_unnorm_int, norm_type = None):
    X = DataFrame()
    for c in X_unnorm_int.columns:
        print(c)
        dfmax = X_unnorm_int[c].max()
        dfmin = X_unnorm_int[c].min()
        dfmean = X_unnorm_int[c].mean()
        dfstd = X_unnorm_int[c].std()
        X[c] = X_unnorm_int.apply(lambda x: normalized_values(x[c],dfmax, dfmin,dfmean,dfstd, norm_type),axis=1)
    return X

def Neural_Net(file):
    df = read_csv('/home/sgirtsou/Documents/processes/pre-processing/BASE/training_dataset.csv')
    df = df.dropna()
    df_part = df[['id','max_temp','min_temp', 'mean_temp', 'res_max', 'dir_max', 'dom_vel', 'dom_dir','Corine',
           'Slope','DEM','Aspect', 'ndvi','fire']].copy()
    # split into input and output columns
    #X_unnorm, y_int = df_part[['max_temp','min_temp', 'mean_temp', 'res_max', 'dir_max', 'dom_vel', 'dom_dir',
     #                          'Corine','Slope','DEM','Aspect', 'ndvi']], df_part['fire']
    X_unnorm, y_int = df_part[['Corine','Slope','DEM','Aspect', 'ndvi']], df_part['fire']
    str_classes = ['Corine']
    X_unnorm_int = index_string_values(X_unnorm, str_classes)
    X = normalize_dataset(X_unnorm_int, 'std')

    y=y_int

    X_ = X.values
    y_ = y.values
    y.shape, X.shape, type(X_), type(y_)

    # split into train and test datasets
    X_train, X_test, y_train, y_test = train_test_split(X_, y_, test_size=0.01)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    # determine the number of input features
    n_features = X_train.shape[1]

    n_features
    # define model

    np.count_nonzero(y_test == 0),len(y_test)

    type(y_train)

    # define model
    model = Sequential()
    model.add(Dense(16, activation='relu', input_shape=(n_features,)))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(4, activation='relu'))
    #model.add(Dense(1, activation='sigmoid'))
    model.add(Dense(2, activation='softmax'))

    # compile the model
    #model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    from tensorflow.keras.optimizers import Adam
    adam = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(optimizer=adam, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # fit the model
    from tensorflow.keras.callbacks import EarlyStopping, TensorBoard

    import os
    es = EarlyStopping(monitor='loss', mode='min', patience = 20)

    log_dir = os.path.join('.\\logs\\s2')
    tb = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=True, profile_batch = 100000000)
    #model.fit(X_train[:,1:], y_train, epochs=100, batch_size=1000, callbacks = [es, tb])
    model.fit(X_train, y_train, epochs=250, batch_size=1000, callbacks = [es, tb])

    # evaluate the model
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print('Test Accuracy: %.3f' % acc)

    preds = np.argmax(model.predict(X_test), axis = 1)

    #file = '/home/sgirtsou/vravrona13082013.csv'
    df_greece = read_csv(file)
    print(file)

    df_greece = df_greece.dropna()

    #X_greece_unnorm = df_greece[['max_temp', 'min_temp', 'mean_temp', 'res_max',
     #          'dir_max', 'dom_vel', 'dom_dir', 'Corine', 'DEM', 'Slope', 'Aspect', 'ndvi']]
    X_greece_unnorm = df_greece[['Corine', 'Slope', 'DEM', 'Aspect', 'ndvi']]

    #Y_greece = df_greece[['fire']]
    print('X_greece_unnorm.shape',X_greece_unnorm.shape)

    str_classes = ['Corine']
    X_greeceunnorm_int = index_string_values(X_greece_unnorm, str_classes)
    for c in X_greeceunnorm_int.columns:
        X_greeceunnorm_int[c] = pd.to_numeric(X_greeceunnorm_int[c], errors='coerce')

    X_greece = normalize_dataset(X_greeceunnorm_int, 'std')
    #X_greece = X_greece.rename(columns={'LU':'Corine'})

    Y_pred_greece = model.predict(X_greece.values)
    Y_pred_greece_cl = model.predict_classes(X_greece.values)
    Y_pred_greece_f = model.predict(X_greece.values)

    Y_pred_f = (Y_pred_greece_f[:,1]>0.5).astype(int)

    Y_pred_greece_cl_df = DataFrame({'Class_pred_nn': Y_pred_greece_cl})
    Y_pred_greece_df = DataFrame({'Class_0_proba_nn': Y_pred_greece[:, 0], 'Class_1_proba_nn': Y_pred_greece[:, 1]})

    df_results = pd.concat([df_greece, Y_pred_greece_cl_df, Y_pred_greece_df], axis=1)

    #df_results.to_csv('/home/sgirtsou/Documents/vravrona/stat_temp_wind2_NN_results.csv')
    return df_results


def logitboost(res):
    #df=pd.read_csv("C:/Users/User/Documents/projects/FFP/ready dataset/dataset_norm_v2.csv")
    df = pd.read_csv('/home/sgirtsou/Documents/processes/pre-processing/BASE/training_dataset.csv', low_memory=False)

    df = df[df.ndvi != '--']
    df = df.dropna() #drop null

    print("df columns", df.columns)
    print("df shape", df.shape)


    #X= df[['max_temp','min_temp', 'mean_temp', 'res_max', 'dir_max', 'dom_vel', 'dom_dir',
    #                           'Corine','Slope','DEM','Aspect', 'ndvi']]

    X = df[['Corine', 'Slope', 'DEM', 'Aspect', 'ndvi']]

    Y = df["fire"]

    # training & testing data
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.01) #20% hold out for testing

    # initialize classifier
    logitb = LogitBoost(DecisionTreeRegressor(max_depth=10),
                        n_estimators=100, random_state=0)

    #train the model
    logitb.fit(X_train, y_train)

    # run on test set
    Y_pred = logitb.predict(X_test)

    report = classification_report(y_test, Y_pred)
    print(report)

    fi = pd.DataFrame({'feature': X_train.columns,
                       'importance': logitb.feature_importances_}).\
                        sort_values('importance', ascending = False)

    importances = logitb.feature_importances_


    indices = np.argsort(importances)[::-1]

    print("Feature ranking:")


    fi = pd.DataFrame({'feature': X.columns,
                       'importance': logitb.feature_importances_}).\
                        sort_values('importance', ascending = False)


    l = [x for _,x in sorted(zip(logitb.feature_importances_,X.columns), reverse=True)]
    for a,b in zip(sorted(logitb.feature_importances_, reverse=True), l):
        print("{0:12s}: {1}".format(b, a))

    df_greece = res
    df_greece = df_greece[df_greece.ndvi != '--']
    print(file)
    #X_greece = df_greece[['max_temp', 'min_temp', 'mean_temp', 'res_max','dir_max', 'dom_vel', 'dom_dir',
     #                             'Corine', 'DEM', 'Slope', 'Aspect', 'ndvi']]
    X_greece = df_greece[['Corine', 'DEM', 'Slope', 'Aspect', 'ndvi']]

    #X_greece = X_greece.rename(columns={'LU':'Corine', 'prcp':'rain_7days'})

    Y_pred_greece = logitb.predict(X_greece)
    Y_pred_greece_proba = logitb.predict_proba(X_greece)

    Y_pred_greece_df = pd.DataFrame({'Class_pred_lb': Y_pred_greece})
    Y_pred_greece_proba_df = pd.DataFrame({'Class_0_proba_lb': Y_pred_greece_proba[:, 0], 'Class_1_proba_lb': Y_pred_greece_proba[:, 1]})
    df_results = pd.concat([df_greece, Y_pred_greece_df, Y_pred_greece_proba_df], axis=1)

    #df_results.to_csv('/home/sgirtsou/Documents/vravrona/stat_temp_wind2_NN_LB_results.csv')
    return df_results

def lb_nn_comb(lb, nn):
    proba = max(lb, nn) if lb > 0.98 else nn
    class1 = 1 if proba >= 0.5 else 0
    return proba, class1

def comb(res_lb_nn):
    df_greece = res_lb_nn

    probas = df_greece[['Class_1_proba_lb', 'Class_1_proba_nn']].to_numpy()

    v_lb_nn_comb = np.vectorize(lb_nn_comb)
    comb_probas = v_lb_nn_comb(probas[:, 0], probas[:, 1])
    # cn+=len(probas[(probas[:,0] >=0.98) & (probas[:,1] < 0.5) & (probas[:,2] == 1)])
    allprobas = np.vstack([comb_probas[0], probas[:, 1]])
    allprobas = np.transpose(allprobas)
    #cn += len(allprobas[(allprobas[:, 0] >= 0.5) & (allprobas[:, 1] == 1)])
    #print(cn)
    if np.any(np.isnan(comb_probas[0])) or np.any(np.isnan(comb_probas[1])):
        print("comb has nan in %s" % file)

    df_greece['Comb_proba_1'] = pd.Series(comb_probas[0])
    df_greece['Comb_class_pred'] = pd.Series(comb_probas[1])

    df_greece.to_csv('/home/sgirtsou/Documents/vravrona/stat_newlu_comb_results.csv')
    return df_greece

if __name__=='__main__':
    file = '/home/sgirtsou/dataset_new_lu.csv'
    res_nn = Neural_Net(file)
    res_lb_nn = logitboost(res_nn)
    comb = comb(res_lb_nn)