import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, KFold, cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error, mean_squared_log_error
###

class MyLabelEncoder(object):
    def __init__(self, nan_val=None):
        self.label_ind = None
        self.label_dict = None
        self.nan_val = nan_val

    def fit(self, column):
        self.label_ind = 0
        self.label_dict = dict()
        for label in column:
            if label not in self.label_dict:
                self.label_dict[label] = self.label_ind
                self.label_ind += 1

    def get_classes(self):
        return list(self.label_dict.keys())

    def get_nan_encoding(self):
        if self.nan_val is not None and self.nan_val in self.label_dict:
            return self.label_dict[self.nan_val]
        else:
            return None

    def transform(self, column):
        column_encodings = []
        #print(self.label_dict, column[0])
        for label in column:
            column_encodings.append(self.label_dict[label])
        return np.array(column_encodings)


def get_feat_to_label_corr():
    pass

def get_feat_to_feat_corr():
    pass

def acc(preds, labels):
    pass


def na_stat(df, drop_bad_features=True):
    Total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum() / df.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([Total, percent], axis=1, keys=['Total', 'Percent'])
    print(missing_data.head(25))
    if drop_bad_features:
        print(" bad features ")
        print(missing_data[missing_data['Percent'] > 0.01].T.columns)
        return missing_data[missing_data['Percent'] > 0.01].T.columns


def categorical_feature_extraction(df, selected_vars, fill_null=False, stage=None):
    # label conversion
    count = 0
    le_dict = dict()
    untransformed_feats = list()

    # convert null values
    target = df['SalePrice']
    df_feats = df.drop(columns=['SalePrice'])
    if fill_null:
        df_feats.fillna("NO VALUE", inplace=True)

    for ind, row in df_feats.T.iterrows():
        if ind in selected_vars:
            #try:
                #le = preprocessing.LabelEncoder()
            if fill_null:
                le = MyLabelEncoder(nan_val="NO VALUE")
            else:
                le = MyLabelEncoder()
            le.fit(row)
            #print(stage, len(row), ind, le.get_classes())
            le_dict[ind] = le
            count += 1
            # except:
            #     untransformed_feats.append(ind)
    print("un-transformed features: ", ind)
    df_feats['SalePrice'] = target
    #print(df_feats.head())
    return df_feats, le_dict


#######

def fit_model_with_parameter_search(estim, params, n_cv, X, y, transform_target=False):
    if not transform_target:
        scorer = make_scorer(mean_squared_log_error, greater_is_better=False)
    else:
        scorer = make_scorer(mean_squared_error, greater_is_better=False)
    grid_model = GridSearchCV(estimator=estim, param_grid=params, cv=n_cv, refit=True, scoring=scorer)
    fit_model = grid_model.fit(X, y)
    print(grid_model.best_params_)
    return fit_model

def fit_test_data(clf, df_test, X_test):
    #le_dict = categorical_feature_extraction(df_test, discrete_vars)
    ids = []
    for ind, row in df_test.iterrows():
        ids.append(row["Id"])
    assert len(ids) == len(X_test)
    y_preds = clf.predict(X_test)
    return ids, y_preds

#######
def msle_cv_eval(X, y, est, transform_target=False):
    if transform_target:
        y = np.log1p(y)
        #scorer = make_scorer(mean_squared_error, greater_is_better=False)
        kf = KFold(5, shuffle=True, random_state=0)
        rmse = np.sqrt(-cross_val_score(est, X, y, scoring="neg_mean_squared_error", cv=kf))
        return rmse
    else:
        #scorer = make_scorer(mean_squared_log_error, greater_is_better=False)
        kf = KFold(5, shuffle=True, random_state=0)
        rmsle = np.sqrt(-cross_val_score(est, X, y, scoring="neg_mean_squared_log_error", cv=kf))
        return rmsle

#######
def convert_preds_to_submis(clf, df_test, X_test, transform_target=False):
    import datetime
    ct = datetime.datetime.now()
    ct = str(ct).replace(" ", "-")
    ids, y_preds = fit_test_data(clf, df_test, X_test)
    if transform_target:
        y_preds = np.expm1(y_preds)
    df = pd.DataFrame({'Id': ids, 'SalePrice': y_preds})
    outfname = "submis-{}.csv".format(ct)
    df.to_csv(outfname, index=False)
    return outfname
