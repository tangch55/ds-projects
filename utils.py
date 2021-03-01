import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
###

def get_feat_to_label_corr():
    pass

def get_feat_to_feat_corr():
    pass

def fit_model_with_parameter_search(estim, params, n_cv, X, y):
    grid_model = GridSearchCV(estimator=estim, param_grid=params, cv=n_cv, refit=True)
    fit_model = grid_model.fit(X, y)
    return fit_model

def categorical_feature_extraction(df, selected_vars):
    # label conversion
    count = 0
    le_dict = dict()
    untransformed_feats = list()
    for ind, row in df.T.iterrows():
        if ind in selected_vars:
            try:
                le = preprocessing.LabelEncoder()
                le.fit(row)
                print(ind, le.classes_)
                le_dict[ind] = le
                count += 1
            except:
                untransformed_feats.append(ind)
    print("un-transformed features: ", ind)
    return le_dict

def fit_test_data(clf, df_test, discrete_vars, cont_vars):
    le_dict = categorical_feature_extraction(df_test, discrete_vars)
    ids, X = [],  []
    bad_ids = []
    for ind, row in df_test.iterrows():
        ids.append(row["Id"])
        feats = []
        for feat in discrete_vars:
            if feat in le_dict:
                le = le_dict[feat]
                if str(row[feat]) == "nan":
                    val = [np.nan]
                    bad_ids.append(ind)
                else:
                    val = le.transform([row[feat]])
                feats.extend(val)
        for feat in cont_vars:
            if feat == "YearBuiltYearRemodAdd":
                feats.append(row["YearRemodAdd"])

        X.append(feats)

    if np.isnan(X).any():
        from sklearn.impute import SimpleImputer
        print("There are nan features: ", bad_ids)
        print("Using simple imputation for the missing values")
        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        X = imp.fit_transform(X)
    y_preds = clf.predict(X)
    return ids, y_preds, bad_ids


def convert_preds_to_submis(clf, df_test, discrete_vars, cont_vars):
    import datetime
    ct = datetime.datetime.now()
    ct = str(ct).replace(" ", "-")
    ids, y_preds, bad_ids = fit_test_data(clf, df_test, discrete_vars, cont_vars)
    df = pd.DataFrame({'Id': ids, 'SalePrice': y_preds})
    outfname = "submis-{}.csv".format(ct)
    df.to_csv(outfname, index=False)
    return outfname
