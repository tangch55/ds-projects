import os
import pandas as pd
import numpy as np
import argparse
#import xgboost as xgb
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn import naive_bayes
from sklearn import ensemble
from sklearn.metrics import mean_squared_log_error

####
from utils import fit_model_with_parameter_search, categorical_feature_extraction, convert_preds_to_submis

DATA_DIR="/Users/chengtang/Documents/Cheng-2021/house-price-prediction/house-prices-advanced-regression-techniques"

highest_cor_vars = [
'OverallQual',
'GrLivArea',
'TotalBsmtSF',
'GarageCars',
'FullBath',
'GarageYrBlt',
'Fireplaces',
]

discrete_vars = ["MSSubClass", "Street", "Alley",
                 "LotShape", "OverallQual", "OverallCond",
                 "LandContour", "Utilities", "LotConfig",
                 "Neighborhood", "Condition1", "Condition2",
                 "BldgType", "HouseStyle", "RoofStyle", "RoofMatl",
                 "Exterior1st", "Exterior2nd", "MasVnrType",
                 "BsmtFinType1", "BsmtFinType2", "BsmtExposure",
                 "Heating", "HeatingQC", "CentralAir",
                 "Electrical", "BsmtFullBath", "FullBath", "HalfBath",
                 "BedroomAbvGr", "KitchenAbvGr", "KitchenQual", "TotRmsAbvGrd",
                 "Functional", "Fireplaces", "FireplaceQu",
                 "GarageType", "GarageFinish", "GarageCars",
                 "GarageQual", "GarageCond",
                 "PavedDrive", "PoolQC", "Fence",
                 "SaleType", "SaleCondition",
                 "YearRemodAdd", "YearBuilt"
                 ]

discrete_highest_cor_vars = set(highest_cor_vars) & set(discrete_vars)
cont_highest_cor_vars = set(highest_cor_vars) - set(discrete_vars)


def load_df(filepath, split_train=False, train_ratio=0.8):
    df = pd.read_csv(filepath)
    if split_train:
        msk = np.random.rand(len(df)) < train_ratio
        df_train = df[msk]
        df_val = df[~msk]
        return (df_train, df_val)
    else:
        return (df,)




def data_conversion(df, le_dict, discrete_vars, cont_vars):
    X, y = [], []
    for ind, row in df.iterrows():
        feats = []
        try:
            for feat in discrete_vars:
                if feat in le_dict:
                    #print(feat, row[feat])
                    le = le_dict[feat]
                    val = le.transform([row[feat]])
                    feats.extend(val)
            for feat in cont_vars:
                if feat == "YearBuiltYearRemodAdd":
                    feats.append(row["YearRemodAdd"])
        except:
            bad_ids.append(ind)
            continue
        y.append(row["SalePrice"])
        X.append(feats)
    return X, y


def train(clf, df_train, discrete_vars, cont_vars):
    le_dict = categorical_feature_extraction(df_train, discrete_vars)
    X_tr, y_tr = data_conversion(df_train, le_dict, discrete_vars, cont_vars)
    clf = clf.fit(X_tr, y_tr)
    return clf


def predict(clf, df_test, discrete_vars, cont_vars):
    le_dict = categorical_feature_extraction(df_test, discrete_vars)
    X_test, y_test = data_conversion(df_test, le_dict, discrete_vars, cont_vars)
    y_preds = clf.predict(X_test)
    error = mean_squared_log_error(y_preds, y_test)
    return error


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--parameter_search', default=False, action='store_true')
    parser.add_argument('--estimator', type=str, default="decision_tree")
    parser.add_argument('--create_submission', action='store_true', default=False)
    args = parser.parse_args()

    ###
    fpath_train = os.path.join(DATA_DIR, "train.csv")
    fpath_test = os.path.join(DATA_DIR, "test.csv")

    df_train, df_val = load_df(fpath_train, split_train=True, train_ratio=0.8)

    df_test = load_df(fpath_test)[0]

    ### --- train and validate
    if args.estimator == "decision_tree":
        clf = tree.DecisionTreeRegressor(random_state=0)
    elif args.estimator == "gbm":
        clf = GradientBoostingClassifier(random_state=0)
    elif args.estimator == "xgboost":
        clf = xgb.Booster({'nthread': 5})
    if not args.parameter_search:
        if not args.estimator == "xgboost":
            clf = train(clf, df_train, discrete_highest_cor_vars, cont_highest_cor_vars)
        else:
            le_dict = categorical_feature_extraction(df_train, discrete_highest_cor_vars)
            X_train, y_train = data_conversion(df_train, le_dict, discrete_highest_cor_vars, cont_highest_cor_vars)
            dtrain = xgb.DMatrix(X_train, label=y_train)
            X_test, y_test = data_conversion(df_test, le_dict, discrete_highest_cor_vars, cont_highest_cor_vars)
            dtest = xgb.DMatrix(X_test, label=y_test)
            evallist = [(dtest, 'eval'), (dtrain, 'train')]
            clf = xgb.train(param, dtest, 10, evallist)

    else:
        ### cv-based grid search
        le_dict = categorical_feature_extraction(df_train, discrete_highest_cor_vars)
        X_train, y_train = data_conversion(df_train, le_dict, discrete_highest_cor_vars, cont_highest_cor_vars)
        params = {
            "min_samples_split": [2, 4, 8, 16]
        }
        clf = fit_model_with_parameter_search(clf, params, 5, X_train, y_train)

    #print("Best parameters: ", clf.best_params_)
    error = predict(clf, df_val, discrete_highest_cor_vars, cont_highest_cor_vars)
    print("Val mean_squared_log_error: ", error)

    if args.create_submission:
        outfname = convert_preds_to_submis(clf, df_test, discrete_highest_cor_vars, cont_highest_cor_vars)
        print("generated submission {}".format(outfname))
