import os
import pandas as pd
import numpy as np
import argparse
#import xgboost as xgb
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso
from sklearn import naive_bayes
from sklearn import ensemble
from sklearn.metrics import mean_squared_log_error, mean_squared_error

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


def data_conversion(df, le_dict, discrete_vars, cont_vars, stage=None, one_hot=False, num_features=None):
    X, y = [], []
    bad_ids = []
    for ind, row in df.iterrows():
        feats = []
        try:
            for feat in discrete_vars:
                if feat in le_dict:
                    #print(feat, row[feat])
                    le = le_dict[feat]
                    val = le.transform([row[feat]])
                    #feats.extend(val)
                    if one_hot:
                        n_feats = len(le.classes_)
                        feat = np.zeros(n_feats)
                        if not np.isnan(val[0]):
                            feat[val[0]] = 1
                        else:
                            feat[val[0]] = np.nan
                        feats.extend(feat)
                    else:
                        feats.extend(val)

            for feat in cont_vars:
                if feat == "YearBuiltYearRemodAdd":
                    feats.append(row["YearRemodAdd"])
        except:
            bad_ids.append(ind)
            continue
        y.append(row["SalePrice"])
        X.append(feats)

    if bad_ids:
        print(stage, bad_ids)

    return X, y


def train(clf, df_train, discrete_vars, cont_vars, le_dict, one_hot=False):
    X_tr, y_tr = data_conversion(df_train, le_dict, discrete_vars, cont_vars, stage="train", one_hot=one_hot)
    clf = clf.fit(X_tr, y_tr)
    return clf


def predict(clf, df, discrete_vars, cont_vars, le_dict, one_hot=False):
    #le_dict = categorical_feature_extraction(df, discrete_vars, stage="val")
    X_test, y_test = data_conversion(df, le_dict, discrete_vars, cont_vars, stage="val", one_hot=one_hot)
    y_preds = clf.predict(X_test)
    for pred, l in zip(y_preds, y_test):
        print(pred, l)
    #print("coeff", clf.coef_)
    error = mean_squared_log_error(y_preds, y_test)
    return error


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--parameter_search', default=False, action='store_true')
    parser.add_argument('--estimator', type=str, default="decision_tree")
    parser.add_argument('--create_submission', action='store_true', default=False)
    parser.add_argument('--use_onehot', default=False, action='store_true')
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
    elif args.estimator == "rf":
        clf = RandomForestRegressor(random_state=0)

    elif args.estimator == "lasso":
        clf = Lasso(alpha=1.0, normalize=True)

    # FEATURE EXTRACTION (using all available data)
    joint_df = pd.concat([df_train, df_val, df_test], ignore_index=True)
    le_dict = categorical_feature_extraction(joint_df, discrete_vars, stage="train")

    if not args.parameter_search:
        if not args.estimator == "xgboost":
            clf = train(clf, df_train, discrete_highest_cor_vars, cont_highest_cor_vars, le_dict, one_hot=args.use_onehot)
        else:
            X_train, y_train = data_conversion(df_train, le_dict, discrete_highest_cor_vars,
                                                cont_highest_cor_vars, one_hot=args.use_onehot)
            dtrain = xgb.DMatrix(X_train, label=y_train)
            X_test, y_test = data_conversion(df_test, le_dict, discrete_highest_cor_vars,
                                                cont_highest_cor_vars, one_hot=args.use_onehot)
            dtest = xgb.DMatrix(X_test, label=y_test)
            evallist = [(dtest, 'eval'), (dtrain, 'train')]
            clf = xgb.train(param, dtest, 10, evallist)
    else:
        ### cv-based grid search
        X_train, y_train = data_conversion(df_train, le_dict, discrete_highest_cor_vars,
                                                cont_highest_cor_vars, one_hot=args.use_onehot)

        if args.estimator == "decision_tree":
            params = {
                "min_samples_split": [2, 4, 8, 16]
            }
        elif args.estimator == "rf":
            params = {
                "n_estimators": [10, 20, 50, 100, 500],
                "min_samples_split": [2, 4, 8, 16]
            }
        clf = fit_model_with_parameter_search(clf, params, 5, X_train, y_train)

    #print("Best parameters: ", clf.best_params_)
    error = predict(clf, df_val, discrete_highest_cor_vars, cont_highest_cor_vars, le_dict, one_hot=args.use_onehot)

    print("val mean_squared_log_error: ", error)

    if args.create_submission:
        outfname = convert_preds_to_submis(clf, le_dict, df_test, discrete_highest_cor_vars,
                                                    cont_highest_cor_vars,
                                                    one_hot=args.use_onehot)

        print("generated submission {}".format(outfname))
