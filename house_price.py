import os
import pickle
import pandas as pd
import numpy as np
import argparse
import xgboost as xgb
from scipy.stats import skew

from sklearn import tree
from sklearn.ensemble import (
GradientBoostingRegressor,
RandomForestRegressor,
ExtraTreesRegressor,
StackingRegressor,
IsolationForest
)
from sklearn.linear_model import Lasso, RidgeCV, LassoCV
from sklearn import naive_bayes
from sklearn.svm import SVC
from sklearn import ensemble
from sklearn.metrics import mean_squared_log_error, mean_squared_error, accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
####
from utils import (
fit_model_with_parameter_search,
categorical_feature_extraction,
convert_preds_to_submis,
na_stat,
msle_cv_eval
)

from viz_and_transform import (
normal_fit_1d,
label_normalization,
correlation_heatmap,
compare_predictor_power
)

np.random.seed(10)

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


cont_vars = ["LotFrontage", "LotArea",  "MasVnrArea",
                   "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF",
                  "BsmtUnfSF", "TotalBsmtSF", "1stFlrSF", "2ndFlrSF",
                  "LowQualFinSF", "GrLivArea", "BsmtHalfBath",
                  "WoodDeckSF", "OpenPorchSF", "EnclosedPorch",
                   "3SsnPorch", "ScreenPorch", "PoolArea", "GarageArea"
            ]

discrete_vars = ["MSSubClass", "Street", "Alley",
                 "ExterQual", "ExterCond", "LandSlope", "Foundation",
                 "BsmtCond", "BsmtQual",
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


time_vars = ["YearBuilt", "YearRemodAdd",
             "GarageYrBlt", "MoSold", "YrSold"]

assert len(set(cont_vars) & set(discrete_vars))==0

discrete_highest_cor_vars = list(set(highest_cor_vars) & set(discrete_vars))
cont_vars = list(set(cont_vars) | set(time_vars))


def load_df(filepath, split_train=False, train_ratio=0.8):
    df = pd.read_csv(filepath)
    if split_train:
        msk = np.random.rand(len(df)) < train_ratio
        df_train = df[msk]
        df_val = df[~msk]
        return (df_train, df_val)
    else:
        return df


def data_conversion(df, le_dict, discrete_vars, cont_vars, stage=None, one_hot=False,
                        use_subfeatures=None, fill_null=False, bad_features=None):
    X, y = [], []
    bad_ids = set()
    feature_names = None
    #test_feat_vals = []
    for ind, row in df.iterrows():
        feats = []
        if feature_names is None:
            feats_names = []
        for feat in discrete_vars:
            if bad_features is not None and feat in bad_features:
                continue
            le = le_dict[feat]
            val = le.transform([row[feat]])
            #print(feat, row[feat], le.get_classes(), val)
            if one_hot:
                n_feats = len(le.get_classes())
                feat_vec = [0] * n_feats
                if le.get_nan_encoding() is None or val[0] != le.get_nan_encoding():
                    feat_vec[val[0]] = 1
                else:
                    feat_vec[val[0]] = np.nan
                    bad_ids.add(ind)
                feats.extend(feat_vec)

                if feature_names is None:
                    feats_names.extend([feat]*len(feat_vec))

            else:
                if le.get_nan_encoding() is None or val[0] != le.get_nan_encoding():
                    feats.append(val[0])
                    # if feat == "OverallQual":
                    #     feat_vals.append(val[0])
                else:
                    feats.append(np.nan)
                    bad_ids.add(ind)
                if feature_names is None:
                    feats_names.append(feat)

        discrete_feats = feats.copy()

        cont_feats = []

        for feat in cont_vars:
            if bad_features is not None and feat in bad_features:
                continue
            try:
                val = float(row[feat])
                #feats.append(val)
                cont_feats.append(val)
            except:
                #feats.append(np.nan)
                cont_feats.append(np.nan)
                bad_ids.add(ind)

            if feature_names is None:
                feats_names.append(feat)

        if use_subfeatures == "discrete":
            feats = discrete_feats
        elif use_subfeatures == "continuous":
            feats = cont_feats
        else:
            feats.extend(cont_feats)

        y.append(row["SalePrice"])

        if use_subfeatures == "discrete":
            X_discrete.append(discrete_feats)
        elif use_subfeatures == "continuous":
            X_cont.append(cont_feats)
        else:
            #assert len(feats) == 78
            X.append(np.array(feats))

        if feature_names is None:
            feature_names = feats_names

    #print("unconverted_cont_feats: ", list(unconverted_cont_feats))
    X = np.array(X)
    y = np.array(y)
    test_ind = np.argwhere(np.isnan(y)).squeeze()

    #print("TEST FEAT stat: ")
    #print(pd.Series(test_feat_vals).describe())

    print("Feature names: ", feature_names, len(feature_names))
    print("There are {} data points with nan features: ".format(len(bad_ids)))

    if fill_null and np.isnan(X).any():
        print("Using simple imputation for all missing values")
        imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        # note: imputation method automatically changes n_features here
        X = imp.fit_transform(X)
        print("imputed X shape", X.shape)
        return X, y, test_ind, feature_names

    else:
        print("original X shape", X.shape)

        return X, y, test_ind, feature_names


def train(clf, df_train, discrete_vars, cont_vars, le_dict, one_hot=False, use_subfeatures=None, transform_target=False):
    X_tr, y_tr = data_conversion(df_train, le_dict, discrete_vars, cont_vars, stage="train", one_hot=one_hot, use_subfeatures=use_subfeatures)
    #normal_fit_1d(y_tr)
    if transform_target:
        #y_tr = label_normalization(y_tr, right_shift=True, center_and_scale=False)
        y_tr = np.log1p(y_tr)

    #normal_fit_1d(y_tr)
    clf = clf.fit(X_tr, y_tr)
    return clf


def predict(clf, X_t, y_t, transform_target=False):
    y_preds = clf.predict(X_t)

    if transform_target:
        y_preds = np.expm1(y_preds)

    for pred, l in zip(y_preds, y_t):
        print(pred, l)

    error = np.sqrt(mean_squared_log_error(y_t, y_preds))

    #cv_error = None
    cv_error = msle_cv_eval(X_t, y_t, clf, transform_target=transform_target)
    return error, cv_error

def squared_log_error(y_gt_i, y_pred_i):
    return (np.log(1+y_gt_i)-np.log(1+y_pred_i))**2

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ### data processing/transformation
    parser.add_argument('--auto_detect_features', default=False, action='store_true')
    parser.add_argument('--auto_imputation', default=False, action='store_true')
    parser.add_argument('--use_onehot', default=False, action='store_true')
    parser.add_argument('--use_selected_features', default=False, action='store_true')
    parser.add_argument('--use_subfeatures', type=str, default=None, help="use discrete or continuous features")
    parser.add_argument('--transform_target', default=False, action='store_true')
    parser.add_argument('--transform_features', default=None, type=str)
    ### model
    parser.add_argument('--parameter_search', default=False, action='store_true')
    parser.add_argument('--estimator', type=str, default="decision_tree")

    ### others
    parser.add_argument('--create_submission', action='store_true', default=False)
    parser.add_argument('--save_estimator', action='store_true', default=False)

    args = parser.parse_args()

    #### load raw data
    fpath_train = os.path.join(DATA_DIR, "train.csv")
    fpath_test = os.path.join(DATA_DIR, "test.csv")

    df_train = load_df(fpath_train, split_train=False)

    df_test = load_df(fpath_test)
    print("---- TRAIN DATA -----")
    print(df_train.head())
    print("---- TEST DATA -----")
    print(df_test.head())

    ##### ---- data processing ---- ####
    # FEATURE CORRELATION
    print("----- Highest correlated features with target `SalePrice` -----")
    #correlation_heatmap(df_train, target='SalePrice', thres=0.5, show=False)

    # FEATURE EXTRACTION/TRANSFORMATION (using all available data)
    joint_df = pd.concat([df_train, df_test], ignore_index=True)

    # drop features that has too many null values
    target = joint_df["SalePrice"]
    joint_df.drop(columns=["SalePrice"])
    bad_features = na_stat(joint_df, drop_bad_features=True)
    joint_df["SalePrice"] = target

    # auto-determine numeric and non-numeric features ??
    if args.auto_detect_features:
        discrete_vars = joint_df.dtypes[joint_df.dtypes != 'float64'].index

        cont_vars = joint_df.dtypes[joint_df.dtypes == 'float64'].index

    print(joint_df.info())
    print("--- Discrete features ---")
    print(discrete_vars)
    print("--- Continuous features ---")
    print(cont_vars)

    # encode discrete features
    df_transformed, le_dict = categorical_feature_extraction(joint_df,
                                                            discrete_vars,
                                                            fill_null=args.auto_imputation,
                                                            stage="train")
    df_transformed.drop("Id", axis=1, inplace=True)


    if not args.auto_imputation:
        # fill null data (if we do this step, data imputation is not needed)
        for feature in cont_vars:
            if feature in df_transformed.columns:
                #print("cont feat", feature)
                df_transformed[feature] = df_transformed[feature].fillna(0)
        for feature in discrete_vars:
            if feature in df_transformed.columns:
                #print("discrete feat", feature)
                df_transformed[feature] = df_transformed[feature].fillna(df_transformed[feature].mode()[0])


    if args.use_selected_features:
        X_all, y_all, test_ind, feature_names = data_conversion(df_transformed, le_dict, discrete_highest_cor_vars,
                            cont_vars, stage="all", one_hot=args.use_onehot,
                            use_subfeatures=None, fill_null=args.auto_imputation, bad_features=bad_features)
    else:
        X_all, y_all, test_ind, feature_names = data_conversion(df_transformed, le_dict, discrete_vars,
                            cont_vars, stage="all", one_hot=args.use_onehot,
                            use_subfeatures=None, fill_null=args.auto_imputation, bad_features=bad_features)

    ## TODO: remove
    X_all_1, y_all_1, test_ind_1, feature_names_1 = data_conversion(df_transformed, le_dict, discrete_vars,
                        cont_vars, stage="all", one_hot=True,
                        use_subfeatures=None, fill_null=args.auto_imputation, bad_features=bad_features)
    X_all_2, y_all_2, test_ind_2, feature_names_2 = data_conversion(df_transformed, le_dict, discrete_vars,
                        cont_vars, stage="all", one_hot=False,
                        use_subfeatures=None, fill_null=args.auto_imputation, bad_features=bad_features)

    print("X: ", X_all.shape, len(feature_names))
    print("X_1: ", X_all_1.shape, len(feature_names_1))
    print("X_2: ", X_all_2.shape, len(feature_names_2))

    if args.transform_features == "skew" or args.transform_features == "skew+scale":
        # skew correction makes individual features more symmetrically distributed
        high_right_skew = []
        high_left_skew = []
        for col_ind, feat in enumerate(feature_names):
            if feat in cont_vars:
                if skew(X_all[:, col_ind]) > 0.5:
                    high_right_skew.append(feat)
                    X_all[:, col_ind] = np.log1p(X_all[:, col_ind])
                elif skew(X_all[:, col_ind]) < -0.5:
                    high_left_skew.append(feat)
                    X_all[:, col_ind] = np.power(X_all[:, col_ind], 3)
        print("--- features with high right skew ---")
        print(high_right_skew)

        print("--- features with high left skew ---")
        print(high_left_skew)


    if args.transform_features == "skew+scale":
        scaler = MinMaxScaler()
        X_all = scaler.fit_transform(X_all)


    ## TODO: remove
    for col_ind, feat in enumerate(feature_names_2):
        if feat in cont_vars:
            if skew(X_all_2[:, col_ind]) > 0.5:
                X_all_2[:, col_ind] = np.log1p(X_all_2[:, col_ind])
            elif skew(X_all_2[:, col_ind]) < -0.5:
                X_all_2[:, col_ind] = np.power(X_all_2[:, col_ind], 3)


    ##
    # # EDA (optional)
    # import seaborn as sns
    # import matplotlib.pyplot as plt
    # #plt.figure(figsize=(20,8))
    #
    # X_new = np.append(X_all, np.expand_dims(y_all, axis=1), axis=1)
    # df_new = pd.DataFrame(X_new, columns=feature_names+["SalePrice"])
    #
    # for feat in df_new.columns:
    #     print(df_new[feat].describe())
    #
    #
    # #print(df_new['OverallQual'])
    # corr = df_new.corr()
    # # print("Corr after transf")
    # # print(corr["SalePrice"]["OverallQual"])
    # highest_corr_features = corr.index[abs(corr["SalePrice"])>0.4]
    # fig, axes = plt.subplots(nrows=int(len(highest_corr_features)/3), ncols=3)
    # for idx, feature in enumerate(highest_corr_features):
    #     #print(feature)
    #     #print(corr["SalePrice"][feature])
    #     #df_new.plot.scatter(x=feature, y='SalePrice', ax=axes[idx])
    #     df_new.boxplot(column=feature, ax=axes[int(idx/3), idx%3])
    #
    # #sns.heatmap(df_new[highest_corr_features].corr(),annot=True,cmap="RdYlGn")
    # #sns.heatmap(corr, vmax=.8, square=True)
    # plt.show()


    # # Final data imputation to remove NaN values
    # if not args.auto_imputation:
    #     imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    #     X_all = imp.fit_transform(X_all)
    #     X_all_1 = imp.fit_transform(X_all_1)
    #     X_all_2 = imp.fit_transform(X_all_2)

    #print("Imputed X shape: ", X_all.shape)

    # split train/val/test
    X_tr = X_all[~test_ind, :]
    X_tr_1 = X_all_1[~test_ind, :] ## to remove
    X_tr_2 = X_all_2[~test_ind, :] ## to remove
    y_tr = y_all[~test_ind]
    msk = np.random.rand(len(y_tr)) < 0.8
    X_train, y_train = X_tr[msk], y_tr[msk]
    X_val, y_val = X_tr[~msk], y_tr[~msk]
    X_val_1 = X_tr_1[~msk] ## to remove
    X_val_2 = X_tr_2[~msk] ## to remove

    X_test_1 = X_all_1[test_ind, :]
    X_test_2 = X_all_2[test_ind, :]
    y_test = y_all[test_ind]

    #assert np.linalg.norm(X_tr - X_tr_1) == 0

    print("Number of train/val/test samples {}/{}/{}".format(len(y_train), len(y_val), len(X_test_1)))


    # Optionally, scale and/or whiten data
    ### --- train and validate
    if args.estimator == "decision_tree":
        clf = tree.DecisionTreeRegressor(random_state=0)
    elif args.estimator == "gbm":
        clf = GradientBoostingRegressor(random_state=0, n_estimators=500, subsample=0.9)
    elif args.estimator == "xgboost":
        clf = xgb.XGBRegressor(random_state=0, nthread=5)
    elif args.estimator == "rf":
        clf = RandomForestRegressor(random_state=0)
    elif args.estimator == "extra_trees":
        clf = ExtraTreesRegressor(random_state=0)
    elif args.estimator == "lasso":
        ana_clf = IsolationForest(random_state=0).fit(X_all_2)
        train_ind = ana_clf.predict(X_train_2)
        X_train, y_train = X_train[train_ind==1], y_train[train_ind==1]
        #scaler = MinMaxScaler()
        # X_train = scaler.fit_transform(X_train)
        # X_val_1_scaled = scaler.transform(X_val_1)
        # X_test_1_scaled = scaler.transform(X_test_1)
        clf = LassoCV(normalize=True, random_state=0, alpha=5)
        #clf = Pipeline([('scaler', MinMaxScaler()), ('lasso', LassoCV(normalize=True, random_state=0))])


    elif args.estimator == "manual_stacking":
        X_train_1 = X_tr_1[msk] # one-hot used as train/test features
        scaler = MinMaxScaler()
        X_train_1_scaled = scaler.fit_transform(X_train_1)
        X_val_1_scaled = scaler.transform(X_val_1)
        X_test_1_scaled = scaler.transform(X_test_1)

        ## outlier detection using non-onehot features
        ana_clf = IsolationForest(random_state=0).fit(X_all_2)
        ana_ind_test = ana_clf.predict(X_test_2)
        ana_ind_val = ana_clf.predict(X_val_2)
        X_train_2 = X_tr_2[msk]
        train_ind = ana_clf.predict(X_train_2)

        ## fit
        est1 = LassoCV(normalize=True, random_state=0).fit(X_train_1_scaled[train_ind==1], np.log1p(y_train[train_ind==1]))
        est2 = GradientBoostingRegressor(random_state=0, n_estimators=500, subsample=0.9).fit(X_train_1_scaled, np.log1p(y_train))

        ## predict
        preds1 = np.expm1(est1.predict(X_val_1_scaled))
        preds2 = np.expm1(est2.predict(X_val_1_scaled))
        fin_preds = []

        ana_val_scores = ana_clf.decision_function(X_val_2)
        ana_val_ind = np.argsort(ana_val_scores)
        sorted_ana_val_scores = ana_val_scores[ana_val_ind]

        # get worst outliers
        ana_val_ind_worst = ana_val_ind[:3]
        print(sorted_ana_val_scores[:3])

        for idx in range(len(y_val)):
            if idx in ana_val_ind_worst:
                fin_preds.append(preds2[idx])
            else:
                fin_preds.append(preds1[idx])

        error1 = np.sqrt(mean_squared_log_error(y_val, preds1))
        error2 = np.sqrt(mean_squared_log_error(y_val, preds2))
        fin_error = np.sqrt(mean_squared_log_error(y_val, fin_preds))
        print("Val error1 {} : error2 {} : error_fin {}".format(error1, error2, fin_error))


        ## generate test submission
        if args.create_submission:
            test_preds1 = np.expm1(est1.predict(X_test_1_scaled))
            test_preds2 = np.expm1(est2.predict(X_test_1_scaled))
            test_preds_fin = []
            n_ana = 0

            ana_scores = ana_clf.decision_function(X_test_2)

            ana_ind = np.argsort(ana_scores)
            sorted_ana_scores = ana_scores[ana_ind]

            # get worst outliers
            ana_ind_worst = ana_ind[:5]
            print(sorted_ana_scores[:5])

            for idx in range(len(X_test_1_scaled)):
                if idx in ana_ind_worst:
                    test_preds_fin.append(test_preds2[idx])
                else:
                    test_preds_fin.append(test_preds1[idx])
            ids = []
            for ind, row in df_test.iterrows():
                ids.append(row["Id"])
            assert len(ids) == len(X_test_2) == len(X_test_1_scaled)
            df = pd.DataFrame({'Id': ids, 'SalePrice': test_preds_fin})
            outfname = "submis-ana-combined.csv"
            df.to_csv(outfname, index=False)

        exit()

    if not args.parameter_search:
        if args.transform_target:
            #y_tr = label_normalization(y_tr, right_shift=True, center_and_scale=False)
            y_train = np.log1p(y_train)

        #normal_fit_1d(y_train)
        clf = clf.fit(X_train, y_train)

        # if args.estimator == "lasso":
        #     print("fitted coeff", clf.lasso.coef_)

    else:
        ### cv-based grid search
        if args.transform_target:
            #y_train = label_normalization(y_train, right_shift=True, center_and_scale=False)
            y_train = np.log1p(y_train)

        if args.estimator == "decision_tree":
            params = {
                "min_samples_split": [2, 4, 8, 16]
            }
        elif args.estimator == "rf":
            params = {
                "n_estimators": [10, 20, 50, 100, 500],
                "min_samples_split": [2, 4, 8, 16]
            }
        elif args.estimator == "lasso":
            params = {
                "alpha": [0.1, 1, 1.5, 2, 5]
            }
        elif args.estimator == "gbm":
            params = {
                "n_estimators": [100, 200, 500],
                "subsample": [0.8, 0.9, 1],
            }
        elif args.estimator == "xgboost":
            params = {
                "n_estimators": [200, 500, 1000, 3000],
                "subsample": [0.7, 0.8, 0.9]
            }

        clf = fit_model_with_parameter_search(clf, params, 5, X_train, y_train, transform_target=args.transform_target)

    #print("Best parameters: ", clf.best_params_)

    error, cv_error = predict(clf, X_val, y_val, transform_target=args.transform_target)

    print("Val rmsle: ", error)
    print("Val 5-fold cv rmsle: ", cv_error)

    if args.save_estimator:
        outfname = "{}_est.pkl".format(args.estimator)
        print("Saving trained estimator {} to file {}".format(args.estimator, outfname))
        with open(outfname, 'wb') as file:
            pickle.dump(clf, file)

    if args.create_submission:
        outfname = convert_preds_to_submis(clf, df_test, X_test, transform_target=args.transform_target)
        print("generated submission {}".format(outfname))
