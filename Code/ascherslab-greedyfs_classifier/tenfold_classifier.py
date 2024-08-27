# ********************************************************
# *   ----------------------------------------------     *
# *       Yoochan Myung - yuchan.m@gmail.com             *
# *   ----------------------------------------------     *
# ********************************************************
import pandas as pd
import numpy as np
import time
import sys
import os
import argparse
import joblib
import re

from sklearnex import patch_sklearn
patch_sklearn()

from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import AdaBoostClassifier,ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import multilabel_confusion_matrix

from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold, StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.metrics import roc_auc_score, f1_score, matthews_corrcoef, balanced_accuracy_score, confusion_matrix

timestr = time.strftime("%Y%m%d_%H%M%S")

def get_kfold_object(strategy, **kwargs):
    """
    Create the appropriate KFold object based on the strategy provided.

    Parameters:
        strategy (str): The name of the KFold strategy (kfold, stratified, group, stratified_group).
        **kwargs: Additional keyword arguments for the specific KFold object.

    Returns:
        KFold object: The corresponding KFold object.
    """
    strategies = {
        'kfold': KFold,
        'groupkfold': GroupKFold,
        'stratified': StratifiedKFold,
        'stratified_group': StratifiedGroupKFold,
    }

    if strategy not in strategies:
        raise ValueError("Invalid strategy. Choose one of: 'kfold', 'stratified', 'groupkfold', 'stratified_group'")

    KFoldClass = strategies[strategy]
    return KFoldClass(**kwargs)

def runML(algorithm, fname, kfold, training_pd, blind_pd, output_dir, target_label, n_cores, random_state, no_scaling, save_model, strategy, groups, greedy=False):
    result_cm = pd.DataFrame() # Confusion Matrix for kCV
    blind_cm = pd.DataFrame() # Confusion Matrix for blind
    outerblind_cv_metrics = pd.DataFrame()
    cv_metrics = pd.DataFrame()
    y_train = np.array(training_pd[target_label])
    if strategy in ['stratified_group']:
        group_info = training_pd[groups].to_numpy()
    training_pd = training_pd.drop('ID', axis=1)
    if strategy in ['stratified_group']:  
        training_pd = training_pd.drop(groups, axis=1)
    X_train = training_pd.drop(target_label, axis=1)

    # Label Encoding
    le = LabelEncoder()
    le.fit(y_train)
    num_of_class = len(le.classes_)
    y_train = le.transform(y_train)
    headers = list(X_train.columns.values)
    headers.sort()
    X_train = X_train[headers]
    if not no_scaling:
        sc = StandardScaler()
        X_train = pd.DataFrame(sc.fit_transform(X_train),columns=headers)
        if save_model:
            joblib.dump(sc, os.path.join(output_dir,'{}_{}_{}_{}CV_transformer.sav'.format(timestr,fname,algorithm,kfold)), compress=True)

    # Kfold cross validation
    predicted_n_actual_pd = pd.DataFrame(columns=['ID', 'predicted', 'actual', 'fold'])
    outerblind_predicted_n_actual_pd = pd.DataFrame(columns=['ID', 'predicted', 'actual'])

    if strategy == 'stratified':
        kfold_obj = get_kfold_object(strategy, n_splits=kfold, shuffle=True, random_state=random_state)
        folds = kfold_obj.split(X_train, y_train)
    elif strategy in ['groupkfold', 'stratified_group']:
        kfold_obj = get_kfold_object(strategy, n_splits=kfold, shuffle=True, random_state=random_state)
        folds = kfold_obj.split(X_train, y_train, groups=group_info)
    else:  # strategy == 'kfold'
        kfold_obj = get_kfold_object(strategy, n_splits=kfold, shuffle=True, random_state=random_state)
        folds = kfold_obj.split(X_train)

    # fold = 1 # for indexing
    # for train, test in kf.split(X_train):
    for fold, (train_idx, val_idx) in enumerate(folds, 1):
        # train and test number(row number) are based on X_train.
        # For example, if '1' from train or test, it would be '1' in X_train

        X_train_cv, X_val_cv, y_train_cv, y_val_cv = X_train.iloc[train_idx],X_train.iloc[val_idx], y_train[train_idx], y_train[val_idx]
        if algorithm == 'GB':
            classifier_cv = GradientBoostingClassifier(n_estimators=300, random_state=1)
            classifier_all = GradientBoostingClassifier(n_estimators=300, random_state=1)

        elif (algorithm == 'XGBOOST'):
            classifier_cv = XGBClassifier(n_estimators=300, random_state=1, n_jobs=n_cores)
            classifier_all = XGBClassifier(n_estimators=300, random_state=1, n_jobs=n_cores)

        elif (algorithm == 'RF'):
            classifier_cv = RandomForestClassifier(n_estimators=300, random_state=1, n_jobs=n_cores)
            classifier_all = RandomForestClassifier(n_estimators=300, random_state=1, n_jobs=n_cores)

        elif (algorithm == 'ExtraTrees'):
            classifier_cv = ExtraTreesClassifier(n_estimators=300, random_state=1, n_jobs=n_cores)
            classifier_all = ExtraTreesClassifier(n_estimators=300, random_state=1, n_jobs=n_cores)

        elif (algorithm == 'GAUSSIAN'):
            classifier_cv = GaussianProcessClassifier(random_state=1)
            classifier_all = GaussianProcessClassifier(random_state=1)

        elif (algorithm == 'ADABOOST'):
            classifier_cv = AdaBoostClassifier(n_estimators=300, random_state=1)
            classifier_all = AdaBoostClassifier(n_estimators=300, random_state=1)

        elif (algorithm == 'KNN'):
            classifier_cv = KNeighborsClassifier(n_neighbors=3, n_jobs=n_cores)
            classifier_all = KNeighborsClassifier(n_neighbors=3, n_jobs=n_cores)

        elif (algorithm == 'SVC'):
            classifier_cv = svm.SVC(kernel='rbf', probability=True)
            classifier_all = svm.SVC(kernel='rbf', probability=True)

        elif (algorithm == 'MLP'):
            classifier_cv = MLPClassifier(random_state=1)
            classifier_all = MLPClassifier(random_state=1)

        elif (algorithm == 'DecisionTree'):
            classifier_cv = DecisionTreeClassifier(random_state=1)
            classifier_all = DecisionTreeClassifier(random_state=1)

        elif (algorithm == 'LG'):
            classifier_cv = LogisticRegression(random_state=1,multi_class='ovr')
            classifier_all = LogisticRegression(random_state=1,multi_class='ovr')

        classifier_cv.fit(X_train_cv, y_train_cv)
        try:
            temp_prediction = classifier_cv.predict(X_val_cv)
        except ValueError:
            raise AssertionError (f"Given FOLD NUMBER {kfold} fails to return correct train and validation indices. Please reduce it.")
        temp_proba = pd.DataFrame(classifier_cv.predict_proba(X_val_cv))
        temp_proba = temp_proba.rename(columns=lambda x: re.sub('^','proba_',str(x)))

        _temp_pd = pd.DataFrame({'ID':val_idx, 'actual':y_val_cv, 'predicted' : temp_prediction, 'fold':fold})
        _temp_pd = pd.concat([_temp_pd,temp_proba],axis=1)
        predicted_n_actual_pd = pd.concat([predicted_n_actual_pd, _temp_pd],ignore_index=True, sort=True)
        fold += 1

    predicted_n_actual_pd.sort_values(by='ID', inplace=True)
    try :
        column_list = predicted_n_actual_pd.columns.tolist()
        proba_columns = [name for name in column_list if 'proba_'in name]
        if num_of_class > 2:
            roc_auc = round(roc_auc_score(predicted_n_actual_pd['actual'].to_list(),predicted_n_actual_pd[proba_columns].to_numpy(), multi_class='ovr'),3)
        else:
            roc_auc = round(roc_auc_score(predicted_n_actual_pd['actual'].to_list(),predicted_n_actual_pd[proba_columns].iloc[:,1]),3)

    except ValueError:
        roc_auc = 0.0

    if num_of_class > 2:
        f1 = round(f1_score(predicted_n_actual_pd['actual'].to_list(),predicted_n_actual_pd['predicted'].to_list(),average='micro'),3)
    else:
        f1 = round(f1_score(predicted_n_actual_pd['actual'].to_list(),predicted_n_actual_pd['predicted'].to_list()),3)

    matthews = round(matthews_corrcoef(predicted_n_actual_pd['actual'].to_list(),predicted_n_actual_pd['predicted'].to_list()),3)
    balanced_accuracy = round(balanced_accuracy_score(predicted_n_actual_pd['actual'].to_list(),predicted_n_actual_pd['predicted'].to_list()),3)

    try:
        if num_of_class > 2:
            mcm = multilabel_confusion_matrix(predicted_n_actual_pd['actual'].to_list(), predicted_n_actual_pd['predicted'].to_list())
            tn = mcm[:,0, 0]
            tp = mcm[:,1, 1]
            fn = mcm[:,1, 0]
            fp = mcm[:,0, 1]
            result_cm = pd.DataFrame(np.column_stack((tn,tp,fn,fp)),columns=['tn','tp','fn','fp'],index=le.classes_)
        else:
            tn, fp, fn, tp = confusion_matrix(predicted_n_actual_pd['actual'].to_list(), predicted_n_actual_pd['predicted'].to_list()).ravel()
            result_cm = pd.DataFrame(np.column_stack((tn,tp,fn,fp)),columns=['tn','tp','fn','fp'])

    except:
        tn, fp, fn, tp = 0

    cv_metrics = pd.concat([cv_metrics,pd.DataFrame(np.column_stack(['cv',roc_auc, matthews,balanced_accuracy, f1 ]), columns=['type','roc_auc','matthew','bacc','f1'])], ignore_index=True, sort=True)

    # # To get ML model in SAV file.
    if not greedy:
        if save_model:
            model = classifier_all.fit(X_train,y_train)
            model_filename = os.path.join(output_dir, f'{timestr}_{fname}_{algorithm}_{strategy}_{kfold}_model.sav')
            joblib.dump(model, model_filename)

    if not isinstance(blind_pd,bool):
        classifier_all.fit(X_train,y_train)
        blind_pd_ID = pd.DataFrame(blind_pd['ID'])
        X_blind = blind_pd[headers]
        y_blind = np.array(blind_pd[target_label])
        y_blind = le.transform(y_blind)
        if not no_scaling:
            X_blind = pd.DataFrame(sc.transform(X_blind),columns=headers)

        ## outerblind-Test
        prediction = classifier_all.predict(X_blind)        
        proba = pd.DataFrame(classifier_all.predict_proba(X_blind))
        proba = proba.rename(columns=lambda x: re.sub('^','proba_',str(x)))

        _temp_pd = pd.DataFrame({'ID':blind_pd_ID['ID'].to_list(), 'actual':y_blind, 'predicted' : prediction})
        _temp_pd = pd.concat([_temp_pd,proba],axis=1)

        outerblind_predicted_n_actual_pd = pd.concat([outerblind_predicted_n_actual_pd, _temp_pd],ignore_index=True, sort=True)

        outerblind_matthews_corrcoef = round(matthews_corrcoef(y_blind, prediction),3)
        outerblind_balanced_accuracy_score = round(balanced_accuracy_score(y_blind, prediction),3)

        try :
            column_list = outerblind_predicted_n_actual_pd.columns.tolist()
            proba_columns = [name for name in column_list if 'proba_'in name]
            if num_of_class > 2:
                outerblind_roc_auc_score = round(roc_auc_score(outerblind_predicted_n_actual_pd['actual'].to_list(), outerblind_predicted_n_actual_pd[proba_columns].to_numpy(),multi_class='ovr'),3)
            else:
                outerblind_roc_auc_score = round(roc_auc_score(outerblind_predicted_n_actual_pd['actual'].to_list(), outerblind_predicted_n_actual_pd[proba_columns].iloc[:,1]),3)

        except ValueError:
            outerblind_roc_auc_score = 0.0

        if num_of_class > 2:
            outerblind_f1_score = round(f1_score(y_blind, prediction,average='micro'),3)
        else:
            outerblind_f1_score = round(f1_score(y_blind, prediction),3)

        try:
            if num_of_class > 2:
                outerblind_mcm = multilabel_confusion_matrix(y_blind, prediction)
                outerblind_tn = outerblind_mcm[:,0, 0]
                outerblind_tp = outerblind_mcm[:,1, 1]
                outerblind_fn = outerblind_mcm[:,1, 0]
                outerblind_fp = outerblind_mcm[:,0, 1]
                blind_cm = pd.DataFrame(np.column_stack((outerblind_tn,outerblind_tp,outerblind_fn,outerblind_fp)),columns=['tn','tp','fn','fp'],index=le.classes_)

            else:
                outerblind_tn, outerblind_fp, outerblind_fn, outerblind_tp = confusion_matrix(y_blind, prediction).ravel()
                blind_cm = pd.DataFrame(np.column_stack((outerblind_tn,outerblind_tp,outerblind_fn,outerblind_fp)),columns=['tn','tp','fn','fp'])

        except:
            outerblind_tn,outerblind_fp,outerblind_fn,outerblind_tp = 0
            blind_cm = pd.DataFrame(np.column_stack((0,0,0,0)),columns=['tn','tp','fn','fp'])

        outerblind_cv_metrics = pd.concat([outerblind_cv_metrics,pd.DataFrame(np.column_stack(['blind-test',outerblind_roc_auc_score, outerblind_matthews_corrcoef,
            outerblind_balanced_accuracy_score, outerblind_f1_score]),
             columns=['type','roc_auc','matthew','bacc','f1'])], ignore_index=True, sort=True)
        outerblind_cv_metrics.set_index([['blind-test']*len(outerblind_cv_metrics)], inplace=True)

    cv_metrics = cv_metrics.round(3)
    outerblind_cv_metrics = outerblind_cv_metrics.round(3)

    predicted_n_actual_pd['predicted'] = le.inverse_transform(predicted_n_actual_pd['predicted'].to_list())
    predicted_n_actual_pd['actual'] = le.inverse_transform(predicted_n_actual_pd['actual'].to_list())
    fname_predicted_n_actual_pd = os.path.join(output_dir,f'{timestr}_{fname}_{algorithm}_{str(random_state)}_{strategy}_{kfold}_{target_label}_training_preds.csv')
    fname_result_cm = os.path.join(output_dir,f'{timestr}_{fname}_{algorithm}_{str(random_state)}_{strategy}_{kfold}_{target_label}_training_cm.csv')
    predicted_n_actual_pd['ID'] = predicted_n_actual_pd['ID'] + 1
    predicted_n_actual_pd = predicted_n_actual_pd.sort_values(by=['ID'])

    if not greedy:
        predicted_n_actual_pd.to_csv(fname_predicted_n_actual_pd,index=False)
        result_cm.to_csv(fname_result_cm, index=False)

    outerblind_predicted_n_actual_pd['predicted'] = le.inverse_transform(outerblind_predicted_n_actual_pd['predicted'].to_list())
    outerblind_predicted_n_actual_pd['actual'] = le.inverse_transform(outerblind_predicted_n_actual_pd['actual'].to_list())

    if (not isinstance(blind_pd,bool) and not greedy):
        fname_outerblind_predicted_n_actual_pd = os.path.join(output_dir,f'{timestr}_{fname}_{algorithm}_{strategy}_{kfold}_{target_label}_blindtest_preds.csv')
        fname_blind_cm = os.path.join(output_dir,f'{timestr}_{fname}_{algorithm}_{strategy}_{kfold}_{target_label}_blindtest_cm.csv')
        outerblind_predicted_n_actual_pd.to_csv(fname_outerblind_predicted_n_actual_pd,index=False)
        blind_cm.to_csv(fname_blind_cm, index=False)
        return cv_metrics, outerblind_cv_metrics
    else:
        return cv_metrics, outerblind_cv_metrics


def main(args):
    # REQUIRED
    algorithm = args.algorithm
    input_csv = args.input_csv
    target_label = args.target_label

    # OPTIONAL
    blind_csv = args.blind_csv
    n_cores = args.n_cores
    num_shuffle = args.num_shuffle
    output_dir = args.output_dir
    kfold = args.kfold

    result_ML = pd.DataFrame()
    blind_result_ML = pd.DataFrame()

    fname = os.path.split(input_csv.name)[1]
    original_dataset = pd.read_csv(input_csv, sep=',', quotechar='\"', header=0)
    print("filename :", fname)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if blind_csv:
        blind_pd = pd.read_csv(blind_csv, header=0)
    else:
        blind_pd = False

    for random_state in range(1, int(num_shuffle) + 1):
        ML_kwargs = {'algorithm': algorithm,'fname': fname, 'kfold': args.kfold, 'training_pd': original_dataset, 'blind_pd': blind_pd,\
                'output_dir': output_dir, 'target_label': target_label, 'n_cores': n_cores, 'random_state': random_state,\
                'no_scaling': args.no_scaling, 'save_model': args.save_model, 'groups': args.groups, 'strategy': args.strategy
        }
        each_result_ML, each_blind_result_ML = runML(**ML_kwargs)
        result_ML = pd.concat([result_ML,each_result_ML], ignore_index=False)  # for general results
        blind_result_ML = pd.concat([blind_result_ML, each_blind_result_ML], ignore_index=False)

    result_ML = result_ML.reset_index(drop=True)
    result_ML.index += 1

    fname_result_ML = os.path.join(output_dir,f'{timestr}_{fname}_{algorithm}_{args.strategy}_{kfold}_{target_label}_training_result.csv')
    result_ML.to_csv(fname_result_ML,index_label='run_ID')

    if not isinstance(blind_pd,bool):
        blind_result_ML = blind_result_ML.reset_index(drop=True)
        blind_result_ML.index += 1

        fname_blind_result_ML = os.path.join(output_dir,f'{timestr}_{fname}_{algorithm}_{args.strategy}_{kfold}_{target_label}_blindtest_result.csv')
        blind_result_ML.to_csv(fname_blind_result_ML,index_label='run_ID')
        print(result_ML)
        print(blind_result_ML)
    else:
        print(result_ML)

if __name__ == "__main__":
    output_dir = os.path.dirname(os.path.abspath(__file__))
 
    # REQUIRED
    parser = argparse.ArgumentParser(description='ex) python tenfold_classifier.py ExtraTrees train.csv age -num_shuffle 1 -blind_csv test.csv')
    parser.add_argument("algorithm", help="Choose algorithm [GB,XGBOOST,RF,ExtraTrees,GAUSSIAN,ADABOOST,KNN,SVC,MLP,DecisionTree,LG]")
    parser.add_argument("input_csv", help="Choose input CSV(comma-separated values) format file",
                        type=argparse.FileType('rt'))
    parser.add_argument("target_label", help="Type the name of label")

    # OPTIONAL
    parser.add_argument("-num_shuffle", help="Choose the number of shuffling", type=int, default=1)
    parser.add_argument("-output_dir", help="Choose folder to save result(CSV)",default=output_dir)
    parser.add_argument("-n_cores", help="Choose the number of cores to use", type=int, default=4)
    parser.add_argument("-no-scaling", help="Disable StandardScaler",action='store_true')    
    parser.add_argument("-blind_csv", help="Choose input CSV(comma-separated values) format file", default=False)
    parser.add_argument("-save_model", help="Enable saving model, order of feature and pipeline",action='store_true')    
    parser.add_argument("-kfold", help="Choose CV Fold number", type=int, default=10)    
    parser.add_argument("-groups", help="Choose group column name", type=str, default=None)
    parser.add_argument("-strategy", help="Choose CV strategy", type=str, choices=['kfold', 'stratified', 'stratified_group'], default='kfold')

    args = parser.parse_args()
    with open(f'{timestr}_CV_args.log', 'w') as f:
        f.write(str(args.__dict__))
    main(args)
