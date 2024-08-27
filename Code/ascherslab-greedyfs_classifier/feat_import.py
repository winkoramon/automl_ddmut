# ********************************************************
# *               University of Melbourne                *
# *   ----------------------------------------------     *
# * Yoochan Myung - ymyung@student.unimelb.edu.au        *
# * Last modification :: 30/04/2020                      *
# *   ----------------------------------------------     *
# ********************************************************
import pandas as pd
import numpy as np
import scipy as sp
import time
import sys
import os
import argparse
from math import sqrt
from scipy import stats

from sklearn.model_selection import cross_validate
from sklearn.metrics import recall_score

from xgboost.sklearn import XGBClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm # for SVC

from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing

from sklearn.metrics import roc_auc_score, f1_score, matthews_corrcoef, accuracy_score, balanced_accuracy_score, confusion_matrix, classification_report


timestr = time.strftime("%Y%m%d_%H%M%S")

def runML(algorithm, fname, training_set, outerblind_set, output_result_dir, label_name, n_cores, random_state):
    result_ML = pd.DataFrame()
    result_importance = pd.DataFrame()

    training_ID = pd.DataFrame(training_set['ID'])
    train_label = np.array(training_set[label_name])
    training_set = training_set.drop('ID', axis=1)
    training_features = training_set.drop(label_name, axis=1)

    le = preprocessing.LabelEncoder()
    le.fit(train_label)
    train_label = le.transform(train_label)
    headers = list(training_features.columns.values)

    cv_metrics = pd.DataFrame()

    # 10-fold cross validation
    predicted_n_actual_pd = pd.DataFrame(columns=['ID', 'predicted', 'actual', 'fold'])
    outerblind_predicted_n_actual_pd = pd.DataFrame(columns=['ID', 'predicted', 'actual'])

    kf = KFold(n_splits=10, shuffle=True, random_state=random_state)
    fold = 1 # for indexing

    for train, test in kf.split(training_features):
        # train and test number(row number) are based on training_features.
        # For example, if '1' from train or test, it would be '1' in training_features

        train_cv_features, test_cv_features, train_cv_label, test_cv_label = training_features.iloc[train], \
                                                                             training_features.iloc[test], train_label[
                                                                                 train], train_label[test]

        if algorithm == 'GB':
            temp_classifier = GradientBoostingClassifier(n_estimators=300, random_state=1)

        elif (algorithm == 'XGBOOST'):
            temp_classifier = XGBClassifier(n_estimators=300, random_state=1, n_jobs=n_cores)

        elif (algorithm == 'RF'):
            temp_classifier = RandomForestClassifier(n_estimators=300, random_state=1, n_jobs=n_cores)

        elif (algorithm == 'M5P'):
            temp_classifier = ExtraTreesClassifier(n_estimators=300, random_state=1, n_jobs=n_cores)

        elif (algorithm == 'GAUSSIAN'):
            temp_classifier = GaussianProcessClassifier(random_state=1)

        elif (algorithm == 'ADABOOST'):
            temp_classifier = AdaBoostClassifier(n_estimators=300, random_state=1)

        elif (algorithm == 'KNN'):
            temp_classifier = KNeighborsClassifier(n_neighbors=3, n_jobs=n_cores)

        elif (algorithm == 'SVC'):
            temp_classifier = svm.SVC(kernel='rbf')

        elif (algorithm == 'NEURAL'):
            temp_classifier = MLPClassifier(random_state=1)


        temp_classifier.fit(train_cv_features, train_cv_label)
        temp_prediction = temp_classifier.predict(test_cv_features)

        if algorithm in ["GB","XGBOOST","RF","M5P"]:
            temp_classifier.fit(train_cv_features,train_cv_label)
            feature_importance = temp_classifier.feature_importances_
            feature_importance = 100 * (feature_importance / feature_importance.max())
            sorted_idx = np.argsort(feature_importance)
            sorted_header = np.asarray(headers)[sorted_idx]
            sorted_importance = feature_importance[sorted_idx]
            result_importance = pd.concat([result_importance,pd.DataFrame({'feature':sorted_header,'relative_importance_{}'.format(fold):sorted_importance.round(2)}).set_index('feature')],axis=1,sort=False)

        predicted_n_actual_pd = predicted_n_actual_pd.append(pd.DataFrame({'ID':test, 'actual':test_cv_label, 'predicted' : temp_prediction, 'fold':fold}),ignore_index=True, sort=True)

        ## Metrics for internal 10-fold 
        # try : 
        #     temp_roc_auc_score = round(roc_auc_score(temp_prediction, test_cv_label),3)
        
        # except ValueError:
        #     temp_roc_auc_score = 0.0

        # temp_matthews_corrcoef = round(matthews_corrcoef(temp_prediction, test_cv_label),3)
        # temp_balanced_accuracy_score = round(balanced_accuracy_score(temp_prediction, test_cv_label),3)
        # temp_f1_score = round(f1_score(temp_prediction, test_cv_label),3)

        # try:
        #     temp_tn, temp_fp, temp_fn, temp_tp = confusion_matrix(temp_prediction, test_cv_label).ravel()

        # except:
        #     temp_tn = 0
        #     temp_fp = 0
        #     temp_fn = 0
        #     temp_tp = 0

        # cv_metrics = cv_metrics.append(pd.DataFrame(np.column_stack([temp_roc_auc_score, temp_matthews_corrcoef,
        #     temp_balanced_accuracy_score, temp_f1_score, temp_tn, temp_fp, temp_fn, temp_tp, fold]), 
        # columns=['roc_auc','matthew','bacc','f1','TN','FP','FN','TP','fold']), ignore_index=True)
        fold += 1

    try : 
        roc_auc = round(roc_auc_score(predicted_n_actual_pd['actual'].to_list(),predicted_n_actual_pd['predicted'].to_list()),3)
    
    except ValueError:
        roc_auc = 0.0

    matthews = round(matthews_corrcoef(predicted_n_actual_pd['actual'].to_list(),predicted_n_actual_pd['predicted'].to_list()),3)
    balanced_accuracy = round(balanced_accuracy_score(predicted_n_actual_pd['actual'].to_list(),predicted_n_actual_pd['predicted'].to_list()),3)
    f1 = round(f1_score(predicted_n_actual_pd['actual'].to_list(),predicted_n_actual_pd['predicted'].to_list()),3)

    try:
        tn, fp, fn, tp = confusion_matrix(predicted_n_actual_pd['actual'].to_list(), predicted_n_actual_pd['predicted'].to_list()).ravel()

    except:
        tn, fp, fn, tp = 0

    cv_metrics = cv_metrics.append(pd.DataFrame(np.column_stack(['cv',roc_auc, matthews,\
        balanced_accuracy, f1, tn, fp, fn, tp]),\
        columns=['type','roc_auc','matthew','bacc','f1','TN','FP','FN','TP']), ignore_index=True, sort=True)

    if outerblind_set is not 'False':

        if algorithm == 'GB':
            classifier = GradientBoostingClassifier(n_estimators=300, random_state=1)

        elif (algorithm == 'XGBOOST'):
            classifier = XGBClassifier(n_estimators=300, random_state=1, n_jobs=n_cores)

        elif (algorithm == 'RF'):
            classifier = RandomForestClassifier(n_estimators=300, random_state=1, n_jobs=n_cores)

        elif (algorithm == 'M5P'):
            classifier = ExtraTreesClassifier(n_estimators=300, random_state=1, n_jobs=n_cores)

        elif (algorithm == 'GAUSSIAN'):
            classifier = GaussianProcessClassifier(random_state=1)

        elif (algorithm == 'ADABOOST'):
            classifier = AdaBoostClassifier(n_estimators=300, random_state=1)

        elif (algorithm == 'KNN'):
            classifier = KNeighborsClassifier(n_neighbors=3, n_jobs=n_cores)

        elif (algorithm == 'SVC'):
            classifier = svm.SVC(kernel='rbf')

        elif (algorithm == 'NEURAL'):
            classifier = MLPClassifier(random_state=1)

        else:
            print("Algorithm Selection ERROR!!")
            sys.exit()
        
        outerblind_cv_metrics = pd.DataFrame()
        outerblind_set_ID = pd.DataFrame(outerblind_set['ID'])
        outerblind_label = np.array(outerblind_set[label_name])
        outerblind_features = outerblind_set[headers]
        outerblind_label = le.transform(outerblind_label)
        headers = list(training_features.columns.values)


        ## outerblind-Test
        classifier.fit(training_features,train_label)
        prediction = classifier.predict(outerblind_features)

        outerblind_predicted_n_actual_pd = outerblind_predicted_n_actual_pd.append(pd.DataFrame({'ID':outerblind_set_ID['ID'].to_list(), 'actual':outerblind_label, 'predicted' : prediction}),ignore_index=True, sort=True)

        try :
            outerblind_roc_auc_score = round(roc_auc_score(outerblind_label, prediction),3)

        except ValueError:
            outerblind_roc_auc_score = 0.0

        outerblind_matthews_corrcoef = round(matthews_corrcoef(outerblind_label, prediction),3)
        outerblind_balanced_accuracy_score = round(balanced_accuracy_score(outerblind_label, prediction),3)
        outerblind_f1_score = round(f1_score(outerblind_label, prediction),3)
        try:
            outerblind_tn, outerblind_fp, outerblind_fn, outerblind_tp = confusion_matrix(outerblind_label, prediction ).ravel()

        except:
            outerblind_tn,outerblind_fp,outerblind_fn,outerblind_tp = 0

        outerblind_cv_metrics = outerblind_cv_metrics.append(pd.DataFrame(np.column_stack(['blind-test',outerblind_roc_auc_score, outerblind_matthews_corrcoef,
            outerblind_balanced_accuracy_score, outerblind_f1_score, outerblind_tn, outerblind_fp, outerblind_fn, outerblind_tp]),
             columns=['type','roc_auc','matthew','bacc','f1','TN','FP','FN','TP']), ignore_index=True, sort=True)
        outerblind_cv_metrics.set_index([['blind-test']*len(outerblind_cv_metrics)], inplace=True)
        cv_metrics = pd.concat([cv_metrics, outerblind_cv_metrics], sort=True)
    
    cv_metrics = cv_metrics.round(3)
    cv_metrics = cv_metrics.astype({'TP':'int64','TN':'int64','FP':'int64','FN':'int64'})
    cv_metrics = cv_metrics[['type','matthew','f1','bacc','roc_auc','TP','TN','FP','FN']]
    

    if algorithm == 'GB':
        classifier = GradientBoostingClassifier(n_estimators=300, random_state=1)

    elif (algorithm == 'XGBOOST'):
        classifier = XGBClassifier(n_estimators=300, random_state=1, n_jobs=n_cores)

    elif (algorithm == 'RF'):
        classifier = RandomForestClassifier(n_estimators=300, random_state=1, n_jobs=n_cores)

    elif (algorithm == 'M5P'):
        classifier = ExtraTreesClassifier(n_estimators=300, random_state=1, n_jobs=n_cores)

    else:
        print("Algorithm Selection ERROR!!")
        sys.exit()

    if algorithm in ["GB","XGBOOST","RF","M5P"]:
        classifier.fit(training_features,train_label)
        feature_importance = classifier.feature_importances_
        feature_importance = 100 * (feature_importance / feature_importance.max())
        sorted_idx = np.argsort(feature_importance)
        sorted_header = np.asarray(headers)[sorted_idx]
        sorted_importance = feature_importance[sorted_idx]
        result_importance = pd.concat([result_importance,pd.DataFrame({'feature':sorted_header,'relative_importance_overall':sorted_importance.round(2)}).set_index('feature')],axis=1,sort=False)
    
    predicted_n_actual_pd['predicted'] = le.inverse_transform(predicted_n_actual_pd['predicted'].to_list())
    predicted_n_actual_pd['actual'] = le.inverse_transform(predicted_n_actual_pd['actual'].to_list())
    fname_predicted_n_actual_pd = os.path.join(output_result_dir,'cv_{}_predited_data.csv'.format(algorithm))
    predicted_n_actual_pd['ID'] = predicted_n_actual_pd['ID'] + 1
    predicted_n_actual_pd = predicted_n_actual_pd.sort_values(by=['ID'])
    predicted_n_actual_pd.to_csv(fname_predicted_n_actual_pd,index=False)

    outerblind_predicted_n_actual_pd['predicted'] = le.inverse_transform(outerblind_predicted_n_actual_pd['predicted'].to_list())
    outerblind_predicted_n_actual_pd['actual'] = le.inverse_transform(outerblind_predicted_n_actual_pd['actual'].to_list())
    
    fname_outerblind_predicted_n_actual_pd = os.path.join(output_result_dir,'blindtest_{}_predicted_data.csv'.format(algorithm))
    fname_result_importance = os.path.join(output_result_dir,'training_feature_importance_{}.csv'.format(algorithm))
    outerblind_predicted_n_actual_pd.to_csv(fname_outerblind_predicted_n_actual_pd,index=False)
    result_importance.to_csv(fname_result_importance,index=False)
    print(result_importance)
    return cv_metrics
    

def main(algorithm, input_csv, outerblindtest_csv, output_result_dir, label_name, n_cores, num_shuffle):
    fname = os.path.split(input_csv.name)[1]
    original_dataset = pd.read_csv(input_csv, sep=',', quotechar='\"', header=0)
    
    print("filename :", fname)

    result_ML = pd.DataFrame()
    result_of_blinds = pd.DataFrame()
    
    if outerblindtest_csv != 'False':
        outerblindtest_set = pd.read_csv(outerblindtest_csv, header=0)

    else:
        outerblindtest_set = "False"

    if original_dataset.columns[0] != 'ID':
        print("'ID' column should be given as 1st column.")
        sys.exit()

    for each in range(1, int(num_shuffle) + 1):

        each_result_ML = runML(algorithm, fname, original_dataset, outerblindtest_set, output_result_dir, label_name,
                               n_cores, each)
        
        result_ML = result_ML.append([each_result_ML], ignore_index=False)  # for general results

    result_ML = result_ML.reset_index(drop=True)
    result_ML.index += 1

    fname_result_ML = os.path.join(output_result_dir,'10CV_{}_result.csv'.format(algorithm))
    result_ML.to_csv(fname_result_ML,index=False)
 
    return result_ML


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ex) python tenfold_classifier.py M5P input.csv . dG 16 10 -outerblindtest_csv outertest.csv')
    parser.add_argument("algorithm", help="Choose algorithm between RF,GB,XGBOOST and M5P")
    parser.add_argument("input_csv", help="Choose input CSV(comma-separated values) format file",
                        type=argparse.FileType('rt'))
    parser.add_argument("output_result_dir", help="Choose folder to save result(CSV)")
    parser.add_argument("label_name", help="Type the name of label")
    parser.add_argument("n_cores", help="Choose the number of cores to use", type=int)
    parser.add_argument("num_shuffle", help="Choose the number of shuffling", type=int)
    parser.add_argument("-outerblind", help="Choose input CSV(comma-separated values) format file",
                        default='False')    # optional
    
    args = parser.parse_args()
    # required
    algorithm = args.algorithm
    input_csv = args.input_csv
    output_result_dir = args.output_result_dir
    label_name = args.label_name
    n_cores = args.n_cores
    num_shuffle = args.num_shuffle
    # optional
    outerblindtest_csv = args.outerblind

    if not os.path.exists(output_result_dir):
        os.makedirs(output_result_dir)

    print(main(algorithm, input_csv, outerblindtest_csv, output_result_dir, label_name, n_cores, num_shuffle))
