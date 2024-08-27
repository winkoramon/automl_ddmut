# ********************************************************
# *   ----------------------------------------------     *
# *       Yoochan Myung - yuchan.m@gmail.com             *
# *   ----------------------------------------------     *
# ********************************************************
import pandas as pd
import numpy as np
import argparse
import time
import os
import re

from multiprocessing import Pool
from tqdm import tqdm

import collections
from tenfold_classifier import runML

timestr = time.strftime("%Y%m%d_%H%M%S")

def forwardGreedyFeature(args):
    # REQUIRED
    algorithm = args.algorithm
    training_csv = args.train_csv
    target_label = args.target_label

    # OPTIONAL
    n_cores = args.n_cores
    metric = args.metric    
    cutoff = args.cutoff
    blind_csv = args.blind_csv
    output_dir = args.output_dir
    no_scaling = args.no_scaling
    mute = args.mute
    resume_csv = args.resume_csv
    soft_cutoff = args.soft_cutoff
    groups = args.groups
    strategy = args.strategy
    max_feats = args.max_feats

    if blind_csv:
        blind_pd = pd.read_csv(blind_csv, quotechar='\"', header=0)
    else:
        blind_pd = False

    input_filename = os.path.split(training_csv)[1]
    
    ## DataFrame 
    X_train = pd.read_csv(training_csv, quotechar='\"', header=0)
    original_feature_list = X_train.columns.to_list()
    original_feature_list.remove('ID')
    if strategy in ['stratified_group']:  
        original_feature_list.remove(groups)
    original_feature_list.remove(target_label)
    
    # Init variables
    selected_feature_list = list()
    total_feature_pd = pd.DataFrame()
    score_book = pd.DataFrame(columns=['training','blind-test','diff'])
    stop_signal = False
    greedy_feature_num = 1
    running_counts = 1

    # Resuming
    if not isinstance(resume_csv,bool):
        resume_pd = pd.read_csv(resume_csv, quotechar='\"', header=0, index_col=[0])
        resume_feature_list = resume_pd['feature'].to_list()

        original_feature_list = [feature for feature in original_feature_list if feature not in resume_feature_list]
        selected_feature_list = resume_feature_list
        total_feature_pd = resume_pd.copy()

    with tqdm(total=int(((len(original_feature_list)*(len(original_feature_list)+1))*0.5)), disable=mute) as pbar:
        if mute:
            print("You muted the progress bar.")
            print("The number of calculations is {}".format(int(((len(original_feature_list)*(len(original_feature_list)+1))*0.5))))

        while stop_signal != True:
            remaining_feature_list = list(set(original_feature_list) - set(selected_feature_list))
            score_book = pd.DataFrame(columns=['training','blind-test','diff']) # emptying

            for _feature in remaining_feature_list:
                temp_feature_list = ['ID',  _feature, target_label] + selected_feature_list # new column + selected feature(s)
                if strategy in ['stratified_group']:
                    temp_feature_list = ['ID', groups, _feature, target_label] + selected_feature_list # new column + selected feature(s)
                temp_feature_list.sort()
                temp_X_train_pd = X_train[temp_feature_list] # training X based on the above column list

                # get 10CV result
                ML_kwargs = {'algorithm': algorithm,'fname': input_filename, 'kfold': args.kfold, 'training_pd': temp_X_train_pd, 'blind_pd': blind_pd,\
                        'output_dir': output_dir, 'target_label': target_label, 'n_cores': n_cores, 'random_state': 1,\
                        'no_scaling': args.no_scaling, 'groups': args.groups, 'strategy': args.strategy, 'save_model': False, 'greedy': True             
                }
                # train_metrics, blindtest_metrics = runML(algorithm, temp_X_train_pd, False, output_dir, target_label, n_cores, 1, no_scaling, greedy=True)
                train_metrics, blindtest_metrics = runML(**ML_kwargs)

                # CV only
                if isinstance(blind_pd,bool):
                    score_book.loc[_feature,'training'] = float(train_metrics[metric].loc[0])
                
                # CV wth blind-test
                else:
                    # get 10CV result
                    train_metrics = train_metrics[metric]
                    blindtest_metrics = blindtest_metrics[metric]

                    # parsing 10CV metrics
                    temp_training_score = round(float(train_metrics.loc[0]),3)
                    temp_blindtest_score = round(float(blindtest_metrics.loc['blind-test']),3)
                    temp_diff_score = abs(round(temp_training_score - temp_blindtest_score,3))

                    # recoring metrics on Dataframe
                    score_book.loc[_feature,'training'] = temp_training_score
                    score_book.loc[_feature,'blind-test'] = temp_blindtest_score
                    score_book.loc[_feature,'diff'] = temp_diff_score
                    score_book.sort_values(by=['training'],inplace=True, ascending=False)

                ## time cost calculation
                pbar.set_description('Feature:{}'.format(_feature))
                pbar.update()
                running_counts +=1
            

            selected_feature_name = str()
            selected_feature_training_score = float()
            selected_feature_blindtest_score = float()
            # selecting criteria
            if isinstance(blind_pd,bool): # only training data is given
                # if there is more than one feature that has same correlation value, then it has to been ordered alphabetically and selected.
                _sorted_score_book = score_book.sort_index(key=lambda x: x.str.lower())
                _sorted_score_book = _sorted_score_book.sort_values(by='training', ascending=False).iloc[0]

                selected_feature_name = _sorted_score_book.name
                selected_feature_training_score = _sorted_score_book['training']
                total_feature_pd = pd.concat([total_feature_pd,pd.DataFrame({'feature':selected_feature_name,'training':selected_feature_training_score},index=[0])],ignore_index=True)

            else:   # when both training and blind-test are given
                _sorted_score_book = score_book.query('diff <= 0.1')

                if len(_sorted_score_book) == 0:
                    if soft_cutoff:
                        print("No features met the cutoff criterion {}, but Soft cutoff is allowed".format(cutoff))
                        _sorted_score_book = score_book # this will allows this to keep going through even though diff > 0.1
                    else:
                        print("No features met the cutoff criterion {}".format(cutoff))
                        break

                _sorted_score_book = _sorted_score_book.sort_index(key=lambda x: x.str.lower())
                _sorted_score_book = _sorted_score_book.sort_values(by='training', ascending=False).iloc[0]
                selected_feature_name = _sorted_score_book.name
                selected_feature_training_score = _sorted_score_book['training']
                selected_feature_blindtest_score = _sorted_score_book['blind-test']
                selected_feature_score_diff = round(abs(selected_feature_training_score - selected_feature_blindtest_score),3)
                total_feature_pd = pd.concat([total_feature_pd,pd.DataFrame({'feature':selected_feature_name,'training':selected_feature_training_score,'blind-test':selected_feature_blindtest_score,'diff':selected_feature_score_diff},index=[0])],ignore_index=True)

            selected_feature_list.append(selected_feature_name)

            if greedy_feature_num == len(original_feature_list) or greedy_feature_num == max_feats:
                stop_signal = True
            else:
                greedy_feature_num +=1

            total_feature_pd.index +=1
            _target_label = re.sub('[()]|\/','_',target_label)
            total_feature_pd.to_csv('{}_greedy_fs_{}_{}_{}_total_feature.csv'.format(timestr,input_filename,algorithm, _target_label))

            if isinstance(blind_pd,bool) and len(remaining_feature_list) ==1:
                print("Your Job is Done")
                stop_signal = True

            print("===== Greedy-based selected features =====")
            print("{}".format(total_feature_pd))
            print("============================================")
    return True


def parallelize_list(_list, func, num_cores): # Under Developmenet
    list_split = np.array_split(_list, num_cores)
    pool = Pool(num_cores)
    df = pool.map(func, list_split)
    pool.close()
    pool.join()

if __name__ == '__main__':
    output_dir = os.path.dirname(os.path.abspath(__file__))

    # REQUIRED    
    parser = argparse.ArgumentParser(description='ex) python greedy_classifier.py ExtraTrees train.csv age')
    parser.add_argument("algorithm", help="Choose algorithm [GB,XGBOOST,RF,ExtraTrees,GAUSSIAN,ADABOOST,KNN,SVC,MLP,DecisionTree,LG]")
    parser.add_argument("train_csv", help="Choose input CSV(comma-separated values) format file")
    parser.add_argument("target_label", help="Type the name of label")

    # OPTIONAL    
    parser.add_argument("-n_cores", help="Choose the number of cores to use", type=int, default=4)
    parser.add_argument("-metric", help="Choose one metric for greedy_feature selection", choices=['roc_auc','matthew','bacc','f1'], default="matthew")    
    parser.add_argument("-output_dir", help="Choose folder to save result(CSV)",default=output_dir)
    parser.add_argument("-cutoff", help="Set cutoff value for the difference between training and blind-test performance, default=0.1", default=0.1, type=float)
    parser.add_argument("-no-scaling", help="Disable StandardScaler",action='store_true')    
    parser.add_argument("-blind_csv", help="Choose input CSV(comma-separated values) format file",
                        default=False)
    parser.add_argument("-mute", help="Disable TQDM progress bar",action='store_true')    
    parser.add_argument('-resume_csv', help="Provide a Greedy result to continue running Greedy from the file", default=False)
    parser.add_argument('-soft_cutoff', help="Enable to keep picking features even if there is no feature satisfying 'Cutoff' criteria.", action='store_true')
    parser.add_argument("-kfold", help="Choose CV Fold number", type=int, default=10)
    parser.add_argument("-groups", help="Choose group column name", type=str, default=None)
    parser.add_argument("-strategy", help="Choose CV strategy", type=str, choices=['kfold', 'groupkfold', 'stratified', 'stratified_group'], default='kfold')
    parser.add_argument("-max_feats", help="Limit the maximum number of features", type=int, default=0)

    args = parser.parse_args()
    with open(f'{timestr}_greedy_args.log', 'w') as f:
        f.write(str(args.__dict__))
    forwardGreedyFeature(args)
