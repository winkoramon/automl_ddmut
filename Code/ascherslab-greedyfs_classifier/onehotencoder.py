# ********************************************************
# *               University of Melborune                *
# *   ----------------------------------------------     *
# * Yoochan Myung - ymyung@student.unimelb.edu.au        *
# * Last modification :: 11/07/2019                      *
# *   ----------------------------------------------     *
# ********************************************************
import os
import sys
import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MultiLabelBinarizer

timestr = time.strftime("%Y%m%d_%H%M%S")

def main(input_csv, target_label):

    output_filename = timestr + "_" + os.path.split(input_csv)[1][:-4] + "_encoded.csv"

    input_csv = pd.read_csv(input_csv, header=0)
    input_ID = input_csv['ID']
    input_label = input_csv[target_label]
    input_csv.drop(['ID', target_label], axis=1, inplace=True)

    onehotencoded_data = pd.get_dummies(input_csv)
    new_dataframe = pd.concat([input_ID,onehotencoded_data,input_label], axis=1)

    new_dataframe.to_csv(output_filename, index=False)

if __name__ == '__main__':
    input_csv = sys.argv[1]
    target_label = sys.argv[2]
    main(input_csv, target_label)