import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch
from matplotlib import pyplot as plt
import random

left_right = ["left_", "right_"]
name_list = ["chaeyun", "chanwoo", "inseo", "jaehoon", "jeongwoo", "junho", "kihyun", "suho", "sunghyun","wonyoung"]
lf_rf = ["_lf", "_rf"]

estim_range = [250, 300, 400, 500]
lr_range = [0.1, 0.15, 0.2, 0.25]

col_first = []
col_second = []
for i in range(99):
    col_first.append("before_"+str(i))
    col_second.append("now_"+str(i))
col_first.append("before_r_k_a")
col_first.append("before_l_k_a")
col_first.append("before_b_a")
col_second.append("now_r_k_a")
col_second.append("now_l_k_a")
col_second.append("now_b_a")
col_name = col_first+col_second

# change this to test others
TEST_NAME = "kihyun"
VALID_NAME = "wonyoung"
test_val_list = [TEST_NAME, VALID_NAME]

def postfix(y_hat, search_len):
    n = len(y_hat)
    k = search_len // 2
    for i in range(k, n - k):
        if y_hat[i] != y_hat[i-1]:
            vote = {0:0, 1:0, 2:0, 3:0, 4:0}
            for val in y_hat[i-k : i + k +1]:
                vote[val] += 1
            y_hat[i] = max(vote, key=vote.get)
    return y_hat

def train_with_temporal(X):
    pass

def train_someone(test_name, valid_name):
    for ff in lf_rf:
        filename = "{}{}.model".format(test_name, ff)
        total_dataset = pd.DataFrame()
        for direction in left_right:
            for name in name_list:
                if name == test_name:
                    continue
                elif name == valid_name:
                    valid_dataset = pd.read_csv("XY_dataset/"+direction+name+ff+".csv")
                else:
                    person_data = pd.read_csv("XY_dataset/"+direction+name+ff+".csv")
                    total_dataset = pd.concat((total_dataset, person_data), sort=False)

        #total_dataset = total_dataset.sample(frac=1).reset_index(drop=True)
        X = total_dataset[col_name]
        #X.drop("Unnamed: 0", axis=1, inplace=True)

        print("X shape :", X.shape)
        Y = total_dataset["label"]
        print("Y shape : ", Y.shape)

        valid_X = valid_dataset[col_name]
        #valid_X.drop("Unnamed: 0", axis=1, inplace=True)

        valid_Y = valid_dataset["label"]
        print("val_X shape : ", valid_X.shape)
        print("val_Y shape : ", valid_Y.shape)

        max_score = -1
        for n_estim in estim_range:
            for lr in lr_range:
                model = XGBClassifier(n_estimators = n_estim, learning_rate = lr, max_depth = 5)
                model.fit(X,Y)
                y_hat = model.predict(valid_X)
                acc_score = accuracy_score(valid_Y, y_hat)
                if acc_score > max_score:
                    max_score = acc_score
                    best_estim = n_estim
                    best_lr = lr
        print("For model ", ff)
        print("Best valid score :", max_score)
        print("Best n_estim :", best_estim)
        print("Best lr :", best_lr)
        model = XGBClassifier(n_estimators = best_estim, learning_rate = best_lr, max_depth = 5)
        total_X = pd.concat([X, valid_X])
        total_Y = pd.concat([Y, valid_Y])
        model.fit(total_X, total_Y)
        model.save_model("./model/"+filename)

def test_someone(someone):
    lf_model = XGBClassifier()
    rf_model = XGBClassifier()
    lf_model.load_model("./model/"+someone + "_lf.model")
    rf_model.load_model("./model/"+someone + "_rf.model")

    lf_1 =  pd.read_csv("XY_dataset/left_"+someone+"_lf.csv")
    lf_2 =  pd.read_csv("XY_dataset/right_"+someone+"_lf.csv")
    rf_1 =  pd.read_csv("XY_dataset/left_"+someone+"_rf.csv")
    rf_2 =  pd.read_csv("XY_dataset/right_"+someone+"_rf.csv")

    lf_test = pd.concat([lf_1, lf_2])
    rf_test = pd.concat([rf_1, rf_2])

    lf_X = lf_test[col_name]
    lf_Y = lf_test['label']

    rf_X = rf_test[col_name]
    rf_Y = rf_test['label']   

    lf_y_hat = lf_model.predict(lf_X)
    rf_y_hat = rf_model.predict(rf_X)

    post_lf_y_hat = postfix(lf_y_hat, 5)
    post_rf_y_hat = postfix(rf_y_hat, 5)

    print("Accuracy score of ",TEST_NAME, " model lf: ", accuracy_score(lf_Y, post_lf_y_hat))
    print("Accuracy score of ",TEST_NAME, " model rf: ", accuracy_score(rf_Y, post_rf_y_hat))

    plt.subplot(2,2,1)
    plt.title("lf_model")
    plt.plot(lf_Y.values)
    plt.plot(lf_y_hat)

    plt.subplot(2,2,2)
    plt.title("post lf_model")
    plt.plot(lf_Y.values)
    plt.plot(post_lf_y_hat)

    plt.subplot(2,2,3)
    plt.title("rf_model")
    plt.plot(rf_Y.values)
    plt.plot(rf_y_hat)

    plt.subplot(2,2,4)
    plt.title("post rf_model")
    plt.plot(rf_Y.values)
    plt.plot(post_rf_y_hat)

    plt.savefig("./output_image/{}_output.png".format(TEST_NAME))

train_someone(TEST_NAME, VALID_NAME)

test_someone(TEST_NAME)



    