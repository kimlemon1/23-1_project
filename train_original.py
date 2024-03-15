import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch
from matplotlib import pyplot as plt


device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
print(device)

left_right = ["left_", "right_"]
name_list = ["chaeyun", "chanwoo", "inseo", "jaehoon", "jeongwoo", "junho", "kihyun", "suho", "sunghyun", "wonyoung"]
lf_rf = ["_lf", "_rf"]

for ff in lf_rf:
    filename = "xgb_model"+ff+".model"
    total_dataset = pd.DataFrame()
    for direction in left_right:
        for name in name_list:
            
            person_data = pd.read_csv("XY_dataset/"+direction+name+ff+".csv")
            total_dataset = pd.concat((total_dataset, person_data), sort=False)
    
    ###### label 2 제거
    total_dataset.loc[total_dataset['label'] == 3, 'label'] = 2
    total_dataset.loc[total_dataset['label'] == 4, 'label'] = 3
    total_dataset.loc[total_dataset['label'] == 5, 'label'] = 4

    X = total_dataset.iloc[:,1:100]
    Y = total_dataset.iloc[:,100]
    #X = torch.tensor(X).to(device)
    #Y = torch.tensor(Y).to(device)
    x_train_all, x_test, y_train_all, y_test = train_test_split(X,Y, test_size=0.1, shuffle=True)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train_all, y_train_all, test_size=0.1, shuffle=True)

    estim_ls = [250, 300, 350, 400, 450, 500]
    lr_ls = [0.1,0.05, 0.15, 0.2,0.25]
    depth_ls = [5, 6]
    
    max_score = -1.0

    for estim in estim_ls:
        for lr in lr_ls:
            for dep in depth_ls:
                model = XGBClassifier(n_estimators = estim, learning_rate = lr, max_depth = dep)

                model.fit(x_train,y_train)
                y_hat = model.predict(x_valid)
                score = accuracy_score(y_hat, y_valid)
                
                if score > max_score:
                    max_param=[estim, lr, dep]
                    max_score = score
    model = XGBClassifier(n_estimators = max_param[0], learning_rate = max_param[1], max_depth = max_param[2], random_state=32)
    model.fit(x_train_all, y_train_all)
    y_pred = model.predict(x_test)
    print("accuracy of model"+ ff + "is :", accuracy_score(y_pred, y_test))
    print("best parameter : ", max_param)
    model.save_model(filename)
    
    