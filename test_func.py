import pandas as pd
import numpy as np
import os
from sklearn.impute import KNNImputer


def Missing_Counts( Data, NoMissing=True ) : 
    missing = Data.isnull().sum()  
    
    if NoMissing==False :
        missing = missing[ missing>0 ]
        
    missing.sort_values( ascending=False, inplace=True )  
    Missing_Count = pd.DataFrame( { 'Column Name':missing.index, 'Missing Count':missing.values } ) 
    Missing_Count[ 'Percentage(%)' ] = Missing_Count['Missing Count'].apply( lambda x: '{:.2%}'.format(x/Data.shape[0] ))
    return Missing_Count

# datasets = ['ad']

# for idx, dataset in enumerate(datasets):
#     data = pd.read_csv(f"dataset/study2/{dataset}.csv")
#     missing_df = Missing_Counts(data, NoMissing=False)
#     print(missing_df)

    # # 刪除缺失比例超過 45% 的欄位
    # columns_to_drop = missing_df[missing_df['Missing Count'] / data.shape[0] > 0.45]['Column Name'].values
    # df_cleaned = data.drop(columns=columns_to_drop)

    # # 使用 KNN Imputation 來填補剩下的缺失值
    # imputer = KNNImputer(n_neighbors=5)
    # df_imputed = pd.DataFrame(imputer.fit_transform(df_cleaned), columns=df_cleaned.columns)

    # df_imputed.to_csv(f"dataset/study2/{dataset}.csv", index=False)


#################################################################################################
# read .mat in python and convert to .py
def convertMatToPy(dataset) :
    from scipy.io import loadmat 
    import pandas as pd
    data = loadmat(rf"C:\Users\a2942\Downloads\{dataset}.mat")
    print(data.keys())
    x = data["X"]
    y = data["Y"]
    df = pd.DataFrame(pd.concat([pd.DataFrame(x), pd.DataFrame(y)], axis=1))
    df.to_csv(f"dataset/study2/{dataset}.csv", index=False)


# dataset = 'PCMAC'
# # convertMatToPy(dataset)
# data = pd.read_csv(f"dataset/study2/{dataset}.csv")
# print(data.isnull().sum())

# data.rename(columns={'0.1': 'Class'}, inplace=True)
# data['Class'] = data['Class'].replace({'-1': 0, '1': 1})
# data.to_csv(f"dataset/study2/{dataset}.csv", index=False)

#################################################################################################
datasets = ['heart_records', 'bank_marketing', 'statlog', 'SPECTF', 'segmentationData', 'taiwanese_bankruptcy', 'lsvt', 'madelon', 'secom', 
            'pd_speech_features', 'qsar_oral_toxicity', 'toxicity', 'ad', 'hiva_agnostic', 'christine', 'kits-subset', 'SMK_CAN_187']

main_dir = r"C:\Users\a2942\Documents\ncu\論文\autoencoder\dataset\study2\feature" 

# 建立資料夾
for dataset in datasets:
    base_path = os.path.join(main_dir, dataset)
    os.makedirs(base_path, exist_ok=True)  # 若資料夾已存在，則不報錯

    for subfolder in ['ae', 'dae', 'sae', 'vae']:
        path = os.path.join(base_path, subfolder)
        os.makedirs(path, exist_ok=True)


print("資料夾建立完成")


#################################################################################################
# 多分類
# from sklearn.svm import SVC
# from sklearn.metrics import roc_auc_score
# from sklearn.preprocessing import LabelEncoder
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.model_selection import train_test_split

# def run_svc(x_train, x_test, y_train, y_test):
#     model = SVC(kernel='rbf', decision_function_shape='ovr', probability=True)
#     model.fit(x_train, y_train)
#     y_probs = model.predict_proba(x_test)
#     encoder = OneHotEncoder()
#     y_test_encoded = encoder.fit_transform(y_test.reshape(-1, 1))
    
#     # 計算多分類 AUC 分數
#     auc_score = roc_auc_score(y_test_encoded, y_probs, multi_class='ovr')

#     return auc_score


# path = r"C:\Users\a2942\Downloads\zoo.csv" 
# df = pd.read_csv(path, delimiter=',')
# x = df.drop(columns=['Class'])
# labelencoder = LabelEncoder()
# y = pd.Series(labelencoder.fit_transform(df['Class']))

# string_columns = x.select_dtypes(include=['object']).columns
# for col in string_columns:
#     labelencoder = LabelEncoder()
#     x[col] = labelencoder.fit_transform(x[col])

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state=42)
# auc = run_svc(x_train, x_test, y_train, y_test)

# print(auc)