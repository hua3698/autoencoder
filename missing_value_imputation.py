import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer


def Missing_Counts( Data, NoMissing=True ) : 
    missing = Data.isnull().sum()  
    
    if NoMissing==False :
        missing = missing[ missing>0 ]
        
    missing.sort_values( ascending=False, inplace=True )  
    Missing_Count = pd.DataFrame( { 'Column Name':missing.index, 'Missing Count':missing.values } ) 
    Missing_Count[ 'Percentage(%)' ] = Missing_Count['Missing Count'].apply( lambda x: '{:.2%}'.format(x/Data.shape[0] ))
    return  Missing_Count


# data = pd.read_csv('dataset/study2/secom.csv')

# missing_df = Missing_Counts(data, NoMissing=False)

# # 刪除缺失比例超過 45% 的欄位
# columns_to_drop = missing_df[missing_df['Missing Count'] / data.shape[0] > 0.45]['Column Name'].values
# df_cleaned = data.drop(columns=columns_to_drop)

# # 使用 KNN Imputation 來填補剩下的缺失值
# imputer = KNNImputer(n_neighbors=5)
# df_imputed = pd.DataFrame(imputer.fit_transform(df_cleaned), columns=df_cleaned.columns)

# df_imputed.to_csv('dataset/study2/secom_imputation.csv', index=False)


#################################################################################################
# read .mat in python and convert to .py
def convertMatToPy(dataset) :
    from scipy.io import loadmat 
    import pandas as pd
    data = loadmat(rf"C:\Users\a2942\Downloads\{dataset}.mat")
    x = data["X"]
    y = data["Y"]
    df = pd.DataFrame(pd.concat([pd.DataFrame(x), pd.DataFrame(y)], axis=1))
    df.to_csv(f"dataset/study2/{dataset}.csv", index=False)



dataset = 'SMK_CAN_187'
# convertMatToPy(dataset)
data = pd.read_csv(f"dataset/study2/{dataset}.csv")
# print(data.isnull().sum())

data.rename(columns={'0.1': 'Class'}, inplace=True)
data['Class'] = data['Class'].replace({'-1': 0, '1': 1})
data.to_csv(f"dataset/study2/{dataset}.csv", index=False)