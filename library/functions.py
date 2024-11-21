from library.common import *

def data_preprocess(df_train, df_test) :

    # Label encode
    labelencoder = LabelEncoder()
    y_train = labelencoder.fit_transform(df_train['Class'])
    y_test = labelencoder.fit_transform(df_test['Class'])

    x_train = df_train.drop(['Class'], axis=1)
    x_test = df_test.drop(['Class'], axis=1)

    string_columns = x_train.select_dtypes(include=['object']).columns

    for col in string_columns:
        labelencoder = LabelEncoder()
        x_train[col] = labelencoder.fit_transform(x_train[col])
        x_test[col] = labelencoder.fit_transform(x_test[col])

    # 特徵縮放 
    # svm knn mlp 對特徵的尺度敏感。如果特徵的範圍不同，它們可能會傾向於範圍較大的特徵，從而影響分類結果。
    # cart xgboost 決策樹類型基於分割節點，只考慮特徵的相對順序，不需要進行正規化
    minmax = preprocessing.MinMaxScaler()
    x_train_minmax = minmax.fit_transform(x_train)
    x_test_minmax = minmax.fit_transform(x_test)

    x_train = pd.DataFrame(x_train_minmax, columns = x_train.columns)
    x_test = pd.DataFrame(x_test_minmax, columns = x_test.columns)

    return x_train, x_test, y_train, y_test

def run_svc(x_train, x_test, y_train, y_test):
    model = SVC(kernel='rbf')
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    return roc_auc_score(y_test, y_predict)

def run_knn(x_train, x_test, y_train, y_test):
    model = KNeighborsClassifier()
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    return roc_auc_score(y_test, y_predict)

def run_cart(x_train, x_test, y_train, y_test):
    model = tree.DecisionTreeClassifier()
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    return roc_auc_score(y_test, y_predict)

def run_mlp(x_train, x_test, y_train, y_test):
    from sklearn.neural_network import MLPClassifier
    model = MLPClassifier(random_state=42, max_iter=300).fit(x_train, y_train)
    y_predict = model.predict(x_test)
    return roc_auc_score(y_test, y_predict)

def run_xgboost(x_train, x_test, y_train, y_test):
    from xgboost import XGBClassifier
    xgboostModel = XGBClassifier(n_estimators=100, learning_rate= 0.3).fit(x_train, y_train)
    y_predict = xgboostModel.predict(x_test)
    return roc_auc_score(y_test, y_predict)

def run_model(name, x_train, x_test, y_train, y_test):
    if name == 'svc':
        auc = run_svc(x_train, x_test, y_train, y_test)
    elif name == 'knn':
        auc = run_knn(x_train, x_test, y_train, y_test)
    elif name == 'cart':
        auc = run_cart(x_train, x_test, y_train, y_test)
    elif name == 'mlp':
        auc = run_mlp(x_train, x_test, y_train, y_test)
    elif name == 'xgb':
        auc = run_xgboost(x_train, x_test, y_train, y_test)
    return auc

def data_resample(sampling_method, x_train, y_train, random_state=42, default_neighbors=5):
    # 獲取少數類別的樣本數
    _, class_counts = np.unique(y_train, return_counts=True)
    minority_class_count = min(class_counts)

    # 計算適當的鄰居數
    n_neighbors = min(default_neighbors, minority_class_count - 1)
    if sampling_method == 'smote':
        smote = SMOTE(random_state=random_state, k_neighbors=n_neighbors)
    elif sampling_method == 'cluster':
        smote = ClusterCentroids(estimator=MiniBatchKMeans(n_init=1), random_state=random_state)
    elif sampling_method == 'smotenn':
        # 當進行smoteenn，會先進行smote使少類樣本增加至1:1，接著enn會將過於接近的樣本進行清除
        smote = SMOTE(k_neighbors=n_neighbors, random_state=random_state)
        smote = SMOTEENN(random_state=random_state, smote=smote)

    x_train_smote, y_train_smote = smote.fit_resample(x_train, y_train)

    return x_train_smote, y_train_smote

def feature_fusion(need, x_train, x_test, ae_train, ae_test):

    if need == False:
        return x_train, x_test

    ae_train = pd.DataFrame(ae_train)
    ae_test = pd.DataFrame(ae_test)
    ae_train.columns = ae_train.columns.astype(str)
    ae_test.columns = ae_test.columns.astype(str)

    fusion_train = pd.concat([x_train.reset_index(drop=True), ae_train.reset_index(drop=True)], axis=1)
    fusion_test = pd.concat([x_test.reset_index(drop=True), ae_test.reset_index(drop=True)], axis=1)

    return fusion_train, fusion_test