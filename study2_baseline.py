from library.common import *
from library.functions import *
from library import *
from sklearn.model_selection import KFold

datasets = ['heart_records', 'bank_marketing', 'statlog', 'SPECTF', 'segmentationData', 'taiwanese_bankruptcy', 'lsvt', 'madelon', 'secom', 
            'pd_speech_features', 'qsar_oral_toxicity', 'toxicity', 'ad', 'hiva_agnostic', 'christine']

for idx, dataset in enumerate(datasets):
    print(dataset)
    svm_auc = [];  knn_auc = []; cart_auc = []; mlp_auc = []; xgb_auc = []
    smote_svm_auc = [];  smote_knn_auc = []; smote_cart_auc = []; smote_mlp_auc = []; smote_xgb_auc = []
    cluster_svm_auc = []; cluster_knn_auc = []; cluster_cart_auc = []; cluster_mlp_auc = []; cluster_xgb_auc = []
    smoteenn_svm_auc = []; smoteenn_knn_auc = []; smoteenn_cart_auc = []; smoteenn_mlp_auc = []; smoteenn_xgb_auc = []

    df = pd.read_csv('dataset/study2/' + dataset + '.csv', delimiter=',')
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    x = df.drop(columns=['Class'])
    labelencoder = LabelEncoder()
    y = pd.Series(labelencoder.fit_transform(df['Class']))

    string_columns = x.select_dtypes(include=['object']).columns
    for col in string_columns:
        labelencoder = LabelEncoder()
        x[col] = labelencoder.fit_transform(x[col])

    for fold, (train_index, test_index) in enumerate(kf.split(x), start=1):
        print(fold)

        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        auc = run_svc(x_train, x_test, y_train, y_test)
        svm_auc.append(auc)

        auc = run_knn(x_train, x_test, y_train, y_test)
        knn_auc.append(auc)

        auc = run_cart(x_train, x_test, y_train, y_test)
        cart_auc.append(auc)

        auc = run_mlp(x_train, x_test, y_train, y_test)
        mlp_auc.append(auc)

        auc = run_xgboost(x_train, x_test, y_train, y_test)
        xgb_auc.append(auc)

        # smote
        x_train_smote, y_train_smote = data_resample('smote', x_train, y_train)
        x_train_smote = pd.DataFrame(x_train_smote, columns=x_train.columns)
        y_train_smote = pd.DataFrame(y_train_smote, columns=['Class'])

        auc = run_svc(x_train_smote, x_test, y_train_smote, y_test)
        smote_svm_auc.append(auc)

        auc = run_knn(x_train_smote, x_test, y_train_smote, y_test)
        smote_knn_auc.append(auc)

        auc = run_cart(x_train_smote, x_test, y_train_smote, y_test)
        smote_cart_auc.append(auc)

        auc = run_mlp(x_train_smote, x_test, y_train_smote, y_test)
        smote_mlp_auc.append(auc)

        auc = run_xgboost(x_train_smote, x_test, y_train_smote, y_test)
        smote_xgb_auc.append(auc)

        # cluster centroid
        x_train_smote, y_train_smote = data_resample('cluster', x_train, y_train)
        x_train_smote = pd.DataFrame(x_train_smote, columns=x_train.columns)
        y_train_smote = pd.DataFrame(y_train_smote, columns=['Class'])

        auc = run_svc(x_train_smote, x_test, y_train_smote, y_test)
        cluster_svm_auc.append(auc)

        auc = run_knn(x_train_smote, x_test, y_train_smote, y_test)
        cluster_knn_auc.append(auc)

        auc = run_cart(x_train_smote, x_test, y_train_smote, y_test)
        cluster_cart_auc.append(auc)

        auc = run_mlp(x_train_smote, x_test, y_train_smote, y_test)
        cluster_mlp_auc.append(auc)

        auc = run_xgboost(x_train_smote, x_test, y_train_smote, y_test)
        cluster_xgb_auc.append(auc)

        # smotenn
        x_train_smote, y_train_smote = data_resample('smotenn', x_train, y_train)
        x_train_smote = pd.DataFrame(x_train_smote, columns=x_train.columns)
        y_train_smote = pd.DataFrame(y_train_smote, columns=['Class'])

        auc = run_svc(x_train_smote, x_test, y_train_smote, y_test)
        smoteenn_svm_auc.append(auc)

        auc = run_knn(x_train_smote, x_test, y_train_smote, y_test)
        smoteenn_knn_auc.append(auc)

        auc = run_cart(x_train_smote, x_test, y_train_smote, y_test)
        smoteenn_cart_auc.append(auc)

        auc = run_mlp(x_train_smote, x_test, y_train_smote, y_test)
        smoteenn_mlp_auc.append(auc)

        auc = run_xgboost(x_train_smote, x_test, y_train_smote, y_test)
        smoteenn_xgb_auc.append(auc)


    print(dataset)
    print(np.mean(svm_auc))
    print(np.mean(smote_svm_auc))
    print(np.mean(cluster_svm_auc))
    print(np.mean(smoteenn_svm_auc))
    print(np.mean(knn_auc))
    print(np.mean(smote_knn_auc))
    print(np.mean(cluster_knn_auc))
    print(np.mean(smoteenn_knn_auc))
    print(np.mean(cart_auc))
    print(np.mean(smote_cart_auc))
    print(np.mean(cluster_cart_auc))
    print(np.mean(smoteenn_cart_auc))
    print('------------------')

    new = [dataset, round(np.mean(svm_auc), 3), round(np.mean(smote_svm_auc), 3),  round(np.mean(cluster_svm_auc), 3), round(np.mean(smoteenn_svm_auc), 3), 
            round(np.mean(knn_auc), 3), round(np.mean(smote_knn_auc), 3), round(np.mean(cluster_knn_auc), 3), round(np.mean(smoteenn_knn_auc), 3), 
            round(np.mean(cart_auc), 3), round(np.mean(smote_cart_auc), 3), round(np.mean(cluster_cart_auc), 3), round(np.mean(smoteenn_cart_auc), 3),
            round(np.mean(mlp_auc), 3), round(np.mean(smote_mlp_auc), 3), round(np.mean(cluster_mlp_auc), 3), round(np.mean(smoteenn_mlp_auc), 3), 
            round(np.mean(xgb_auc), 3), round(np.mean(smote_xgb_auc), 3), round(np.mean(cluster_xgb_auc), 3), round(np.mean(smoteenn_xgb_auc), 3),]

    with open('result/study2/baseline.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(new)

