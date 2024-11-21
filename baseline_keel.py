from library.common import *
from library.functions import *
from library import *

result = pd.DataFrame()
datasets = ['kddcup-guess_passwd_vs_satan', 'kddcup-land_vs_portsweep', 'kr-vs-k-zero_vs_eight', 'kddcup-land_vs_satan', 'kr-vs-k-zero_vs_fifteen']

for idx, dataset in enumerate(datasets):

    minmax = preprocessing.MinMaxScaler()

    # svm_acc = [];  knn_acc = []
    svm_auc = [];  knn_auc = []; cart_auc = []; mlp_auc = []; xgb_auc = []
    smote_svm_auc = [];  smote_knn_auc = []; smote_cart_auc = []; smote_mlp_auc = []; smote_xgb_auc = []
    cluster_svm_auc = []; cluster_knn_auc = []; cluster_cart_auc = []; cluster_mlp_auc = []; cluster_xgb_auc = []
    smoteenn_svm_auc = []; smoteenn_knn_auc = []; smoteenn_cart_auc = []; smoteenn_mlp_auc = []; smoteenn_xgb_auc = []

    for times in range(1,6):

        train = "{}-5-{}{}.dat".format(dataset, times, 'tra')
        test = "{}-5-{}{}.dat".format(dataset, times, 'tst')

        df_train = pd.read_csv('dataset/keel/' + dataset + '/' + train, delimiter=',')
        df_test = pd.read_csv('dataset/keel/' + dataset + '/' + test, delimiter=',')

        x_train, x_test, y_train, y_test = data_preprocess(df_train, df_test)

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

    new = [dataset, np.mean(svm_auc), np.mean(smote_svm_auc),  np.mean(cluster_svm_auc), np.mean(smoteenn_svm_auc), 
            np.mean(knn_auc), np.mean(smote_knn_auc), np.mean(cluster_knn_auc), np.mean(smoteenn_knn_auc), 
            np.mean(cart_auc), np.mean(smote_cart_auc), np.mean(cluster_cart_auc), np.mean(smoteenn_cart_auc),
            np.mean(mlp_auc), np.mean(smote_mlp_auc), np.mean(cluster_mlp_auc), np.mean(smoteenn_mlp_auc), 
            np.mean(xgb_auc), np.mean(smote_xgb_auc), np.mean(cluster_xgb_auc), np.mean(smoteenn_xgb_auc),]

    with open('result/baseline.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(new)

  # new=pd.DataFrame({'dataset':dataset,
  #           'svm_auc': np.mean(svm_auc),
  #           'knn_auc': np.mean(knn_auc),
  #           'c45_auc': np.mean(c45_auc),
  #           'cart_auc': np.mean(cart_auc),
  #           'smote_svm_auc': np.mean(smote_svm_auc),
  #           'smote_knn_auc': np.mean(smote_knn_auc),
  #           'smote_c45_auc': np.mean(smote_c45_auc),
  #           'smote_cart_auc': np.mean(smote_cart_auc),
  #           'now': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
  #           }, index=[idx])
  
  # result = pd.concat([result, new], ignore_index=True)
# result.to_csv('result/bank/.csv', index=False, encoding='utf-8')
