from library.common import *
from library import *

result = pd.DataFrame()
datasets = ['german', 'tej', 'polish_year1', 'polish_year2', 'polish_year3', 'polish_year4', 'polish_year5']

for idx, dataset in enumerate(datasets):

    minmax = preprocessing.MinMaxScaler()

    # svm_auc = [];  knn_auc = []; cart_auc = []
    # smote_svm_auc = [];  smote_knn_auc = [];  smote_cart_auc = []
    # cluster_svm_auc = []; cluster_knn_auc = []; cluster_cart_auc = []
    # smoteenn_svm_auc = [] ;smoteenn_knn_auc = [] ;smoteenn_cart_auc = []
    xgboost_auc = []
    smote_xgboost_auc = []
    cluster_xgboost_auc = []
    smoteenn_xgboost_auc = []

    for times in range(1,6):

        train = "{}_{}{}.csv".format(dataset, 'train', times)
        test = "{}_{}{}.csv".format(dataset, 'test', times)

        df_train = pd.read_csv('../dataset/bank/' + dataset + '/' + train, delimiter=',')
        df_test = pd.read_csv('../dataset/bank/' + dataset + '/' + test, delimiter=',')
        
        x_train, x_test, y_train, y_test = data_preprocess(df_train, df_test)


        # auc = run_svc(x_train, x_test, y_train, y_test)
        # svm_auc.append(auc)

        # auc = run_knn(x_train, x_test, y_train, y_test)
        # knn_auc.append(auc)

        # auc = run_cart(x_train, x_test, y_train, y_test)
        # cart_auc.append(auc)

        auc = run_xgboost(x_train, x_test, y_train, y_test)
        xgboost_auc.append(auc)

        # smote
        x_train_smote, y_train_smote = data_resample('smote', x_train, y_train)
        x_train_smote = pd.DataFrame(x_train_smote, columns=x_train.columns)
        y_train_smote = pd.DataFrame(y_train_smote, columns=['Class'])

        # auc = run_svc(x_train_smote, x_test, y_train_smote, y_test)
        # smote_svm_auc.append(auc)

        # auc = run_knn(x_train_smote, x_test, y_train_smote, y_test)
        # smote_knn_auc.append(auc)

        # auc = run_cart(x_train_smote, x_test, y_train_smote, y_test)
        # smote_cart_auc.append(auc)

        auc = run_xgboost(x_train_smote, x_test, y_train_smote, y_test)
        smote_xgboost_auc.append(auc)

        # cluster centroid
        x_train_smote, y_train_smote = data_resample('cluster', x_train, y_train)
        x_train_smote = pd.DataFrame(x_train_smote, columns=x_train.columns)
        y_train_smote = pd.DataFrame(y_train_smote, columns=['Class'])

        # auc = run_svc(x_train_smote, x_test, y_train_smote, y_test)
        # cluster_svm_auc.append(auc)

        # auc = run_knn(x_train_smote, x_test, y_train_smote, y_test)
        # cluster_knn_auc.append(auc)

        # auc = run_cart(x_train_smote, x_test, y_train_smote, y_test)
        # cluster_cart_auc.append(auc)

        auc = run_xgboost(x_train_smote, x_test, y_train_smote, y_test)
        cluster_xgboost_auc.append(auc)

        # smotenn
        x_train_smote, y_train_smote = data_resample('smotenn', x_train, y_train)
        x_train_smote = pd.DataFrame(x_train_smote, columns=x_train.columns)
        y_train_smote = pd.DataFrame(y_train_smote, columns=['Class'])

        # auc = run_svc(x_train_smote, x_test, y_train_smote, y_test)
        # smoteenn_svm_auc.append(auc)

        # auc = run_knn(x_train_smote, x_test, y_train_smote, y_test)
        # smoteenn_knn_auc.append(auc)

        # auc = run_cart(x_train_smote, x_test, y_train_smote, y_test)
        # smoteenn_cart_auc.append(auc)

        auc = run_xgboost(x_train_smote, x_test, y_train_smote, y_test)
        smoteenn_xgboost_auc.append(auc)


    print(dataset)
    # print(np.mean(svm_auc))
    # print(np.mean(smote_svm_auc))
    # print(np.mean(cluster_svm_auc))
    # print(np.mean(smoteenn_svm_auc))
    # print(np.mean(knn_auc))
    # print(np.mean(smote_knn_auc))
    # print(np.mean(cluster_knn_auc))
    # print(np.mean(smoteenn_knn_auc))
    # print(np.mean(cart_auc))
    # print(np.mean(smote_cart_auc))
    # print(np.mean(cluster_cart_auc))
    # print(np.mean(smoteenn_cart_auc))
    print(np.mean(xgboost_auc))
    print(np.mean(smote_xgboost_auc))
    print(np.mean(cluster_xgboost_auc))
    print(np.mean(smoteenn_xgboost_auc))
    print('------------------')

    # new = [dataset, np.mean(svm_auc), np.mean(smote_svm_auc),  np.mean(cluster_svm_auc), np.mean(smoteenn_svm_auc), 
    #         np.mean(knn_auc), np.mean(smote_knn_auc), np.mean(cluster_knn_auc), np.mean(smoteenn_knn_auc), 
    #         np.mean(cart_auc), np.mean(smote_cart_auc), np.mean(cluster_cart_auc), np.mean(smoteenn_cart_auc)]

    # with open('result/bank.csv', 'a', newline='') as csvfile:
    #     writer = csv.writer(csvfile, delimiter=',')
    #     writer.writerow(new)
