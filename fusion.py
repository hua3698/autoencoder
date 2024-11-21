from library.common import *
from library import *
from ae import *

def train_ae(ver, size, x_train, x_test):
    if ver == 'sae_210':
        x_train_encoded, x_test_encoded = train_sae_210(x_train, x_test, size)

    elif ver == 'sae_220':
        x_train_encoded, x_test_encoded = train_sae_220(x_train, x_test, size) 

    elif ver == 'sae_230':
        x_train_encoded, x_test_encoded = train_sae_230(x_train, x_test, size)

    elif ver == 'sae_240':
        x_train_encoded, x_test_encoded = train_sae_240(x_train, x_test, size)

    return x_train_encoded, x_test_encoded


result = pd.DataFrame()
ae_version = 'sae_220'
model = 'knn'  
set_size = 32
datasets = ['page-blocks-1-3_vs_4', 'glass5', 'poker-9_vs_7', 'poker-8-9_vs_6']

for idx, dataset in enumerate(datasets):

    ae = []
    ae_smote = []; ae_cluster = []; ae_smotenn = []
    smote_ae = []; cluster_ae = []; smotenn_ae = []

    for times in range(1,6):

        train = "{}-5-{}{}.dat".format(dataset, times, 'tra')
        test = "{}-5-{}{}.dat".format(dataset, times, 'tst')
        df_train = pd.read_csv('../dataset/keel/' + dataset + '-5-fold/' + train, delimiter=',')
        df_test = pd.read_csv('../dataset/keel/' + dataset + '-5-fold/' + test, delimiter=',')

        x_train, x_test, y_train, y_test = data_preprocess(df_train, df_test)

        # ae>fusion
        ae_train, ae_test = train_ae(ae_version, set_size, x_train, x_test)
        fusion_train, fusion_test = feature_fusion(x_train, x_test, ae_train, ae_test)
        auc1 = run_model(model, fusion_train, fusion_test, y_train, y_test)
        ae.append(auc1)

        # ae>fusion+smote
        ae_x_train_smote, ae_y_train_smote = data_resample('smote', fusion_train, y_train)
        auc2 = run_model(model, ae_x_train_smote, fusion_test, ae_y_train_smote, y_test)
        ae_smote.append(auc2)

        # ae>fusion+cluster
        ae_x_train_smote, ae_y_train_smote = data_resample('cluster', fusion_train, y_train)
        auc3 = run_model(model, ae_x_train_smote, fusion_test, ae_y_train_smote, y_test)
        ae_cluster.append(auc3)

        # ae>fusion+smotenn
        ae_x_train_smote, ae_y_train_smote = data_resample('smotenn', fusion_train, y_train)
        auc4 = run_model(model, ae_x_train_smote, fusion_test, ae_y_train_smote, y_test)
        ae_smotenn.append(auc4)

        # smote+ae>fusion
        x_train_smote, y_train_smote = data_resample('smote', x_train, y_train)
        ae_train, ae_test = train_ae(ae_version, set_size, x_train_smote, x_test)
        fusion_train, fusion_test = feature_fusion(x_train_smote, x_test, ae_train, ae_test)
        auc5 = run_model(model, fusion_train, fusion_test, y_train_smote, y_test)
        smote_ae.append(auc5)

        # cluster+ae>fusion
        x_train_smote, y_train_smote = data_resample('cluster', x_train, y_train)
        ae_train, ae_test = train_ae(ae_version, set_size, x_train_smote, x_test)
        fusion_train, fusion_test = feature_fusion(x_train_smote, x_test, ae_train, ae_test)
        auc6 = run_model(model, fusion_train, fusion_test, y_train_smote, y_test)
        cluster_ae.append(auc6)

        # smotenn+ae>fusion
        x_train_smote, y_train_smote = data_resample('smotenn', x_train, y_train)
        ae_train, ae_test = train_ae(ae_version, set_size, x_train_smote, x_test)
        fusion_train, fusion_test = feature_fusion(x_train_smote, x_test, ae_train, ae_test)
        auc7 = run_model(model, fusion_train, fusion_test, y_train_smote, y_test)
        smotenn_ae.append(auc7)

        new = [dataset, model, ae_version, 
               round(np.mean(ae), 3), round(np.mean(ae_smote), 3), round(np.mean(smote_ae), 3), round(np.mean(ae_cluster), 3), 
               round(np.mean(cluster_ae), 3), round(np.mean(ae_smotenn), 3), round(np.mean(smotenn_ae), 3)]

    with open('result/under15.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(new)
