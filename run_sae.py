from library.common import *
from library.functions import *
from library.sae import *

def train_ae(ver, size, x_train, x_test):
    if ver == 'sae_210':
        x_train_encoded, x_test_encoded = train_sae_210(x_train, x_test, size)
    elif ver == 'sae_220':
        x_train_encoded, x_test_encoded = train_sae_220(x_train, x_test, size)
    elif ver == 'sae_230':
        x_train_encoded, x_test_encoded = train_sae_230(x_train, x_test, size)
    elif ver == 'sae_240':
        x_train_encoded, x_test_encoded = train_sae_240(x_train, x_test, size)
    elif ver == 'sae_310':
        x_train_encoded, x_test_encoded = train_sae_310(x_train, x_test, size)
    elif ver == 'sae_320':
        x_train_encoded, x_test_encoded = train_sae_320(x_train, x_test, size)
    elif ver == 'sae_330':
        x_train_encoded, x_test_encoded = train_sae_330(x_train, x_test, size)
    elif ver == 'sae_410':
        x_train_encoded, x_test_encoded = train_sae_410(x_train, x_test, size)
    elif ver == 'sae_420':
        x_train_encoded, x_test_encoded = train_sae_420(x_train, x_test, size)
    elif ver == 'sae_510':
        x_train_encoded, x_test_encoded = train_sae_510(x_train, x_test, size)
    elif ver == 'sae_610':
        x_train_encoded, x_test_encoded = train_sae_610(x_train, x_test, size)
    elif ver == 'sae_710':
        x_train_encoded, x_test_encoded = train_sae_710(x_train, x_test, size)
    return x_train_encoded, x_test_encoded

model = 'knn'
ae_version = ['sae_210', 'sae_220', 'sae_230', 'sae_240']
# ae_version = ['sae_220', 'sae_310', 'sae_410', 'sae_610']
result_file = 'test.csv'

for i, ae_version in enumerate(ae_version):
    for idx, dataset in enumerate(datasets):
        ae = []
        ae_smote = []; ae_cluster = []; ae_smotenn = []
        smote_ae = []; cluster_ae = []; smotenn_ae = []

        for times in range(1,6):

            train = "{}-5-{}{}.dat".format(dataset, times, 'tra')
            test = "{}-5-{}{}.dat".format(dataset, times, 'tst')

            df_train = pd.read_csv('dataset/keel/' + dataset + '/' + train, delimiter=',')
            df_test = pd.read_csv('dataset/keel/' + dataset + '/' + test, delimiter=',')

            x_train, x_test, y_train, y_test = data_preprocess(df_train, df_test)

            # ae
            ae_train, ae_test = train_ae(ae_version, set_size, x_train, x_test)
            auc1 = run_model(model, ae_train, ae_test, y_train, y_test)
            ae.append(auc1)

            # ae+smote
            ae_x_train_smote, ae_y_train_smote = data_resample('smote', ae_train, y_train)
            auc2 = run_model(model, ae_x_train_smote, ae_test, ae_y_train_smote, y_test)
            ae_smote.append(auc2)

            # ae+cluster
            ae_x_train_smote, ae_y_train_smote = data_resample('cluster', ae_train, y_train)
            auc3 = run_model(model, ae_x_train_smote, ae_test, ae_y_train_smote, y_test)
            ae_cluster.append(auc3)

            # ae+smotenn
            ae_x_train_smote, ae_y_train_smote = data_resample('smotenn', ae_train, y_train)
            auc4 = run_model(model, ae_x_train_smote, ae_test, ae_y_train_smote, y_test)
            ae_smotenn.append(auc4)

            # smote+ae
            x_train_smote, y_train_smote = data_resample('smote', x_train, y_train)
            ae_train, ae_test = train_ae(ae_version, set_size, x_train_smote, x_test)
            auc5 = run_model(model, ae_train, ae_test, y_train_smote, y_test)
            smote_ae.append(auc5)

            # cluster+ae
            x_train_smote, y_train_smote = data_resample('cluster', x_train, y_train)
            ae_train, ae_test = train_ae(ae_version, set_size, x_train_smote, x_test)
            auc6 = run_model(model, ae_train, ae_test, y_train_smote, y_test)
            cluster_ae.append(auc6)

            # smotenn+ae
            x_train_smote, y_train_smote = data_resample('smotenn', x_train, y_train)
            ae_train, ae_test = train_ae(ae_version, set_size, x_train_smote, x_test)
            auc7 = run_model(model, ae_train, ae_test, y_train_smote, y_test)
            smotenn_ae.append(auc7)

        f_ae = []
        f_ae_smote = []; f_ae_cluster = []; f_ae_smotenn = []
        f_smote_ae = []; f_cluster_ae = []; f_smotenn_ae = []

        fusion = True

        for times in range(1,6):

            training = "{}-5-{}{}.dat".format(dataset, times, 'tra')
            testing = "{}-5-{}{}.dat".format(dataset, times, 'tst')

            df_train = pd.read_csv('dataset/keel/' + dataset + '/' + training, delimiter=',')
            df_test = pd.read_csv('dataset/keel/' + dataset + '/' + testing, delimiter=',')

            x_train, x_test, y_train, y_test = data_preprocess(df_train, df_test)

            # ae>fusion
            ae_train, ae_test = train_ae(ae_version, set_size, x_train, x_test)
            fusion_train, fusion_test = feature_fusion(fusion, x_train, x_test, ae_train, ae_test)
            auc8 = run_model(model, fusion_train, fusion_test, y_train, y_test)
            f_ae.append(auc8)

            # ae>fusion+smote
            ae_x_train_smote, ae_y_train_smote = data_resample('smote', fusion_train, y_train)
            auc9 = run_model(model, ae_x_train_smote, fusion_test, ae_y_train_smote, y_test)
            f_ae_smote.append(auc9)

            # ae>fusion+cluster
            ae_x_train_smote, ae_y_train_smote = data_resample('cluster', fusion_train, y_train)
            auc10 = run_model(model, ae_x_train_smote, fusion_test, ae_y_train_smote, y_test)
            f_ae_cluster.append(auc10)

            # ae>fusion+smotenn
            ae_x_train_smote, ae_y_train_smote = data_resample('smotenn', fusion_train, y_train)
            auc11 = run_model(model, ae_x_train_smote, fusion_test, ae_y_train_smote, y_test)
            f_ae_smotenn.append(auc11)

            # smote+ae>fusion
            x_train_smote, y_train_smote = data_resample('smote', x_train, y_train)
            ae_train, ae_test = train_ae(ae_version, set_size, x_train_smote, x_test)
            fusion_train, fusion_test = feature_fusion(fusion, x_train_smote, x_test, ae_train, ae_test)
            auc12 = run_model(model, fusion_train, fusion_test, y_train_smote, y_test)
            f_smote_ae.append(auc12)

            # cluster+ae>fusion
            x_train_smote, y_train_smote = data_resample('cluster', x_train, y_train)
            ae_train, ae_test = train_ae(ae_version, set_size, x_train_smote, x_test)
            fusion_train, fusion_test = feature_fusion(fusion, x_train_smote, x_test, ae_train, ae_test)
            auc13 = run_model(model, fusion_train, fusion_test, y_train_smote, y_test)
            f_cluster_ae.append(auc13)

            # smotenn+ae>fusion
            x_train_smote, y_train_smote = data_resample('smotenn', x_train, y_train)
            ae_train, ae_test = train_ae(ae_version, set_size, x_train_smote, x_test)
            fusion_train, fusion_test = feature_fusion(fusion, x_train_smote, x_test, ae_train, ae_test)
            auc14 = run_model(model, fusion_train, fusion_test, y_train_smote, y_test)
            f_smotenn_ae.append(auc14)

        new = [dataset, model, ae_version,
                round(np.mean(ae), 4), round(np.mean(ae_smote), 4), round(np.mean(smote_ae), 4), 
                round(np.mean(ae_cluster), 4), round(np.mean(cluster_ae), 4), round(np.mean(ae_smotenn), 4), round(np.mean(smotenn_ae), 4),
                round(np.mean(f_ae), 4), round(np.mean(f_ae_smote), 4), round(np.mean(f_smote_ae), 4), 
                round(np.mean(f_ae_cluster), 4), round(np.mean(f_cluster_ae), 4), round(np.mean(f_ae_smotenn), 4), round(np.mean(f_smotenn_ae), 4),
                datetime.now()]

        with open('result/' + result_file, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(new)

        K.clear_session()
        gc.collect()