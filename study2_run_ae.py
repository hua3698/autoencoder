from library.common import *
from library.functions import *
from sklearn.model_selection import KFold

model = 'svc'
autoencoder = 'ae'
ae_version = ['210']
datasets = ['heart_records', 'bank_marketing', 'statlog', 'SPECTF', 'segmentationData', 'taiwanese_bankruptcy', 'lsvt', 'madelon', 
            'secom', 'pd_speech_features', 'qsar_oral_toxicity', 'toxicity', 'ad', 'hiva_agnostic', 'christine']
result_file = 'study2_ae_base.csv'

for i, ae_version in enumerate(ae_version):
    for idx, dataset in enumerate(datasets):
        df = pd.read_csv('dataset/study2/' + dataset + '.csv', delimiter=',')
        x = df.drop(columns=['Class'])
        labelencoder = LabelEncoder()
        y = pd.Series(labelencoder.fit_transform(df['Class']))

        string_columns = x.select_dtypes(include=['object']).columns
        for col in string_columns:
            labelencoder = LabelEncoder()
            x[col] = labelencoder.fit_transform(x[col])

        ae = []
        ae_smote = []; ae_cluster = []; ae_smotenn = []
        smote_ae = []; cluster_ae = []; smotenn_ae = []

        start = datetime.now()

        ae_full = autoencoder + '_' + ae_version
        folder = 'dataset/study2/feature/' + dataset + '/' + autoencoder
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        for fold, (train_index, test_index) in enumerate(kf.split(x), start=1):

            x_train, x_test = x.iloc[train_index], x.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            y_train = pd.DataFrame(y_train)
            y_test = pd.DataFrame(y_test)
            y_train.to_csv(folder + '/train_ans_' + ae_version + '_' + str(fold) + '.csv', index=False, encoding='utf-8')
            y_test.to_csv(folder + '/test_ans_' + ae_version + '_' + str(fold) + '.csv', index=False, encoding='utf-8')

            # ae
            ae_train, ae_test = train_ae(ae_full, set_size, x_train, x_test)
            auc1 = run_model(model, ae_train, ae_test, y_train, y_test)
            ae.append(auc1)
            ae_train = pd.DataFrame(ae_train)
            ae_test = pd.DataFrame(ae_test)
            ae_train.to_csv(folder + '/1ae_train_' + ae_version + '_' + str(fold) + '.csv', index=False, encoding='utf-8')
            ae_test.to_csv(folder + '/1ae_test_' + ae_version + '_' + str(fold) + '.csv', index=False, encoding='utf-8')

            # ae+smote 
            ae_x_train_smote, ae_y_train_smote = data_resample('smote', ae_train, y_train)
            auc2 = run_model(model, ae_x_train_smote, ae_test, ae_y_train_smote, y_test)
            ae_smote.append(auc2)
            ae_x_train_smote = pd.DataFrame(ae_x_train_smote)
            ae_test = pd.DataFrame(ae_test)
            ae_x_train_smote.to_csv(folder + '/2ae+smote_train_' + ae_version + '_' + str(fold) + '.csv', index=False, encoding='utf-8')
            ae_test.to_csv(folder + '/2ae+smote_test_' + ae_version + '_' + str(fold) + '.csv', index=False, encoding='utf-8')
            ae_y_train_smote = pd.DataFrame(ae_y_train_smote)
            ae_y_train_smote.to_csv(folder + '/2ae+smote_train_ans_' + ae_version + '_' + str(fold) + '.csv', index=False, encoding='utf-8')

            # ae+cluster
            ae_x_train_smote, ae_y_train_smote = data_resample('cluster', ae_train, y_train)
            auc3 = run_model(model, ae_x_train_smote, ae_test, ae_y_train_smote, y_test)
            ae_cluster.append(auc3)
            ae_x_train_smote = pd.DataFrame(ae_x_train_smote)
            ae_test = pd.DataFrame(ae_test)
            ae_x_train_smote.to_csv(folder + '/3ae+cluster_train_' + ae_version + '_' + str(fold) + '.csv', index=False, encoding='utf-8')
            ae_test.to_csv(folder + '/3ae+cluster_test_' + ae_version + '_' + str(fold) + '.csv', index=False, encoding='utf-8')
            ae_y_train_smote = pd.DataFrame(ae_y_train_smote)
            ae_y_train_smote.to_csv(folder + '/3ae+cluster_train_ans_' + ae_version + '_' + str(fold) + '.csv', index=False, encoding='utf-8')

            # ae+smotenn
            ae_x_train_smote, ae_y_train_smote = data_resample('smotenn', ae_train, y_train)
            auc4 = run_model(model, ae_x_train_smote, ae_test, ae_y_train_smote, y_test)
            ae_smotenn.append(auc4)
            ae_x_train_smote = pd.DataFrame(ae_x_train_smote)
            ae_test = pd.DataFrame(ae_test)
            ae_x_train_smote.to_csv(folder + '/4ae+smotenn_train_' + ae_version + '_' + str(fold) + '.csv', index=False, encoding='utf-8')
            ae_test.to_csv(folder + '/4ae+smotenn_test_' + ae_version + '_' + str(fold) + '.csv', index=False, encoding='utf-8')
            ae_y_train_smote = pd.DataFrame(ae_y_train_smote)
            ae_y_train_smote.to_csv(folder + '/4ae+smotenn_train_ans_' + ae_version + '_' + str(fold) + '.csv', index=False, encoding='utf-8')

            # smote+ae
            x_train_smote, y_train_smote = data_resample('smote', x_train, y_train)
            ae_train, ae_test = train_ae(ae_full, set_size, x_train_smote, x_test)
            auc5 = run_model(model, ae_train, ae_test, y_train_smote, y_test)
            smote_ae.append(auc5)
            ae_train = pd.DataFrame(ae_train)
            ae_test = pd.DataFrame(ae_test)
            ae_train.to_csv(folder + '/5smote+ae_train_' + ae_version + '_' + str(fold) + '.csv', index=False, encoding='utf-8')
            ae_test.to_csv(folder + '/5smote+ae_test_' + ae_version + '_' + str(fold) + '.csv', index=False, encoding='utf-8')
            y_train_smote = pd.DataFrame(y_train_smote)
            y_train_smote.to_csv(folder + '/5smote+ae_train_ans_' + ae_version + '_' + str(fold) + '.csv', index=False, encoding='utf-8')

            # cluster+ae
            x_train_smote, y_train_smote = data_resample('cluster', x_train, y_train)
            ae_train, ae_test = train_ae(ae_full, set_size, x_train_smote, x_test)
            auc6 = run_model(model, ae_train, ae_test, y_train_smote, y_test)
            cluster_ae.append(auc6)
            ae_train = pd.DataFrame(ae_train)
            ae_test = pd.DataFrame(ae_test)
            ae_train.to_csv(folder + '/6cluster+ae_train_' + ae_version + '_' + str(fold) + '.csv', index=False, encoding='utf-8')
            ae_test.to_csv(folder + '/6cluster+ae_test_' + ae_version + '_' + str(fold) + '.csv', index=False, encoding='utf-8')
            y_train_smote = pd.DataFrame(y_train_smote)
            y_train_smote.to_csv(folder + '/6cluster+ae_train_ans_' + ae_version + '_' + str(fold) + '.csv', index=False, encoding='utf-8')

            # smotenn+ae
            x_train_smote, y_train_smote = data_resample('smotenn', x_train, y_train)
            ae_train, ae_test = train_ae(ae_full, set_size, x_train_smote, x_test)
            auc7 = run_model(model, ae_train, ae_test, y_train_smote, y_test)
            smotenn_ae.append(auc7)
            ae_train = pd.DataFrame(ae_train)
            ae_test = pd.DataFrame(ae_test)
            ae_train.to_csv(folder + '/7smotenn+ae_train_' + ae_version + '_' + str(fold) + '.csv', index=False, encoding='utf-8')
            ae_test.to_csv(folder + '/7smotenn+ae_test_' + ae_version + '_' + str(fold) + '.csv', index=False, encoding='utf-8')
            y_train_smote = pd.DataFrame(y_train_smote)
            y_train_smote.to_csv(folder + '/7smotenn+ae_train_ans_' + ae_version + '_' + str(fold) + '.csv', index=False, encoding='utf-8')

            print(dataset)

        fusion = True

        f_ae = []
        f_ae_smote = []; f_ae_cluster = []; f_ae_smotenn = []
        f_smote_ae = []; f_cluster_ae = []; f_smotenn_ae = []

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        for fold, (train_index, test_index) in enumerate(kf.split(x), start=1):

            x_train, x_test = x.iloc[train_index], x.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            y_train = pd.DataFrame(y_train)
            y_test = pd.DataFrame(y_test)

            # ae>fusion
            ae_train, ae_test = train_ae(ae_full, set_size, x_train, x_test)
            fusion_train, fusion_test = feature_fusion(fusion, x_train, x_test, ae_train, ae_test)
            auc1 = run_model(model, fusion_train, fusion_test, y_train, y_test)
            f_ae.append(auc1)
            fusion_train = pd.DataFrame(fusion_train)
            fusion_test = pd.DataFrame(fusion_test)
            fusion_train.to_csv(folder + '/fusion_ae_train_' + ae_version + '_' + str(fold) + '.csv', index=False, encoding='utf-8')
            fusion_test.to_csv(folder + '/fusion_ae_test_' + ae_version + '_' + str(fold) + '.csv', index=False, encoding='utf-8')

            # ae>fusion+smote
            ae_x_train_smote, ae_y_train_smote = data_resample('smote', fusion_train, y_train)
            auc2 = run_model(model, ae_x_train_smote, fusion_test, ae_y_train_smote, y_test)
            f_ae_smote.append(auc2)
            ae_x_train_smote = pd.DataFrame(ae_x_train_smote)
            ae_y_train_smote = pd.DataFrame(ae_y_train_smote)
            ae_x_train_smote.to_csv(folder + '/fusion_ae+smote_train_' + ae_version + '_' + str(fold) + '.csv', index=False, encoding='utf-8')
            ae_y_train_smote.to_csv(folder + '/fusion_ae+smote_train_ans_' + ae_version + '_' + str(fold) + '.csv', index=False, encoding='utf-8')

            # ae>fusion+cluster
            ae_x_train_smote, ae_y_train_smote = data_resample('cluster', fusion_train, y_train)
            auc3 = run_model(model, ae_x_train_smote, fusion_test, ae_y_train_smote, y_test)
            f_ae_cluster.append(auc3)
            ae_x_train_smote = pd.DataFrame(ae_x_train_smote)
            ae_y_train_smote = pd.DataFrame(ae_y_train_smote)
            ae_x_train_smote.to_csv(folder + '/fusion_ae+cluster_train_' + ae_version + '_' + str(fold) + '.csv', index=False, encoding='utf-8')
            ae_y_train_smote.to_csv(folder + '/fusion_ae+cluster_train_ans_' + ae_version + '_' + str(fold) + '.csv', index=False, encoding='utf-8')

            # ae>fusion+smotenn
            ae_x_train_smote, ae_y_train_smote = data_resample('smotenn', fusion_train, y_train)
            auc4 = run_model(model, ae_x_train_smote, fusion_test, ae_y_train_smote, y_test)
            f_ae_smotenn.append(auc4)
            ae_x_train_smote = pd.DataFrame(ae_x_train_smote)
            ae_y_train_smote = pd.DataFrame(ae_y_train_smote)
            ae_x_train_smote.to_csv(folder + '/fusion_ae+smotenn_train_' + ae_version + '_' + str(fold) + '.csv', index=False, encoding='utf-8')
            ae_y_train_smote.to_csv(folder + '/fusion_ae+smotenn_train_ans_' + ae_version + '_' + str(fold) + '.csv', index=False, encoding='utf-8')

            # smote+ae>fusion
            x_train_smote, y_train_smote = data_resample('smote', x_train, y_train)
            ae_train, ae_test = train_ae(ae_full, set_size, x_train_smote, x_test)
            fusion_train, fusion_test = feature_fusion(fusion, x_train_smote, x_test, ae_train, ae_test)
            auc5 = run_model(model, fusion_train, fusion_test, y_train_smote, y_test)
            f_smote_ae.append(auc5)
            fusion_train = pd.DataFrame(fusion_train)
            fusion_test = pd.DataFrame(fusion_test)
            fusion_train.to_csv(folder + '/fusion_smote+ae_train_' + ae_version + '_' + str(fold) + '.csv', index=False, encoding='utf-8')
            fusion_test.to_csv(folder + '/fusion_smote+ae_test_' + ae_version + '_' + str(fold) + '.csv', index=False, encoding='utf-8') 
            y_train_smote = pd.DataFrame(y_train_smote)
            y_train_smote.to_csv(folder + '/fusion_smote+ae_train_ans_' + ae_version + '_' + str(fold) + '.csv', index=False, encoding='utf-8') 

            # cluster+ae>fusion
            x_train_smote, y_train_smote = data_resample('cluster', x_train, y_train)
            ae_train, ae_test = train_ae(ae_full, set_size, x_train_smote, x_test)
            fusion_train, fusion_test = feature_fusion(fusion, x_train_smote, x_test, ae_train, ae_test)
            auc6 = run_model(model, fusion_train, fusion_test, y_train_smote, y_test)
            f_cluster_ae.append(auc6)
            fusion_train = pd.DataFrame(fusion_train)
            fusion_test = pd.DataFrame(fusion_test)
            fusion_train.to_csv(folder + '/fusion_cluster+ae_train_' + ae_version + '_' + str(fold) + '.csv', index=False, encoding='utf-8')
            fusion_test.to_csv(folder + '/fusion_cluster+ae_test_' + ae_version + '_' + str(fold) + '.csv', index=False, encoding='utf-8')
            y_train_smote = pd.DataFrame(y_train_smote)
            y_train_smote.to_csv(folder + '/fusion_cluster+ae_train_ans_' + ae_version + '_' + str(fold) + '.csv', index=False, encoding='utf-8') 

            # smotenn+ae>fusion
            x_train_smote, y_train_smote = data_resample('smotenn', x_train, y_train)
            ae_train, ae_test = train_ae(ae_full, set_size, x_train_smote, x_test)
            fusion_train, fusion_test = feature_fusion(fusion, x_train_smote, x_test, ae_train, ae_test)
            auc7 = run_model(model, fusion_train, fusion_test, y_train_smote, y_test)
            f_smotenn_ae.append(auc7)
            fusion_train = pd.DataFrame(fusion_train)
            fusion_test = pd.DataFrame(fusion_test)
            fusion_train.to_csv(folder + '/fusion_smotenn+ae_train_' + ae_version + '_' + str(fold) + '.csv', index=False, encoding='utf-8')
            fusion_test.to_csv(folder + '/fusion_smotenn+ae_test_' + ae_version + '_' + str(fold) + '.csv', index=False, encoding='utf-8')
            y_train_smote = pd.DataFrame(y_train_smote)
            y_train_smote.to_csv(folder + '/fusion_smotenn+ae_train_ans_' + ae_version + '_' + str(fold) + '.csv', index=False, encoding='utf-8') 

        new = [dataset, model, ae_version,
                round(np.mean(ae), 4), round(np.mean(ae_smote), 4), round(np.mean(smote_ae), 4), 
                round(np.mean(ae_cluster), 4), round(np.mean(cluster_ae), 4), round(np.mean(ae_smotenn), 4), round(np.mean(smotenn_ae), 4),
                round(np.mean(f_ae), 4), round(np.mean(f_ae_smote), 4), round(np.mean(f_smote_ae), 4), 
                round(np.mean(f_ae_cluster), 4), round(np.mean(f_cluster_ae), 4), round(np.mean(f_ae_smotenn), 4), round(np.mean(f_smotenn_ae), 4),
                start, datetime.now()]

        with open('result/study2' + result_file, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(new)
        K.clear_session()
        gc.collect()