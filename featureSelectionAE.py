from sklearn.model_selection import KFold
from library.common import *
from library.functions import *

model = 'svc'
ae_version = ['vae_210']
# ae_version = ['vae_210', 'vae_220', 'vae_230', 'vae_240']
features = ['texture_mean', 'perimeter_mean', 'area_mean', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst', 'symmetry_worst']
feature_count = len(features)
dataset = 'breast_cancer'
result_file = 'feature_selection.csv'

# 讀取資料
df = pd.read_csv('dataset/breast_cancer.csv', delimiter=',')
x = df.loc[:, df.columns.isin(features)]
# x = df.drop(columns=['id', 'diagnosis'])
y = (df['diagnosis'] == "M").astype(int)

for i, ae_version in enumerate(ae_version):

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    ae = []
    ae_smote = []; ae_cluster = []; ae_smotenn = []
    smote_ae = []; cluster_ae = []; smotenn_ae = []

    f_ae = []
    f_ae_smote = []; f_ae_cluster = []; f_ae_smotenn = []
    f_smote_ae = []; f_cluster_ae = []; f_smotenn_ae = []

    # 進行 5 折 KFOLD 拆分
    for train_index, test_index in kf.split(x):
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # ae
        ae_train, ae_test = train_ae(ae_version, set_size, x_train, x_test)
        auc1 = run_model(model, ae_train, ae_test, y_train, y_test)
        ae.append(auc1)

        # ae+smote
        ae_x_train_smote, ae_y_train_smote = data_resample('smote', ae_train, y_train)
        auc2 = run_model(model, ae_x_train_smote, ae_test, ae_y_train_smote, y_test)
        ae_smote.append(auc2)
        ae_smote_ratio = f"{np.sum(ae_y_train_smote == 0)}:{np.sum(ae_y_train_smote == 1)}"

        # ae+cluster
        ae_x_train_smote, ae_y_train_smote = data_resample('cluster', ae_train, y_train)
        auc3 = run_model(model, ae_x_train_smote, ae_test, ae_y_train_smote, y_test)
        ae_cluster.append(auc3)
        ae_cluster_ratio = f"{np.sum(ae_y_train_smote == 0)}:{np.sum(ae_y_train_smote == 1)}"

        # ae+smotenn
        ae_x_train_smote, ae_y_train_smote = data_resample('smotenn', ae_train, y_train)
        ae_smotenn_ratio = f"{np.sum(ae_y_train_smote == 0)}:{np.sum(ae_y_train_smote == 1)}"
        if np.sum(ae_y_train_smote == 0) != 0 and np.sum(ae_y_train_smote == 1) != 0:
            auc4 = run_model(model, ae_x_train_smote, ae_test, ae_y_train_smote, y_test)
            ae_smotenn.append(auc4)

        # smote+ae
        x_train_smote, y_train_smote = data_resample('smote', x_train, y_train)
        ae_train, ae_test = train_ae(ae_version, set_size, x_train_smote, x_test)
        auc5 = run_model(model, ae_train, ae_test, y_train_smote, y_test)
        smote_ae.append(auc5)
        smote_ae_ratio = f"{np.sum(y_train_smote == 0)}:{np.sum(y_train_smote == 1)}"

        # cluster+ae
        x_train_smote, y_train_smote = data_resample('cluster', x_train, y_train)
        ae_train, ae_test = train_ae(ae_version, set_size, x_train_smote, x_test)
        auc6 = run_model(model, ae_train, ae_test, y_train_smote, y_test)
        cluster_ae.append(auc6)
        cluster_ae_ratio = f"{np.sum(y_train_smote == 0)}:{np.sum(y_train_smote == 1)}"

        # smotenn+ae
        x_train_smote, y_train_smote = data_resample('smotenn', x_train, y_train)
        ae_train, ae_test = train_ae(ae_version, set_size, x_train_smote, x_test)
        auc7 = run_model(model, ae_train, ae_test, y_train_smote, y_test)
        smotenn_ae.append(auc7)
        smotenn_ae_ratio = f"{np.sum(y_train_smote == 0)}:{np.sum(y_train_smote == 1)}"

        fusion = True

        # ae>fusion
        ae_train, ae_test = train_ae(ae_version, set_size, x_train, x_test)
        fusion_train, fusion_test = feature_fusion(fusion, x_train, x_test, ae_train, ae_test)
        auc1 = run_model(model, fusion_train, fusion_test, y_train, y_test)
        f_ae.append(auc1)

        # ae>fusion+smote
        ae_x_train_smote, ae_y_train_smote = data_resample('smote', fusion_train, y_train)
        auc2 = run_model(model, ae_x_train_smote, fusion_test, ae_y_train_smote, y_test)
        f_ae_smote.append(auc2)

        # ae>fusion+cluster
        ae_x_train_smote, ae_y_train_smote = data_resample('cluster', fusion_train, y_train)
        auc3 = run_model(model, ae_x_train_smote, fusion_test, ae_y_train_smote, y_test)
        f_ae_cluster.append(auc3)

        # ae>fusion+smotenn
        ae_x_train_smote, ae_y_train_smote = data_resample('smotenn', fusion_train, y_train)
        auc4 = run_model(model, ae_x_train_smote, fusion_test, ae_y_train_smote, y_test)
        f_ae_smotenn.append(auc4)

        # smote+ae>fusion
        x_train_smote, y_train_smote = data_resample('smote', x_train, y_train)
        ae_train, ae_test = train_ae(ae_version, set_size, x_train_smote, x_test)
        fusion_train, fusion_test = feature_fusion(fusion, x_train_smote, x_test, ae_train, ae_test)
        auc5 = run_model(model, fusion_train, fusion_test, y_train_smote, y_test)
        f_smote_ae.append(auc5)

        # cluster+ae>fusion
        x_train_smote, y_train_smote = data_resample('cluster', x_train, y_train)
        ae_train, ae_test = train_ae(ae_version, set_size, x_train_smote, x_test)
        fusion_train, fusion_test = feature_fusion(fusion, x_train_smote, x_test, ae_train, ae_test)
        auc6 = run_model(model, fusion_train, fusion_test, y_train_smote, y_test)
        f_cluster_ae.append(auc6)

        # smotenn+ae>fusion
        x_train_smote, y_train_smote = data_resample('smotenn', x_train, y_train)
        ae_train, ae_test = train_ae(ae_version, set_size, x_train_smote, x_test)
        fusion_train, fusion_test = feature_fusion(fusion, x_train_smote, x_test, ae_train, ae_test)
        auc7 = run_model(model, fusion_train, fusion_test, y_train_smote, y_test)
        f_smotenn_ae.append(auc7)

    new = [dataset, model, ae_version, feature_count,
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
