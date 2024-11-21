from library.common import *
from library.functions import *
from library.dae import *

def train_ae(ver, size, x_train, x_test):
    if ver == 'dae_210':
        x_train_encoded, x_test_encoded = train_dae_210(x_train, x_test, size)
    elif ver == 'dae_220':
        x_train_encoded, x_test_encoded = train_dae_220(x_train, x_test, size)
    elif ver == 'dae_230':
        x_train_encoded, x_test_encoded = train_dae_230(x_train, x_test, size)
    elif ver == 'dae_240':
        x_train_encoded, x_test_encoded = train_dae_240(x_train, x_test, size)
    elif ver == 'dae_310':
        x_train_encoded, x_test_encoded = train_dae_310(x_train, x_test, size)
    elif ver == 'dae_320':
        x_train_encoded, x_test_encoded = train_dae_320(x_train, x_test, size)
    elif ver == 'dae_330':
        x_train_encoded, x_test_encoded = train_dae_330(x_train, x_test, size)
    elif ver == 'dae_410':
        x_train_encoded, x_test_encoded = train_dae_410(x_train, x_test, size)
    elif ver == 'dae_420':
        x_train_encoded, x_test_encoded = train_dae_420(x_train, x_test, size)
    elif ver == 'dae_510':
        x_train_encoded, x_test_encoded = train_dae_510(x_train, x_test, size)
    elif ver == 'dae_610':
        x_train_encoded, x_test_encoded = train_dae_610(x_train, x_test, size)
    elif ver == 'dae_710':
        x_train_encoded, x_test_encoded = train_dae_710(x_train, x_test, size)
    return x_train_encoded, x_test_encoded


model = 'mlp'
set_size = 64
datasets = ['ecoli-0-6-7_vs_3-5', 'shuttle-c0-vs-c4', 'winequality-white-9_vs_4']
autoencoder = ['dae_210', 'dae_220', 'dae_230', 'dae_240', 'dae_310', 'dae_320', 'dae_330', 'dae_410', 'dae_420', 'dae_510', 'dae_610', 'dae_710']
 
result = pd.DataFrame()
for idx, dataset in enumerate(datasets):
    for i, ae_version in enumerate(autoencoder):
        ae = []; ae_smote = []; smote_ae = []

        for times in range(1,6):

            train = "{}-5-{}{}.dat".format(dataset, times, 'tra')
            test = "{}-5-{}{}.dat".format(dataset, times, 'tst')
            df_train = pd.read_csv('../dataset/keel/' + dataset + '-5-fold/' + train, delimiter=',')
            df_test = pd.read_csv('../dataset/keel/' + dataset + '-5-fold/' + test, delimiter=',')

            x_train, x_test, y_train, y_test = data_preprocess(df_train, df_test)

            # ae
            ae_train, ae_test = train_ae(ae_version, set_size, x_train, x_test)

            auc1 = run_model(model, ae_train, ae_test, y_train, y_test)
            ae.append(auc1)

            # ae+smote
            ae_x_train_smote, ae_y_train_smote = data_resample('smote', ae_train, y_train)
            auc2 = run_model(model, ae_x_train_smote, ae_test, ae_y_train_smote, y_test)
            ae_smote.append(auc2)

            # smote+ae
            x_train_smote, y_train_smote = data_resample('smote', x_train, y_train)
            ae_train, ae_test = train_ae(ae_version, set_size, x_train_smote, x_test)
            auc3 = run_model(model, ae_train, ae_test, y_train_smote, y_test)
            smote_ae.append(auc3)

        new = [dataset, model, ae_version, round(np.mean(ae), 4), round(np.mean(ae_smote), 4), round(np.mean(smote_ae), 4)]

        print(dataset)
        print(new)

        with open('result/pretest.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(new)

    