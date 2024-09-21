from library.common import *
from library.ae import *

def train_ae(ver, size, x_train, x_test):
  if ver == 'ae_210':
    x_train_encoded, x_test_encoded = train_ae_210(x_train, x_test, size)

  elif ver == 'ae_220':
    x_train_encoded, x_test_encoded = train_ae_220(x_train, x_test, size)

  elif ver == 'ae_230':
    x_train_encoded, x_test_encoded = train_ae_230(x_train, x_test, size)

  elif ver == 'ae_240':
    x_train_encoded, x_test_encoded = train_ae_240(x_train, x_test, size)

  return x_train_encoded, x_test_encoded


result = pd.DataFrame()
ae_version = 'ae_210'
model = 'svc'
set_size = 32
datasets = ['yeast5', 'winequality-red-8_vs_6', 'winequality-red-3_vs_5', 'poker-8-9_vs_5', 'kddcup-land_vs_satan', 
            'abalone-21_vs_8', 'abalone-20_vs_8-9-10', 'poker-8_vs_6']

for idx, dataset in enumerate(datasets):

  ae = []; ae_tn = 0; ae_fp = 0; ae_fn = 0; ae_tp = 0
  ae_smote = []; ae_smote_tn = 0; ae_smote_fp = 0; ae_smote_fn = 0; ae_smote_tp = 0
  smote_ae = []; smote_ae_tn = 0; smote_ae_fp = 0; smote_ae_fn = 0; smote_ae_tp = 0
  ae_time = 0 ; ae_smote_time = 0 ; smote_ae_time = 0

  for times in range(1,6):

    train = "{}-5-{}{}.dat".format(dataset, times, 'tra')
    test = "{}-5-{}{}.dat".format(dataset, times, 'tst')
    df_train = pd.read_csv('../dataset/keel/' + dataset + '-5-fold/' + train, delimiter=',')
    df_test = pd.read_csv('../dataset/keel/' + dataset + '-5-fold/' + test, delimiter=',')

    x_train, x_test, y_train, y_test = data_preprocess(df_train, df_test)

    # ae
    time1 = time.process_time()
    ae_train, ae_test = train_ae(ae_version, set_size, x_train, x_test)
    time2 = time.process_time()
    tn, fp, fn, tp, auc1 = run_model(model, ae_train, ae_test, y_train, y_test)
    ae.append(auc1); ae_tn += tn; ae_fp += fp; ae_fn += fn; ae_tp += tp
    ae_time += (time2 - time1)

    # ae+smote
    time1 = time.process_time()
    ae_x_train_smote, ae_y_train_smote = data_resample(ae_train, y_train)
    time2 = time.process_time()
    tn, fp, fn, tp, auc2 = run_model(model, ae_x_train_smote, ae_test, ae_y_train_smote, y_test)
    ae_smote.append(auc2); ae_smote_tn += tn; ae_smote_fp += fp; ae_smote_fn += fn; ae_smote_tp += tp
    ae_smote_time = ae_smote_time + (time2 - time1) + ae_time

    # smote+ae
    time1 = time.process_time()
    x_train_smote, y_train_smote = data_resample(x_train, y_train)
    ae_train, ae_test = train_ae(ae_version, set_size, x_train_smote, x_test)
    time2 = time.process_time()
    tn, fp, fn, tp, auc3 = run_model(model, ae_train, ae_test, y_train_smote, y_test)
    smote_ae.append(auc3); smote_ae_tn += tn; smote_ae_fp += fp; smote_ae_fn += fn; smote_ae_tp += tp
    smote_ae_time += (time2 - time1)

  ae_matrix = [ae_tn, ae_fp, ae_fn, ae_tp]
  ae_smote_matrix = [ae_smote_tn, ae_smote_fp, ae_smote_fn, ae_smote_tp]
  smote_ae_matrix = [smote_ae_tn, smote_ae_fp, smote_ae_fn, smote_ae_tp]
  ae_matrix = ', '.join(map(str, ae_matrix))
  ae_smote_matrix = ', '.join(map(str, ae_smote_matrix))
  smote_ae_matrix = ', '.join(map(str, smote_ae_matrix))


  new = [dataset, model, ae_version, np.mean(ae), ae_time, np.mean(ae_smote), np.mean(smote_ae), ae_smote_time, smote_ae_time,
         ae_matrix, ae_smote_matrix, smote_ae_matrix]

  with open('result/keel2.csv', 'a', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(new)