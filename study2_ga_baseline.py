
from library.common import *
from library.functions import *
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from deap import base, creator, tools
import random

# 評估函數
def evaluate(individual):
    global X, y  # 明確宣告使用全域變數
    individual = np.array(individual)
    if sum(individual) == 0:
        return 0,  # 防止所有特徵被選擇為 0
    X_subset = X[:, individual.astype(bool)]  # 確保 X 是 numpy 陣列
    model = LogisticRegression(solver='liblinear')
    scores = cross_val_score(model, X_subset, y, cv=5, scoring='accuracy')
    return np.mean(scores),

# 定義遺傳演算法的設定
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.01)  # 使用 indpb 參數
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

# GA參數設定
NGEN = 20
POP_SIZE = 30
CXPB = 0.9
MUTPB = 0.01

# 主程式
def main():
    population = toolbox.population(n=POP_SIZE)

    # 計算初始適應度
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    for gen in range(NGEN):
        # print(f"Generation {gen + 1}")

        # 選擇
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        # 交叉
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # 突變
        for mutant in offspring:
            if random.random() < MUTPB:  # 檢查是否進行突變
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # 評估適應度
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # 更新整個種群
        population[:] = offspring

        # 找出最佳個體
        fits = [ind.fitness.values[0] for ind in population]
        length = len(population)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5
        # print(f"  Min: {min(fits)}, Max: {max(fits)}, Avg: {mean}, Std: {std}")

    # 返回最佳個體
    best_ind = tools.selBest(population, 1)[0]
    # print(f"Best individual is {best_ind}, {best_ind.fitness.values}")
    return best_ind


def process_train_test(dataset, ae_version, times):

    train_file = "{}/1ae_train_{}_{}.csv".format(dataset, ae_version, times)
    test_file = "{}/1ae_test_{}_{}.csv".format(dataset, ae_version, times)
    x_train = pd.read_csv('dataset/study2/feature/' + train_file, delimiter=',')
    x_test = pd.read_csv('dataset/study2/feature/' + test_file, delimiter=',')

    y_train_file = "{}/train_ans_{}_{}.csv".format(dataset, ae_version, times)
    y_test_file = "{}/test_ans_{}_{}.csv".format(dataset, ae_version, times)
    y_train = pd.read_csv('dataset/study2/feature/' + y_train_file , delimiter=',')
    y_test = pd.read_csv('dataset/study2/feature/' + y_test_file , delimiter=',')

    return x_train, x_test, y_train, y_test

datasets = ['heart_records', 'bank_marketing', 'statlog', 'SPECTF', 'segmentationData', 'taiwanese_bankruptcy', 'lsvt', 'madelon', 'secom', 
            'pd_speech_features', 'qsar_oral_toxicity', 'toxicity', 'ad', 'hiva_agnostic', 'christine']

for idx, dataset in enumerate(datasets):
    print(dataset)
    start = datetime.now()
    svm_auc = [];  knn_auc = []; cart_auc = []; mlp_auc = []; xgb_auc = []
    smote_svm_auc = [];  smote_knn_auc = []; smote_cart_auc = []; smote_mlp_auc = []; smote_xgb_auc = []
    cluster_svm_auc = []; cluster_knn_auc = []; cluster_cart_auc = []; cluster_mlp_auc = []; cluster_xgb_auc = []
    smoteenn_svm_auc = []; smoteenn_knn_auc = []; smoteenn_cart_auc = []; smoteenn_mlp_auc = []; smoteenn_xgb_auc = []

    df = pd.read_csv('dataset/study2/' + dataset + '.csv', delimiter=',')
    df_x = df.drop(columns=['Class'])
    labelencoder = LabelEncoder()
    df_y = pd.Series(labelencoder.fit_transform(df['Class']))

    string_columns = df_x.select_dtypes(include=['object']).columns
    for col in string_columns:
        labelencoder = LabelEncoder()
        df_x[col] = labelencoder.fit_transform(df_x[col])

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_index, test_index) in enumerate(kf.split(df), start=1):
        print(fold)
        x_train, x_test = df_x.iloc[train_index], df_x.iloc[test_index]
        y_train, y_test = df_y.iloc[train_index], df_y.iloc[test_index]

        X = x_train.values
        y = y_train.values.ravel()

        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=X.shape[1])
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        print('tool')

        best_individual = main()

        print('main')

        selected_indices = [index for index, value in enumerate(best_individual) if value == 1]
        x_train = x_train.iloc[:, selected_indices]
        x_test = x_test.iloc[:, selected_indices]

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
        print('a')

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
        print('b')

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
        print('c')

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

        end = datetime.now()


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
            round(np.mean(xgb_auc), 3), round(np.mean(smote_xgb_auc), 3), round(np.mean(cluster_xgb_auc), 3), round(np.mean(smoteenn_xgb_auc), 3),
            start, end]

    with open('result/study2/ga_baseline.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(new)

