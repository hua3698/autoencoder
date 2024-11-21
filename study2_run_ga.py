
from library.common import *
from library.functions import *
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
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

model = 'svc'
autoencoder = 'ae'
ae_version = ['210']
datasets = ['heart_records']
# datasets = ['heart_records', 'bank_marketing', 'statlog', 'SPECTF', 'segmentationData', 'taiwanese_bankruptcy', 'lsvt', 'madelon', 'secom', 'pd_speech_features', 'toxicity']
# datasets = ['heart_records', 'bank_marketing', 'statlog', 'SPECTF', 'segmentationData', 'taiwanese_bankruptcy', 'lsvt', 'madelon', 'secom', 
#             'pd_speech_features', 'qsar_oral_toxicity', 'toxicity', 'ad', 'hiva_agnostic', 'christine']

# for i, ae_version in enumerate(ae_version):
#     for idx, dataset in enumerate(datasets):
#         ae = []; ae_smote = []; ae_cluster = []; ae_smotenn = []
#         smote_ae = []; cluster_ae = []; smotenn_ae = []
#         start = datetime.now()

#         for times in range(1, 6):
#             print(times)

#             # ae
#             train_file = "{}/{}/1ae_train_{}_{}.csv".format(dataset, autoencoder, ae_version, times)
#             test_file = "{}/{}/1ae_test_{}_{}.csv".format(dataset, autoencoder, ae_version, times)
#             x_train = pd.read_csv('dataset/study2/feature/' + train_file, delimiter=',')
#             x_test = pd.read_csv('dataset/study2/feature/' + test_file, delimiter=',')

#             y_train_file = "{}/{}/train_ans_{}_{}.csv".format(dataset, autoencoder, ae_version, times)
#             y_test_file = "{}/{}/test_ans_{}_{}.csv".format(dataset, autoencoder, ae_version, times)
#             y_train = pd.read_csv('dataset/study2/feature/' + y_train_file , delimiter=',')
#             y_test = pd.read_csv('dataset/study2/feature/' + y_test_file , delimiter=',')

#             X = x_train.values
#             y = y_train.values.ravel()

#             toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=X.shape[1])
#             toolbox.register("population", tools.initRepeat, list, toolbox.individual)

#             best_individual = main()

#             selected_indices = [index for index, value in enumerate(best_individual) if value == 1]
#             x_train = x_train.iloc[:, selected_indices]
#             x_test = x_test.iloc[:, selected_indices]

#             auc1 = run_model(model, x_train, x_test, y_train, y_test)
#             ae.append(auc1)

#             # ae+smote 
#             train_file = "{}/{}/2ae+smote_train_{}_{}.csv".format(dataset, autoencoder, ae_version, times)
#             test_file = "{}/{}/2ae+smote_test_{}_{}.csv".format(dataset, autoencoder, ae_version, times)
#             x_train = pd.read_csv('dataset/study2/feature/' + train_file, delimiter=',')
#             x_test = pd.read_csv('dataset/study2/feature/' + test_file, delimiter=',')

#             y_train_file = "{}/{}/2ae+smote_train_ans_{}_{}.csv".format(dataset, autoencoder, ae_version, times)
#             y_test_file = "{}/{}/test_ans_{}_{}.csv".format(dataset, autoencoder, ae_version, times)
#             y_train = pd.read_csv('dataset/study2/feature/' + y_train_file , delimiter=',')
#             y_test = pd.read_csv('dataset/study2/feature/' + y_test_file , delimiter=',')

#             X = x_train.values
#             y = y_train.values.ravel()

#             toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=X.shape[1])
#             toolbox.register("population", tools.initRepeat, list, toolbox.individual)

#             best_individual = main()

#             selected_indices = [index for index, value in enumerate(best_individual) if value == 1]
#             x_train = x_train.iloc[:, selected_indices]
#             x_test = x_test.iloc[:, selected_indices]

#             auc2 = run_model(model, x_train, x_test, y_train, y_test)
#             ae_smote.append(auc2)

#             # ae+cluster
#             train_file = "{}/{}/3ae+cluster_train_{}_{}.csv".format(dataset, autoencoder, ae_version, times)
#             test_file = "{}/{}/3ae+cluster_test_{}_{}.csv".format(dataset, autoencoder, ae_version, times)
#             x_train = pd.read_csv('dataset/study2/feature/' + train_file, delimiter=',')
#             x_test = pd.read_csv('dataset/study2/feature/' + test_file, delimiter=',')

#             y_train_file = "{}/{}/3ae+cluster_train_ans_{}_{}.csv".format(dataset, autoencoder, ae_version, times)
#             y_test_file = "{}/{}/test_ans_{}_{}.csv".format(dataset, autoencoder, ae_version, times)
#             y_train = pd.read_csv('dataset/study2/feature/' + y_train_file , delimiter=',')
#             y_test = pd.read_csv('dataset/study2/feature/' + y_test_file , delimiter=',')

#             X = x_train.values
#             y = y_train.values.ravel()

#             toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=X.shape[1])
#             toolbox.register("population", tools.initRepeat, list, toolbox.individual)

#             best_individual = main()

#             selected_indices = [index for index, value in enumerate(best_individual) if value == 1]
#             x_train = x_train.iloc[:, selected_indices]
#             x_test = x_test.iloc[:, selected_indices]

#             auc3 = run_model(model, x_train, x_test, y_train, y_test)
#             ae_cluster.append(auc3)

#             # ae+smotenn
#             train_file = "{}/{}/4ae+smotenn_train_{}_{}.csv".format(dataset, autoencoder, ae_version, times)
#             test_file = "{}/{}/4ae+smotenn_test_{}_{}.csv".format(dataset, autoencoder, ae_version, times)
#             x_train = pd.read_csv('dataset/study2/feature/' + train_file, delimiter=',')
#             x_test = pd.read_csv('dataset/study2/feature/' + test_file, delimiter=',')

#             y_train_file = "{}/{}/4ae+smotenn_train_ans_{}_{}.csv".format(dataset, autoencoder, ae_version, times)
#             y_test_file = "{}/{}/test_ans_{}_{}.csv".format(dataset, autoencoder, ae_version, times)
#             y_train = pd.read_csv('dataset/study2/feature/' + y_train_file , delimiter=',')
#             y_test = pd.read_csv('dataset/study2/feature/' + y_test_file , delimiter=',')

#             X = x_train.values
#             y = y_train.values.ravel()

#             toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=X.shape[1])
#             toolbox.register("population", tools.initRepeat, list, toolbox.individual)

#             best_individual = main()

#             selected_indices = [index for index, value in enumerate(best_individual) if value == 1]
#             x_train = x_train.iloc[:, selected_indices]
#             x_test = x_test.iloc[:, selected_indices]

#             auc4 = run_model(model, x_train, x_test, y_train, y_test)
#             ae_smotenn.append(auc4)

#             # smote+ae 
#             train_file = "{}/{}/5smote+ae_train_{}_{}.csv".format(dataset, autoencoder, ae_version, times)
#             test_file = "{}/{}/5smote+ae_test_{}_{}.csv".format(dataset, autoencoder, ae_version, times)
#             x_train = pd.read_csv('dataset/study2/feature/' + train_file, delimiter=',')
#             x_test = pd.read_csv('dataset/study2/feature/' + test_file, delimiter=',')

#             y_train_file = "{}/{}/5smote+ae_train_ans_{}_{}.csv".format(dataset, autoencoder, ae_version, times)
#             y_test_file = "{}/{}/test_ans_{}_{}.csv".format(dataset, autoencoder, ae_version, times)
#             y_train = pd.read_csv('dataset/study2/feature/' + y_train_file , delimiter=',')
#             y_test = pd.read_csv('dataset/study2/feature/' + y_test_file , delimiter=',')

#             X = x_train.values
#             y = y_train.values.ravel()

#             toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=X.shape[1])
#             toolbox.register("population", tools.initRepeat, list, toolbox.individual)

#             best_individual = main()

#             selected_indices = [index for index, value in enumerate(best_individual) if value == 1]
#             x_train = x_train.iloc[:, selected_indices]
#             x_test = x_test.iloc[:, selected_indices]

#             auc5 = run_model(model, x_train, x_test, y_train, y_test)
#             smote_ae.append(auc5)

#             # cluster+ae
#             train_file = "{}/{}/6cluster+ae_train_{}_{}.csv".format(dataset, autoencoder, ae_version, times)
#             test_file = "{}/{}/6cluster+ae_test_{}_{}.csv".format(dataset, autoencoder, ae_version, times)
#             x_train = pd.read_csv('dataset/study2/feature/' + train_file, delimiter=',')
#             x_test = pd.read_csv('dataset/study2/feature/' + test_file, delimiter=',')

#             y_train_file = "{}/{}/6cluster+ae_train_ans_{}_{}.csv".format(dataset, autoencoder, ae_version, times)
#             y_test_file = "{}/{}/test_ans_{}_{}.csv".format(dataset, autoencoder, ae_version, times)
#             y_train = pd.read_csv('dataset/study2/feature/' + y_train_file , delimiter=',')
#             y_test = pd.read_csv('dataset/study2/feature/' + y_test_file , delimiter=',')

#             X = x_train.values
#             y = y_train.values.ravel()

#             toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=X.shape[1])
#             toolbox.register("population", tools.initRepeat, list, toolbox.individual)

#             best_individual = main()

#             selected_indices = [index for index, value in enumerate(best_individual) if value == 1]
#             x_train = x_train.iloc[:, selected_indices]
#             x_test = x_test.iloc[:, selected_indices]

#             auc6 = run_model(model, x_train, x_test, y_train, y_test)
#             cluster_ae.append(auc6)

#             # smotenn+ae
#             train_file = "{}/{}/7smotenn+ae_train_{}_{}.csv".format(dataset, autoencoder, ae_version, times)
#             test_file = "{}/{}/7smotenn+ae_test_{}_{}.csv".format(dataset, autoencoder, ae_version, times)
#             x_train = pd.read_csv('dataset/study2/feature/' + train_file, delimiter=',')
#             x_test = pd.read_csv('dataset/study2/feature/' + test_file, delimiter=',')

#             y_train_file = "{}/{}/7smotenn+ae_train_ans_{}_{}.csv".format(dataset, autoencoder, ae_version, times)
#             y_test_file = "{}/{}/test_ans_{}_{}.csv".format(dataset, autoencoder, ae_version, times)
#             y_train = pd.read_csv('dataset/study2/feature/' + y_train_file , delimiter=',')
#             y_test = pd.read_csv('dataset/study2/feature/' + y_test_file , delimiter=',')

#             X = x_train.values
#             y = y_train.values.ravel()

#             toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=X.shape[1])
#             toolbox.register("population", tools.initRepeat, list, toolbox.individual)

#             best_individual = main()

#             selected_indices = [index for index, value in enumerate(best_individual) if value == 1]
#             x_train = x_train.iloc[:, selected_indices]
#             x_test = x_test.iloc[:, selected_indices]

#             auc7 = run_model(model, x_train, x_test, y_train, y_test)
#             smotenn_ae.append(auc7)

#         end = datetime.now()

#         new = [dataset, model, ae_version, autoencoder,
#                round(np.mean(ae), 3), round(np.mean(ae_smote), 3), round(np.mean(ae_cluster), 3), round(np.mean(ae_smotenn), 3),
#                round(np.mean(smote_ae), 3), round(np.mean(cluster_ae), 3), round(np.mean(smotenn_ae), 3), start, end]

#         with open('result/study2/ga.csv', 'a', newline='') as csvfile:
#             writer = csv.writer(csvfile, delimiter=',')
#             writer.writerow(new)

###############################################################################################################
# fusion

for i, ae_version in enumerate(ae_version):
    for idx, dataset in enumerate(datasets):
        f_ae = []
        f_ae_smote = []; f_ae_cluster = []; f_ae_smotenn = []
        f_smote_ae = []; f_cluster_ae = []; f_smotenn_ae = []
        start = datetime.now()

        for times in range(1, 6):
            print(times)

            # ae>fusion
            train_file = "{}/{}/fusion_ae_train_{}_{}.csv".format(dataset, autoencoder, ae_version, times)
            test_file = "{}/{}/fusion_ae_test_{}_{}.csv".format(dataset, autoencoder, ae_version, times)
            x_train = pd.read_csv('dataset/study2/feature/' + train_file, delimiter=',')
            x_test = pd.read_csv('dataset/study2/feature/' + test_file, delimiter=',')

            y_train_file = "{}/{}/train_ans_{}_{}.csv".format(dataset, autoencoder, ae_version, times)
            y_test_file = "{}/{}/test_ans_{}_{}.csv".format(dataset, autoencoder, ae_version, times)
            y_train = pd.read_csv('dataset/study2/feature/' + y_train_file , delimiter=',')
            y_test = pd.read_csv('dataset/study2/feature/' + y_test_file , delimiter=',')

            X = x_train.values
            y = y_train.values.ravel()

            toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=X.shape[1])
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)

            print(X.shape)
            best_individual = main()
            print(best_individual)

            selected_indices = [index for index, value in enumerate(best_individual) if value == 1]
            print(selected_indices)
            x_train = x_train.iloc[:, selected_indices]
            x_test = x_test.iloc[:, selected_indices]

            auc1 = run_model(model, x_train, x_test, y_train, y_test)
            f_ae.append(auc1)

        #     # ae>fusion+smote
        #     train_file = "{}/{}/fusion_ae+smote_train_{}_{}.csv".format(dataset, autoencoder, ae_version, times)
        #     test_file = "{}/{}/fusion_ae_test_{}_{}.csv".format(dataset, autoencoder, ae_version, times)
        #     x_train = pd.read_csv('dataset/study2/feature/' + train_file, delimiter=',')
        #     x_test = pd.read_csv('dataset/study2/feature/' + test_file, delimiter=',')

        #     y_train_file = "{}/{}/fusion_ae+smote_train_ans_{}_{}.csv".format(dataset, autoencoder, ae_version, times)
        #     y_test_file = "{}/{}/test_ans_{}_{}.csv".format(dataset, autoencoder, ae_version, times)
        #     y_train = pd.read_csv('dataset/study2/feature/' + y_train_file , delimiter=',')
        #     y_test = pd.read_csv('dataset/study2/feature/' + y_test_file , delimiter=',')

        #     X = x_train.values
        #     y = y_train.values.ravel()

        #     toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=X.shape[1])
        #     toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        #     best_individual = main()

        #     selected_indices = [index for index, value in enumerate(best_individual) if value == 1]
        #     x_train = x_train.iloc[:, selected_indices]
        #     x_test = x_test.iloc[:, selected_indices]

        #     auc1 = run_model(model, x_train, x_test, y_train, y_test)
        #     f_ae_smote.append(auc1)

        #     # ae>fusion+cluster
        #     train_file = "{}/{}/fusion_ae+cluster_train_{}_{}.csv".format(dataset, autoencoder, ae_version, times)
        #     test_file = "{}/{}/fusion_ae_test_{}_{}.csv".format(dataset, autoencoder, ae_version, times)
        #     x_train = pd.read_csv('dataset/study2/feature/' + train_file, delimiter=',')
        #     x_test = pd.read_csv('dataset/study2/feature/' + test_file, delimiter=',')

        #     y_train_file = "{}/{}/fusion_ae+cluster_train_ans_{}_{}.csv".format(dataset, autoencoder, ae_version, times)
        #     y_test_file = "{}/{}/test_ans_{}_{}.csv".format(dataset, autoencoder, ae_version, times)
        #     y_train = pd.read_csv('dataset/study2/feature/' + y_train_file , delimiter=',')
        #     y_test = pd.read_csv('dataset/study2/feature/' + y_test_file , delimiter=',')

        #     X = x_train.values
        #     y = y_train.values.ravel()

        #     toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=X.shape[1])
        #     toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        #     best_individual = main()

        #     selected_indices = [index for index, value in enumerate(best_individual) if value == 1]
        #     x_train = x_train.iloc[:, selected_indices]
        #     x_test = x_test.iloc[:, selected_indices]

        #     auc1 = run_model(model, x_train, x_test, y_train, y_test)
        #     f_ae_cluster.append(auc1)

        #     # ae>fusion+smotenn
        #     train_file = "{}/{}/fusion_ae+smotenn_train_{}_{}.csv".format(dataset, autoencoder, ae_version, times)
        #     test_file = "{}/{}/fusion_ae_test_{}_{}.csv".format(dataset, autoencoder, ae_version, times)
        #     x_train = pd.read_csv('dataset/study2/feature/' + train_file, delimiter=',')
        #     x_test = pd.read_csv('dataset/study2/feature/' + test_file, delimiter=',')

        #     y_train_file = "{}/{}/fusion_ae+smotenn_train_ans_{}_{}.csv".format(dataset, autoencoder, ae_version, times)
        #     y_test_file = "{}/{}/test_ans_{}_{}.csv".format(dataset, autoencoder, ae_version, times)
        #     y_train = pd.read_csv('dataset/study2/feature/' + y_train_file , delimiter=',')
        #     y_test = pd.read_csv('dataset/study2/feature/' + y_test_file , delimiter=',')

        #     X = x_train.values
        #     y = y_train.values.ravel()

        #     toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=X.shape[1])
        #     toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        #     best_individual = main()

        #     selected_indices = [index for index, value in enumerate(best_individual) if value == 1]
        #     x_train = x_train.iloc[:, selected_indices]
        #     x_test = x_test.iloc[:, selected_indices]

        #     auc1 = run_model(model, x_train, x_test, y_train, y_test)
        #     f_ae_smotenn.append(auc1)

        #     # smote+ae>fusion
        #     train_file = "{}/{}/fusion_smote+ae_train_{}_{}.csv".format(dataset, autoencoder, ae_version, times)
        #     test_file = "{}/{}/fusion_smote+ae_test_{}_{}.csv".format(dataset, autoencoder, ae_version, times)
        #     x_train = pd.read_csv('dataset/study2/feature/' + train_file, delimiter=',')
        #     x_test = pd.read_csv('dataset/study2/feature/' + test_file, delimiter=',')

        #     y_train_file = "{}/{}/fusion_smote+ae_train_ans_{}_{}.csv".format(dataset, autoencoder, ae_version, times)
        #     y_test_file = "{}/{}/test_ans_{}_{}.csv".format(dataset, autoencoder, ae_version, times)
        #     y_train = pd.read_csv('dataset/study2/feature/' + y_train_file , delimiter=',')
        #     y_test = pd.read_csv('dataset/study2/feature/' + y_test_file , delimiter=',')

        #     X = x_train.values
        #     y = y_train.values.ravel()

        #     toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=X.shape[1])
        #     toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        #     best_individual = main()

        #     selected_indices = [index for index, value in enumerate(best_individual) if value == 1]
        #     x_train = x_train.iloc[:, selected_indices]
        #     x_test = x_test.iloc[:, selected_indices]

        #     auc1 = run_model(model, x_train, x_test, y_train, y_test)
        #     f_smote_ae.append(auc1)

        #     # cluster+ae>fusion
        #     train_file = "{}/{}/fusion_cluster+ae_train_{}_{}.csv".format(dataset, autoencoder, ae_version, times)
        #     test_file = "{}/{}/fusion_cluster+ae_test_{}_{}.csv".format(dataset, autoencoder, ae_version, times)
        #     x_train = pd.read_csv('dataset/study2/feature/' + train_file, delimiter=',')
        #     x_test = pd.read_csv('dataset/study2/feature/' + test_file, delimiter=',')

        #     y_train_file = "{}/{}/fusion_cluster+ae_train_ans_{}_{}.csv".format(dataset, autoencoder, ae_version, times)
        #     y_test_file = "{}/{}/test_ans_{}_{}.csv".format(dataset, autoencoder, ae_version, times)
        #     y_train = pd.read_csv('dataset/study2/feature/' + y_train_file , delimiter=',')
        #     y_test = pd.read_csv('dataset/study2/feature/' + y_test_file , delimiter=',')

        #     X = x_train.values
        #     y = y_train.values.ravel()

        #     toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=X.shape[1])
        #     toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        #     best_individual = main()

        #     selected_indices = [index for index, value in enumerate(best_individual) if value == 1]
        #     x_train = x_train.iloc[:, selected_indices]
        #     x_test = x_test.iloc[:, selected_indices]

        #     auc1 = run_model(model, x_train, x_test, y_train, y_test)
        #     f_cluster_ae.append(auc1)

        #     # smotenn+ae>fusion
        #     train_file = "{}/{}/fusion_smotenn+ae_train_{}_{}.csv".format(dataset, autoencoder, ae_version, times)
        #     test_file = "{}/{}/fusion_smotenn+ae_test_{}_{}.csv".format(dataset, autoencoder, ae_version, times)
        #     x_train = pd.read_csv('dataset/study2/feature/' + train_file, delimiter=',')
        #     x_test = pd.read_csv('dataset/study2/feature/' + test_file, delimiter=',')

        #     y_train_file = "{}/{}/fusion_smotenn+ae_train_ans_{}_{}.csv".format(dataset, autoencoder, ae_version, times)
        #     y_test_file = "{}/{}/test_ans_{}_{}.csv".format(dataset, autoencoder, ae_version, times)
        #     y_train = pd.read_csv('dataset/study2/feature/' + y_train_file , delimiter=',')
        #     y_test = pd.read_csv('dataset/study2/feature/' + y_test_file , delimiter=',')

        #     X = x_train.values
        #     y = y_train.values.ravel()

        #     toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=X.shape[1])
        #     toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        #     best_individual = main()

        #     selected_indices = [index for index, value in enumerate(best_individual) if value == 1]
        #     x_train = x_train.iloc[:, selected_indices]
        #     x_test = x_test.iloc[:, selected_indices]

        #     auc1 = run_model(model, x_train, x_test, y_train, y_test)
        #     f_smotenn_ae.append(auc1)

        # end = datetime.now()

        # new = [dataset, model, ae_version, autoencoder, 'fusion',
        #        round(np.mean(f_ae), 3), round(np.mean(f_ae_smote), 3), round(np.mean(f_ae_cluster), 3), round(np.mean(f_ae_smotenn), 3),
        #        round(np.mean(f_smote_ae), 3), round(np.mean(f_cluster_ae), 3), round(np.mean(f_smotenn_ae), 3),
        #        start, end]

        # with open('result/study2/ga.csv', 'a', newline='') as csvfile:
        #     writer = csv.writer(csvfile, delimiter=',')
        #     writer.writerow(new)