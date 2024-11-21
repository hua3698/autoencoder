from library.common import *
from library.functions import *
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from deap import base, creator, tools
import random

# data = pd.read_csv("dataset/breast_cancer.csv")
# X = data.drop(columns=['id', 'diagnosis']).values
# y = data['diagnosis'].map({'M': 1, 'B': 0}).values 

# 定義適應度函數
def evaluate(individual):
    global X, y  # 明確宣告使用全域變數
    individual = np.array(individual)
    if sum(individual) == 0:
        return 0,  # 防止所有特徵被選擇為 0
    X_subset = X[:, individual.astype(bool)]
    model = LogisticRegression(solver='liblinear')
    scores = cross_val_score(model, X_subset, y, cv=5, scoring='accuracy')
    return np.mean(scores),

# 定義遺傳演算法的設定
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=X.shape[1])
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.01)  # 使用 indpb 參數
toolbox.register("select", tools.selTournament, tournsize=3)

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
        print(f"Generation {gen + 1}")

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
        print(f"  Min: {min(fits)}, Max: {max(fits)}, Avg: {mean}, Std: {std}")

    # 返回最佳個體
    best_ind = tools.selBest(population, 1)[0]
    print(f"Best individual is {best_ind}, {best_ind.fitness.values}")
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


model = 'svc'
ae_version = ['ae_210', 'ae_220', 'ae_230', 'ae_240']
datasets = ['sonar', 'SPECTF', 'MUSK_Clean1']


for idx, dataset in enumerate(datasets):
    for i, ae_version in enumerate(ae_version):
        ae = []
        ae_smote = []; ae_cluster = []; ae_smotenn = []
        smote_ae = []; cluster_ae = []; smotenn_ae = []
        for times in range(1, 6):

            x_train, x_test, y_train, y_test = process_train_test(dataset, ae_version, times)

            X = x_train.values
            y = y_train.values.ravel()

            toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=X.shape[1])
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)

            best_individual = main()

            selected_indices = [index for index, value in enumerate(best_individual) if value == 1]
            x_train = x_train.iloc[:, selected_indices]
            x_test = x_test.iloc[:, selected_indices]

            auc1 = run_model(model, x_train, x_test, y_train, y_test)
            ae.append(auc1)
