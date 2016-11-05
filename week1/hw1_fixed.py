import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler

eps = 1e-3


def getPredict(theta, x):
    # Функція передбачає ціну оренди квартири, використовуючи лінійну регресію
    # з ознаками (фічами) X та вагами theta

    # ====================== ВАШ КОД ТУТ ======================
    # Інструкції: Завершіть наступний код, щоб робити передбачення,
    #               використовуючи лінійну регресію. Ваш код повинен
    #    	присвоювати передбачені ціни змінній p, яку повертає
    #		ця функція
    #		(Орієнтовно 1 строчка коду)
    # =========================================================================
    pass


def getError(h, y):
    # Функція рахує помилку передбачення, якщо передбачені значення - h,
    # а реальні значення - y

    # ====================== ВАШ КОД ТУТ ======================
    # Інструкції: Завершіть наступний код, щоб розраховувати помилку
    #               передбачення.
    #		Ваш код повинен	присвоювати розраховану похибку
    #		змінній e, яку повертає	ця функція
    #		(Орієнтовно 1 строчка коду)
    # =========================================================================
    pass


def getGradient(theta, x, y):
    # Функція рахує градієнт (часткові похідні) помилки, які необхідні
    # для здійснення "кроку" навчання алгоритму

    # ====================== ВАШ КОД ТУТ ======================
    # Інструкції: Завершіть наступний код, щоб розрахувати градієнт
    #		функції помилки найменших квадратів (least squares)
    #		для лінійної регресії з вхідними даними x та вагами theta.
    #		Ваш код повинен	присвоювати розрахований градієнт
    #		змінній g, яку повертає	ця функція
    #		(орієнтовно 2 строчки коду)
    # =========================================================================

    pass


def getGradientDescentStep(theta, alpha, x, y):
    # Функція робить один "крок" градієнтного спуску і повертає значення
    # оновлених вагів next_theta

    # ====================== ВАШ КОД ТУТ ======================
    # Інструкції: Завершіть наступний код, щоб розрахувати нові значення вагів next_theta
    #		після здійснення одного кроку градієнтного спуску з коефіцієнтом навчання alpha
    #		для точки зі значеннями вагів theta та тренувальною вибіркою (x, y)
    #		Ваш код повинен	присвоювати розрахований градієнт
    #		змінній next_theta, яку повертає ця функція
    #		(орієнтовно 2 строчки коду)
    # =========================================================================
    pass


def getNormalEquations(x, y):
    # Функція розраховує значення вагів за допомогою нормальних рівнянь

    # ====================== ВАШ КОД ТУТ ======================
    # Інструкції: Завершіть наступний код, щоб розрахувати оптимальні ваги
    #		best_thetas за допомогою нормальних рівнянь
    #		Ваш код повинен	присвоювати розраховані ваги
    #		змінній best_theta, яку повертає ця функція
    #		(орієнтовно 1 строчка коду)

    # =========================================================================
    pass

def getWeightedLRPrediction(alpha, n_iterations, x_query, tau, x, y):
    # Функція передбачає значення y в точці x_query за допомогою зваженої лінійної регресії (weighted linear regression).
    # Для цього необхідно зробити n_itererations кроків градієнтного спуску для навчання зваженої лінійної регресії, а після
    # цього використати навчені ваги для передбачення y.
    #
    # Параметри функції:
    # alpha - крок навчання (learning rate)
    # n_iterations - кількість кроків градієнтного спуску
    # x_query - точка для розрахунку передбачень
    # tau - коефіцієнт тау для розрахунку вагів

    # ====================== ВАШ КОД ТУТ ======================
    # Інструкції: Завершіть наступний код, щоб передбачити значення y в точці x_query.
    #		Для цього необхідно навчити weighted linear regression і використати
    #		навчені ваги для передбачення
    #		Ваш код повинен	присвоювати розраховане передбачення змінній predicted,
    #		яку повертає ця функція
    #		(орієнтовно 8-10 строчок коду)
    pass

    
#my functions
def normalize_dataset(X_train, X_test):
    #Напишите функцию, которая нормализирует тренировочный датасет X_train
    #и применяет трансформацию к тестирующему датасету X_test
    #Нельзя нормировать по всему датасету сразу!!!
    #Функция возвращает два нормализованных датасета
    #Замечание: Используйте для этого StandardScaler из sklearn
    pass

def create_train_test(data, train_percentage):
    
    #Напишите функцию для разбиения датасета на тренировочный и тестовый. Возьмите ПЕРВЫЕ 80% выборки
    #в пропорциях train_percentage / (1 - train_percentage)
    #После разбиения надо нормализировать датасеты'

    pass



def is_equal(a, b, eps=1e-3):
    return np.abs(a - b) < eps


def test():
    data = pd.read_csv("prices.csv")
    data = data.astype(np.float32)    

    X_train_intercept, X_test_intercept, y_train, y_test = create_train_test(data, 0.8)

    theta = np.zeros(X_train_intercept.shape[1])
    prediction = getPredict(theta, X_train_intercept)

    print "Testing getPrediction"
    if (len(prediction) == 76 and np.sum(prediction) == 0):
        print "PASSED TEST"
    else:
        raise Exception("Failed Test. Something wrong in function getPrediction")

    print "------------------------------------------------------"
    print "Testing getError"

    error = getError(prediction, y_train)
    if len(error) == 76 and is_equal(error[0], 81000000.0) and is_equal(error[6], 68062500.0):
        print "PASSED TEST"
    else:
        raise Exception("Failed Test. Something wrong in function getError")


    print "------------------------------------------------------"
    print "Testing getGradient.py"

    gradient = getGradient(theta, X_train_intercept, y_train)
    if len(gradient) == 9 and np.abs(gradient[0] + 936300) < 100 and np.abs(gradient[1] + 148758) < 100:
        print "PASSED TEST"
    else:
        raise Exception("Failed Test. Something wrong in function getGradient")

    
    for i in xrange(100):
        prediction = getPredict(theta, X_train_intercept)
        if i % 10 == 0:
            print("Iteration %d, sum error = %d\n", i, np.sum(getError(prediction, y_train)))
        theta = getGradientDescentStep(theta, 0.001, X_train_intercept, y_train)
    trained_theta = theta
    print "------------------------------------------------------"
    print "Testing getGradientDescentStep"
    if np.abs(theta[0] - 6159.0) < 2 and np.abs(theta[2] - 106) < 2:
            print "PASSED TEST"
    else:
        raise Exception("Failed Test. Something wrong in function getGradientDescentStep")

    best_theta = getNormalEquations(X_train_intercept, y_train)
    print "best_theta       :     theta by gradient descent"
    for x in zip(best_theta, theta):
        print x
    print "------------------------------------------------------"
    print "Testing getNormalEquations"
    if np.abs(best_theta[0] - 6159.0) < 2 and np.abs(best_theta[2] - 106) < 2:
            print "PASSED TEST"
    else:
        raise Exception("Failed Test. Something wrong in function getNormalEquations")


    predicts = []
    preds = []
    for i in xrange(len(X_test_intercept)):
        x_query = X_test_intercept[i, :]
        predicted_wlr = getWeightedLRPrediction(0.001, 100, x_query, 0.5, X_train_intercept, y_train)
        predicted_lr = getPredict(trained_theta, x_query)
        predicts.append(predicted_wlr)
        preds.append(predicted_lr)
    predicts = np.array(predicts)
    preds = np.array(preds)
    error_wlr = np.sum((predicts - y_test) ** 2)
    error_lr =  np.sum((preds - y_test) ** 2)
    print "error weighted linear regression: ", error_wlr
    print "error linear regression:          ", error_lr

    print "------------------------------------------------------"
    print "Testing getWeightedLRPrediction (1)"

    if (error_lr > error_wlr):
        print "PASSED TEST"
    else:
        raise Exception("Failed Test. Something wrong in function getWeightedLRPrediction")


    print "------------------------------------------------------"
    test_example_id = 3
    x_query = X_test_intercept[test_example_id, :]
    predicted_wlr = getWeightedLRPrediction(0.001, 100, x_query, 0.5, X_train_intercept, y_train) * 1000
    predicted_lr = getPredict(trained_theta, x_query) * 1000
    print predicted_lr, predicted_wlr

    print "Predicted price with weighted linear regression", predicted_wlr
    print "Predicted price with standard linear regression", predicted_lr
    print "Real price", y_test[test_example_id] * 1000
    print "------------------------------------------------------"
    print "Testing getWeightedLRPrediction(2)"
    if np.abs(predicted_wlr - 4391591) < 10:
            print "PASSED TEST"
    else:
        raise Exception("Failed Test. Something wrong in function getWeightedLRPrediction")


if __name__ == "__main__":
    test()