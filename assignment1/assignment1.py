import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import operator

class Regression:

    def __init__(self):
        self.dataset = pd.read_csv('housing.csv',delim_whitespace=True)
        self.dataset.columns=['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD','TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

    def linear_model(self):
        X = self.dataset['LSTAT']
        Y = self.dataset['MEDV']
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 3)
        x_train_df=pd.DataFrame(x_train)
        x_test_df=pd.DataFrame(x_test)
        regr = linear_model.LinearRegression()
        regr.fit(x_train_df,y_train)
        y_pred = regr.predict(x_test_df)
        plt.scatter(x_test_df, y_test, color='cyan')
        plt.plot(x_test_df, y_pred, color='blue')
        plt.xlabel('LSTAT')
        plt.ylabel('MEDV')
        plt.show()
        print(f"Linear Regression: Root Mean squared error:{mean_squared_error(y_test,y_pred,squared=False)}")
        print(f"Linear Regression: R2 Score using Linear Regression:{r2_score(y_test,y_pred)}")


    def polynomial_model(self):
        X_poly = self.dataset['LSTAT']
        Y_poly = self.dataset['MEDV']
        x_train, x_test, y_train, y_test = train_test_split(X_poly, Y_poly, test_size = 0.3, random_state = 3)
        x_train_df=pd.DataFrame(x_train)
        x_test_df=pd.DataFrame(x_test)
        regr_poly = PolynomialFeatures(2)
        x_poly_train= regr_poly.fit_transform(x_train_df)
        x_poly_test= regr_poly.fit_transform(x_test_df)
        reg= linear_model.LinearRegression()
        reg.fit(x_poly_train,y_train)
        y_pred_poly = reg.predict(x_poly_test)
        plt.scatter(x_test, y_test, color='cyan')
        axis = operator.itemgetter(0)
        sortzip = sorted(zip(x_test, y_pred_poly), key = axis)
        X_test2, Y_pred2 = zip(*sortzip)
        plt.plot(X_test2, Y_pred2)
        plt.xlabel('LSTAT')
        plt.ylabel('MEDV')
        plt.show()
        print(f"Polynomial Regression: Root Mean squared error:{mean_squared_error(y_test,y_pred_poly,squared=False)}")
        print(f"Polynomial Regression: R2 Score using Linear Regression:{r2_score(y_test,y_pred_poly)}")

        #Degree 20
        regr_poly = PolynomialFeatures(20)
        x_poly_train= regr_poly.fit_transform(x_train_df)
        x_poly_test= regr_poly.fit_transform(x_test_df)
        reg= linear_model.LinearRegression()
        reg.fit(x_poly_train,y_train)
        y_pred_poly = reg.predict(x_poly_test)
        plt.scatter(x_test, y_test, color='cyan')
        axis = operator.itemgetter(0)
        sortzip = sorted(zip(x_test, y_pred_poly), key = axis)
        X_test2, Y_pred2 = zip(*sortzip)
        plt.plot(X_test2, Y_pred2)
        plt.xlabel('LSTAT')
        plt.ylabel('MEDV')
        plt.show()

    def multiple_model(self):
        X_multi = self.dataset[['LSTAT','RM','PTRATIO']]
        Y_multi = self.dataset['MEDV']
        x_train, x_test, y_train, y_test = train_test_split(X_multi, Y_multi, test_size = 0.3, random_state = 3)
        regr_multi = linear_model.LinearRegression()
        regr_multi.fit(x_train,y_train)
        y_pred = regr_multi.predict(x_test)
        print(f"Multiple Regression: Root Mean squared error:{mean_squared_error(y_test,y_pred,squared=False)}")
        print(f"Multiple Regression: R2 Score using Linear Regression:{r2_score(y_test,y_pred)}")
        print(f"Multiple Regression: Adjusted R2 Score using Linear Regression:{1-(1-r2_score(y_test,y_pred))*(len(x_test)-1)/(len(x_test)-len(x_test.values[0])-1)}")


def test() -> None:
    boston=Regression()
    boston.linear_model()
    boston.polynomial_model()
    boston.multiple_model()



if __name__ == "__main__":
    # execute only if run as a script
    test()