import pandas as pd
import numpy as np

import seaborn as sns
from matplotlib import pyplot as plt

class CarPrice:

    def __init__(self):
        self.df = pd.read_csv('data/data.csv')
        print(f'${len(self.df)} lines loaded')
        self.df.columns = self.df.columns.str.lower().str.replace(' ', '_')

        string_columns = list(self.df.dtypes[self.df.dtypes == 'object'].index)

        for col in string_columns:
            self.df[col] = self.df[col].str.lower().str.replace(' ', '_')

    def set_base(self,base):
        self.base = base;

    def trim(self):
        self.df.columns = self.df.columns.str.lower().str.replace(' ', '_')
        string_columns = list(self.df.dtypes[self.df.dtypes == 'object'].index)
        for col in string_columns:
            self.df[col] = self.df[col].str.lower().str.replace(' ', '_')

    def validate(self):
        np.random.seed(2)
        n = len(self.df)
        n_val = int(0.2 * n)
        n_test = int(0.2 * n)
        n_train = n - n_val - n_test

        idx = np.arange(n)
        np.random.shuffle(idx)
        df_shuffled = self.df.iloc[idx]
        self.train = df_shuffled.iloc[:n_train].copy()
        self.val = df_shuffled.iloc[n_train: n_train + n_val].copy()
        self.test = df_shuffled.iloc[n_train + n_val:].copy()
        y_train_orig = self.train.msrp.values
        y_val_orig = self.val.msrp.values
        y_test_orig = self.test.msrp.values

        self.y_train = np.log1p(y_train_orig)
        self.y_val = np.log1p(y_val_orig)
        self.y_test =  np.log1p(y_test_orig)

        del self.train['msrp']
        del self.val['msrp']
        del self.test['msrp']
        pass

    def prepare_X(self, X):
        df = X.copy()
        self.features = self.base.copy()
        df['age'] = 2017 - df.year
        self.features.append('age')

        for v in [2, 3, 4]:
            feature = 'num_doors_%s' % v
            df[feature] = (df['number_of_doors'] == v).astype(int)
            self.features.append(feature)

        for v in ['chevrolet', 'ford', 'volkswagen', 'toyota', 'dodge']:
            feature = 'is_make_%s' % v
            df[feature] = (df['make'] == v).astype(int)
            self.features.append(feature)

        for v in ['regular_unleaded', 'premium_unleaded_(required)', 'premium_unleaded_(recommended)', 'flex-fuel_(unleaded/e85)']:
            feature = 'is_type_%s' % v
            df[feature] = (df['engine_fuel_type'] == v).astype(int)
            self.features.append(feature)

        for v in ['automatic', 'manual', 'automated_manual']:
            feature = 'is_transmission_%s' % v
            df[feature] = (df['transmission_type'] == v).astype(int)
            self.features.append(feature)

        for v in ['front_wheel_drive', 'rear_wheel_drive', 'all_wheel_drive', 'four_wheel_drive']:
            feature = 'is_driven_wheens_%s' % v
            df[feature] = (df['driven_wheels'] == v).astype(int)
            self.features.append(feature)

        for v in ['crossover', 'flex_fuel', 'luxury', 'luxury,performance', 'hatchback']:
            feature = 'is_mc_%s' % v
            df[feature] = (df['market_category'] == v).astype(int)
            self.features.append(feature)

        for v in ['compact', 'midsize', 'large']:
            feature = 'is_size_%s' % v
            df[feature] = (df['vehicle_size'] == v).astype(int)
            self.features.append(feature)

        for v in ['sedan', '4dr_suv', 'coupe', 'convertible', '4dr_hatchback']:
            feature = 'is_style_%s' % v
            df[feature] = (df['vehicle_style'] == v).astype(int)
            self.features.append(feature)


        df_num = df[self.features]
        df_num = df_num.fillna(0)
        X = df_num.values
        return X
        
    def linear_regression(self, X, y):
        ones = np.ones(X.shape[0])
        X = np.column_stack([ones, X])

        XTX = X.T.dot(X)
        XTX_inv = np.linalg.inv(XTX)
        w = XTX_inv.dot(X.T).dot(y)
        return w[0], w[1:]

    def linear_regression_reg(self, X, y, r):
        ones = np.ones(X.shape[0])
        X = np.column_stack([ones, X])

        XTX = X.T.dot(X)
        reg = r * np.eye(XTX.shape[0])
        XTX = XTX + reg

        XTX_inv = np.linalg.inv(XTX)
        w = XTX_inv.dot(X.T).dot(y)
        
        return w[0], w[1:]

    def rmse(self, y, y_pred):
        error = y_pred - y
        mse = (error ** 2).mean()
        return np.sqrt(mse)




carprice = CarPrice()
carprice.trim()
carprice.validate()
base = ['engine_hp', 'engine_cylinders', 'highway_mpg', 'city_mpg', 'popularity']
carprice.set_base(base)
X_train = carprice.prepare_X(carprice.train)
X_test = carprice.prepare_X(carprice.test)
X_val = carprice.prepare_X(carprice.val)
#print(carprice.features)
#train the models
w_0, w = carprice.linear_regression(X_train, carprice.y_train)
w_0_r, w_r = carprice.linear_regression_reg(X_train, carprice.y_train, 0.001)
print('w_0 & w: ', w_0, w)
print('w_0 & w with regulation: ', w_0_r, w_r)
y_pred_train = w_0 + X_train.dot(w)
y_pred_train_r = w_0_r + X_train.dot(w_r)

#use train dataset predicted MSRP and train dataset real MSRP plot a chart
plt.figure(figsize = (6, 4))
sns.distplot(carprice.y_train, label = 'target', kde = False, hist_kws = dict(color = '#222222', alpha = 0.6))
sns.distplot(y_pred_train_r, label = 'prediction' , kde = False, hist_kws = dict(color = '#aaaaaa', alpha = 0.8))
plt.legend()
plt.ylabel('Frequency')
plt.xlabel('Log(price + 1)')
plt.title('Predictions vs actual distribution')
plt.show(block = True)


#print RMSE for train/test/validation dataset from linear_regression model and linear_regression_reg model
print('Train dataset RMSE: ', carprice.rmse(carprice.y_train, y_pred_train))

y_pred_test = w_0 + X_test.dot(w)
print('Test dataset RMSE: ', carprice.rmse(carprice.y_test, y_pred_test))

y_pred_val = w_0 + X_val.dot(w)
print('Validation dataset RMSE: ', carprice.rmse(carprice.y_val, y_pred_val))


print('Train dataset RMSE with regulation: ', carprice.rmse(carprice.y_train, y_pred_train_r))

y_pred_test_r = w_0_r + X_test.dot(w_r)
print('Test dataset RMSE with regulation: ', carprice.rmse(carprice.y_test, y_pred_test_r))

y_pred_val_r = w_0_r + X_val.dot(w_r)
print('Validation dataset RMSE with regulation: ', carprice.rmse(carprice.y_val, y_pred_val_r))


#using the model
cars = [1, 16, 27, 35, 84]
output_base = ["engine_cylinders", "transmission_type", "driven_wheels", "number_of_doors", "market_category", "vehicle_size", "vehicle_style", "highway_mpg", "city_mpg", "popularity", "msrp", "msrp_pred"]
print(*output_base, sep = " | ")
output_base.remove('msrp')
output_base.remove('msrp_pred')
for i in cars:
    ad = carprice.test.iloc[i].to_dict()
    one_car = carprice.prepare_X(pd.DataFrame([ad]))[0]
    y_pred = w_0_r + one_car.dot(w_r)
    suggestion = np.expm1(y_pred)
    print(*carprice.test[output_base].iloc[i].values, np.expm1(carprice.y_test[i]).round(2),  suggestion.round(2), sep = "  |  ")