import pandas as pd
import numpy as np

import seaborn as sns
from matplotlib import pyplot as plt

class CarPrice:

    def __init__(self):
        self.df = pd.read_csv('data/data.csv')
        print(f'${len(self.df)} lines loaded')

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
        n_train = n - (n_val + n_test)

        idx = np.arange(n)
        np.random.shuffle(idx)

        df_shuffled = self.df.iloc[idx]

        df_train = df_shuffled.iloc[:n_train].copy()
        df_val = df_shuffled.iloc[n_train:n_train+n_val].copy()
        df_test = df_shuffled.iloc[n_train+n_val:].copy()

        y_train_orig = df_train.msrp.values
        y_val_orig = df_val.msrp.values
        y_test_orig = df_test.msrp.values

        y_train = np.log1p(df_train.msrp.values)
        y_val = np.log1p(df_val.msrp.values)
        y_test = np.log1p(df_test.msrp.values)

        return df_train, df_val, df_test, y_train, y_val, y_test, y_val_orig,y_train_orig
     

    def prepare_X(self, df):
        base = ['engine_hp', 'engine_cylinders', 'highway_mpg', 'city_mpg', 'popularity']
        df = df.copy()
        features = base.copy()

        df['age'] = 2017 - df.year
        features.append('age')
        
        for v in [2, 3, 4]:
            feature = 'num_doors_%s' % v
            df[feature] = (df['number_of_doors'] == v).astype(int)
            features.append(feature)

        for v in ['chevrolet', 'ford', 'volkswagen', 'toyota', 'dodge']:
            feature = 'is_make_%s' % v
            df[feature] = (df['make'] == v).astype(int)
            features.append(feature)

        for v in ['regular_unleaded', 'premium_unleaded_(required)', 
                'premium_unleaded_(recommended)', 'flex-fuel_(unleaded/e85)']:
            feature = 'is_type_%s' % v
            df[feature] = (df['engine_fuel_type'] == v).astype(int)
            features.append(feature)

        for v in ['automatic', 'manual', 'automated_manual']:
            feature = 'is_transmission_%s' % v
            df[feature] = (df['transmission_type'] == v).astype(int)
            features.append(feature)

        for v in ['front_wheel_drive', 'rear_wheel_drive', 'all_wheel_drive', 'four_wheel_drive']:
            feature = 'is_driven_wheens_%s' % v
            df[feature] = (df['driven_wheels'] == v).astype(int)
            features.append(feature)

        for v in ['crossover', 'flex_fuel', 'luxury', 'luxury,performance', 'hatchback']:
            feature = 'is_mc_%s' % v
            df[feature] = (df['market_category'] == v).astype(int)
            features.append(feature)

        for v in ['compact', 'midsize', 'large']:
            feature = 'is_size_%s' % v
            df[feature] = (df['vehicle_size'] == v).astype(int)
            features.append(feature)

        for v in ['sedan', '4dr_suv', 'coupe', 'convertible', '4dr_hatchback']:
            feature = 'is_style_%s' % v
            df[feature] = (df['vehicle_style'] == v).astype(int)
            features.append(feature)

        df_num = df[features]
        df_num = df_num.fillna(0)
        X = df_num.values
        return X


    def linear_regression_reg(self, X, y, r=0.0):
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



def test() -> None:
    carprice=CarPrice()
    carprice.trim()
    df_train,df_val,df_test,y_train,y_val,y_test, y_val_orig, y_train_orig= carprice.validate()
    X_train = carprice.prepare_X(df_train)
    w_0, w = carprice.linear_regression_reg(X_train,y_train,r=0.01)
    X_val = carprice.prepare_X(df_val)
    y_pred = w_0 + X_val.dot(w)
    X_test = carprice.prepare_X(df_test)
    y_pred = w_0 + X_test.dot(w)

    columns= []
    for i in range(5):
        ad = df_test.iloc[i].to_dict()
     
        X_test = carprice.prepare_X(pd.DataFrame([ad]))[0]
        y_pred = w_0 + X_test.dot(w)
        suggestion = np.expm1(y_pred)
        ad["predicted_msrp"] =  suggestion
        columns.append(ad)   

    df = pd.DataFrame(columns)  
    print(df[["engine_cylinders","transmission_type", "driven_wheels", "number_of_doors", 
    "market_category", "vehicle_size", "vehicle_style", "highway_mpg", "city_mpg", 
    "popularity", "msrp", "predicted_msrp"]].to_markdown())



if __name__ == "__main__":
    # execute only if run as a script
    test()