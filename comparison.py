import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, median_absolute_error
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from pmdarima.arima import auto_arima

def evaluate_models(make_plots, print_errors, print_details):
    df = pd.read_csv("~/Desktop/Data Science/AI Forex/complete_model_data.csv")
    df.set_index('Month', inplace=True)

    combinations = df[['Entity', 'Currency Pair', 'Account']].drop_duplicates()

    model_scores = {
        'ARIMA': 0, 
        'Random Forest': 0,
        'FP&A': 0,
        'TimeGen1': 0,
        'XGBoost': 0
    }

    model_combinations = {
        'ARIMA': [],
        'Random Forest': [],
        'FP&A': [],
        'TimeGen1': [],
        'XGBoost': []
    }

    best_model_predictions = {}

    for _, row in combinations.iterrows():
        try:
            entity, currency_pair, account = row['Entity'], row['Currency Pair'], row['Account']
            df_subset = df.loc[(df['Entity'] == int(entity)) & (df['Currency Pair'] == str(currency_pair)) & (df['Account'] == str(account))]
            df_tg_fpa = df_subset.copy()

            df_feature = df_subset[['Activity']].copy()
            df_feature['lag1'] = df_feature['Activity'].shift(1)
            df_feature['lag12'] = df_feature['Activity'].shift(12)
            df_feature['rolling_mean'] = df_feature['Activity'].rolling(window=12).mean()
            df_feature.dropna(inplace=True)

            split_point = '1/1/2024'
            train_feature, test_feature = df_feature[:split_point], df_feature[split_point:]
            X_train = train_feature.drop('Activity', axis=1)
            y_train = train_feature['Activity']
            X_test = test_feature.drop('Activity', axis=1)
            y_test = test_feature['Activity']

            arima_split_point = '12/1/2023'
            arima_train_feature, arima_test_feature = df_feature[:arima_split_point], df_feature[arima_split_point:]
            arima_y_train = arima_train_feature['Activity']
            arima_y_test = arima_test_feature['Activity']

            arima_model = auto_arima(
                y=arima_y_train,
                X=None,
                start_p=2,
                d=None, start_q=2,
                max_p=5,
                max_d=2,
                max_q=5,
                start_P=1,
                D=None,
                start_Q=1,
                max_P=2,
                max_D=1,
                max_Q=2,
                max_order=5,
                m=1,
                seasonal=True,
                stationary=False,
                information_criterion='aic',
                alpha=0.01,
                test='kpss',
                seasonal_test='ocsb',
                stepwise=True,
                n_jobs=1,
                start_params=None,
                trend=None,
                method='lbfgs',
                maxiter=50,
                offset_test_args=None,
                seasonal_test_args=None,
                suppress_warnings=True,
                error_action='trace',
                trace=False,
                random=False,
                random_state=20,
                n_fits=10,
                return_valid_fits=False,
                out_of_sample_size=0,
                scoring='mse',
                scoring_args=None,
                with_intercept='auto', 
                sarimax_kwargs=None)

            arima = pd.DataFrame(arima_model.predict(n_periods=6), index=arima_y_test.index).dropna()

            random_forest_model = RandomForestRegressor(n_estimators=100, random_state=42)
            random_forest_model.fit(X_train, y_train)
            random_forest = random_forest_model.predict(X_test)

            xgb_model = xgb.XGBRegressor(base_score=0.5,
                                         subsample=0.7,
                                         booster='gbtree',
                                         n_estimators=1000,
                                         objective='reg:squarederror',
                                         max_depth=None,
                                         colsample_bytree=0.9,
                                         reg_lambda=0.7,
                                         reg_alpha=0.7)
            xgb_model.fit(X_train, y_train)
            xgboost = xgb_model.predict(X_test)

            fpa = df_tg_fpa.loc[split_point:]['FPA']
            time_gen = df_tg_fpa.loc[split_point:]['TimeGen']

            mape_arima = mean_absolute_percentage_error(y_test, arima)
            mape_fpa = mean_absolute_percentage_error(y_test, fpa)
            mape_time_gen = mean_absolute_percentage_error(y_test, time_gen)
            mape_random_forest = mean_absolute_percentage_error(y_test, random_forest)
            mape_xgboost = mean_absolute_percentage_error(y_test, xgboost)

            mse_arima = math.sqrt(mean_squared_error(y_test, arima))
            mse_fpa = math.sqrt(mean_squared_error(y_test, fpa))
            mse_time_gen = math.sqrt(mean_squared_error(y_test, time_gen))
            mse_random_forest = math.sqrt(mean_squared_error(y_test, random_forest))
            mse_xgboost = math.sqrt(mean_squared_error(y_test, xgboost))

            medae_arima = median_absolute_error(y_test, arima)
            medae_fpa = median_absolute_error(y_test, fpa)
            medae_time_gen = median_absolute_error(y_test, time_gen)
            medae_random_forest = median_absolute_error(y_test, random_forest)
            medae_xgboost = median_absolute_error(y_test, xgboost)

            errors = pd.DataFrame({
                'Model': ['ARIMA', 'Random Forest', 'FP&A', 'TimeGen1', 'XGBoost'],
                'MAPE': [mape_arima, mape_random_forest, mape_fpa, mape_time_gen, mape_xgboost],
                'RMSE': [mse_arima, mse_random_forest, mse_fpa, mse_time_gen, mse_xgboost],
                'MedAE': [medae_arima, medae_random_forest, medae_fpa, medae_time_gen, medae_xgboost]
            }).round({'MAPE': 5, 'RMSE': 2, 'MedAE': 2})

            compare_models = {
                'ARIMA': 0,
                'Random Forest': 0,
                'FP&A': 0,
                'TimeGen1': 0,
                'XGBoost': 0
            }

            for metric in ['MAPE', 'RMSE', 'MedAE']:
                compare_models[errors.loc[errors[metric].idxmin(), 'Model']] += 1

            best_model = max(compare_models, key=compare_models.get)

            model_scores[best_model] += 1
            model_combinations[best_model].append(f"{entity}-{currency_pair}-{account}")

            model_predictions = {
                'ARIMA': arima.astype(int),
                'Random Forest': random_forest.astype(int),
                'FP&A': fpa.astype(int),
                'TimeGen1': time_gen.astype(int),
                'XGBoost': xgboost.astype(int)
            }

            if make_plots == True and best_model == 'ARIMA':
                best_model_predictions[f"{entity}-{currency_pair}-{account}"] = model_predictions[best_model]
                fig, axs = plt.subplots(2, 2)
                axs[0, 0].plot(y_train.index, y_train, label='Training Data', color='blue')
                axs[0, 0].plot(y_test.index, y_test, label='Testing Data', color='green')
                axs[0, 0].plot(y_test.index, model_predictions[best_model], label=f'{best_model}', color='gold')
                axs[0, 0].legend()
                new_xticks = axs[0, 0].get_xticks()[::6]
                axs[0, 0].set_xticks(new_xticks)
                axs[0, 0].tick_params(axis='x', rotation=45)
                fig.suptitle(f'Actual vs Predicted for {entity}-{currency_pair}-{account} with {best_model}')
                axs[0, 0].set_facecolor('#F8F8F8')
                axs[0, 0].set_title(f'Best Perfoming Model ({best_model}) in Gold')

                axs[1, 0].plot(y_test.index[-6:], y_test[-6:], label='Actual')
                for model, predictions in model_predictions.items():
                    if model == str(best_model):
                        axs[1, 0].plot(y_test.index[-6:], model_predictions[best_model][-6:], label=f'{best_model}', color='gold')
                    else:
                        axs[1, 0].plot(y_test.index[-6:], model_predictions[model][-6:], label=f'{model}')
                axs[1, 0].legend(loc='upper left')
                axs[1, 0].tick_params(axis='x', rotation=45)
                axs[1, 0].set_title('Comparison With Other Models')
                axs[1, 0].set_facecolor('#F8F8F8')

                axs[0, 1].bar(['Arima', 'Random Forest', 'XGBoost', 'TimeGen1', 'FPA'], [mse_arima, mse_random_forest, mse_xgboost, mse_time_gen, mse_fpa])
                axs[0, 1].set_xlabel('Reporting Month', fontsize=10)
                axs[0, 1].set_ylabel('Error', fontsize=10)
                axs[0, 1].set_title('Root Mean Squared Error')

                axs[1, 1].bar(['Arima', 'Random Forest', 'XGBoost', 'TimeGen1', 'FPA'], [medae_arima, medae_random_forest, medae_xgboost, medae_time_gen, medae_fpa])
                axs[1, 1].set_xlabel('Reporting Month', fontsize=10)
                axs[1, 1].set_ylabel('Error', fontsize=10)
                axs[1, 1].set_title('Median Absolute Error')

                plt.tight_layout()
                plt.show()
        
        except (ValueError, IndexError, KeyError) as e:
            if print_errors == True:
                print(f"Skipping combination {entity}-{currency_pair}-{account} due to error: {e}")
            continue
    
    print("Best model counts across all combinations:")
    for model, score in model_scores.items():
        print(f"{model}: {score}")

    if print_details == True:
        print(f"\nCombinations by best model ({sum(list(model_scores.values()))} total models):")
        for model, combinations in model_combinations.items():
            print(f"\n{model}:")
            for combo in combinations:
                print(f" - {combo}")

evaluate_models(make_plots=False, print_errors=False, print_details=True)
