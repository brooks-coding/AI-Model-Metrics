from shiny import App, render, ui, reactive
import shinyswatch
import math
import pandas as pd
from pmdarima.arima import auto_arima
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, median_absolute_error
import matplotlib.pyplot as plt
import xgboost as xgb

df = pd.read_csv('~/Desktop/Data Science/AI Forex/complete_model_data.csv')
entity_counts = df['Entity'].value_counts()
valid_entities = entity_counts[entity_counts > 6].index
df = df.loc[df['Entity'].isin(valid_entities)]
df['Entity'] = df['Entity'].astype(str)
df['Currency Pair'] = df['Currency Pair'].astype(str)
df['Account'] = df['Account'].astype(str)
df = df.drop_duplicates()
df = df.drop_duplicates(subset=['Entity', 'Currency Pair', 'Account'])

app_ui = ui.page_fluid(
    ui.h1("Comparing Various Forecasting Machine Learning Models"),
    ui.p('Forecasting is the process of analyzing past data to make predictions about the future. \
         Techniques are applicable in a wide variety of disciplines including but not limited to meteorology, health care, and finance. \
         Looking at the specific domain of Forex currency rate hedging, FP&A or financial planning and analysis teams have traditionally done the bulk of predictive calculations. \
         However, it has become more apparent that better forecasts can be made by leveraging artificial intelligence models. \
         In this app, you can explore the capabilities of these algorithms for various currency pairs and entities against past predictions of FP&A teams.'),
    ui.h2("Actual vs. FP&A"),
    ui.p("Before we get into any AI modeling, let's see how the FP&A team's predictions do against the actual currency rates. Select an entity, currency pair, and account below."),
    ui.layout_sidebar(
        ui.sidebar(
            ui.p("Select an entity, then choose from available currency pairs and accounts."),
            ui.input_select("entity", "Select Entity", choices=df['Entity'].unique().tolist()),
            ui.input_select("currency_pair", "Select Currency Pair", choices=[]),
            ui.input_select("account", "Select Account", choices=[])
        ),
        ui.layout_columns(
            ui.output_plot("plot2"),
            ui.row(
                ui.output_data_frame("df_errors"),
                ui.p(""),
                ui.output_text("model_comparison")
            )
        )
    ),
    ui.p("FP&A predictions have been overestimating the actual rates more often than not. \
          That being said, it is reasonable to declare that no prediction can be perfect. \
          However, it is definitely possible to improve the accuracy with AI. \
          Let's add some to the plot above."),
    ui.h2("Models vs. FP&A"),
    #ui.output_plot("plot2"),
    ui.p("Let's also look at some plots that describe the error for every model to further display this difference."),
    ui.layout_columns(
        ui.output_plot("plot3"),
        ui.output_plot("plot4"),
        ui.output_plot("plot5")
    ),
    ui.p("Once again, across all time spreads and metrics, FP&A underperforms in comparison to all models."),
    theme=shinyswatch.theme.darkly
)

def server(input, output, session):
    @reactive.Effect
    @reactive.event(input.entity)
    def update_currency_pairs():
        entity = input.entity()
        filtered_pairs = df[df['Entity'] == entity]['Currency Pair'].unique().tolist()
        ui.update_select("currency_pair", choices=filtered_pairs)

    @reactive.Effect
    @reactive.event(input.entity)
    def update_accounts():
        entity = input.entity()
        filtered_accounts = df[df['Entity'] == entity]['Account'].unique().tolist()
        ui.update_select("account", choices=filtered_accounts)

    @render.data_frame
    def df_errors():
        df = pd.read_csv('~/Desktop/Data Science/AI Forex/complete_model_data.csv')
        df.set_index('Month', inplace=True)
        df = df.loc[(df['Entity'] == int(input.entity())) & (df['Currency Pair'] == input.currency_pair()) & (df['Account'] == input.account())]
        df_tg_fpa = df.copy()
        df = df[['Activity']]
        df_feature = df.copy()
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
        train = df[:split_point]
        test = df[split_point:]

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

        df = df_tg_fpa['1/1/2024':]
        fpa = df['FPA']
        time_gen = df['TimeGen']
        random_forest = random_forest_model.predict(X_test)
        xgboost = xgb_model.predict(X_test)

        mape_arima = mean_absolute_percentage_error(test['Activity'], arima)
        mape_fpa = mean_absolute_percentage_error(test['Activity'], fpa)
        mape_time_gen = mean_absolute_percentage_error(test['Activity'], time_gen)
        mape_random_forest = mean_absolute_percentage_error(test['Activity'], random_forest)
        mape_xgboost = mean_absolute_percentage_error(test['Activity'], xgboost)

        mse_arima = math.sqrt(mean_squared_error(test['Activity'], arima))
        mse_fpa = math.sqrt(mean_squared_error(test['Activity'], fpa))
        mse_time_gen = math.sqrt(mean_squared_error(test['Activity'], time_gen))
        mse_random_forest = math.sqrt(mean_squared_error(test['Activity'], random_forest))
        mse_xgboost = math.sqrt(mean_squared_error(test['Activity'], xgboost))

        medae_arima = median_absolute_error(test['Activity'], arima)
        medae_fpa = median_absolute_error(test['Activity'], fpa)
        medae_time_gen = median_absolute_error(test['Activity'], time_gen)
        medae_random_forest = median_absolute_error(test['Activity'], random_forest)
        medae_xgboost = median_absolute_error(test['Activity'], xgboost)

        errors = pd.DataFrame({
            'Model': ['ARIMA', 'Random Forest', 'FP&A', 'TimeGen1', 'XGBoost'],
            'MAPE': [mape_arima, mape_random_forest, mape_fpa, mape_time_gen, mape_xgboost],
            'RMSE': [mse_arima, mse_random_forest, mse_fpa, mse_time_gen, mse_xgboost],
            'MedAE': [medae_arima, medae_random_forest, medae_fpa, medae_time_gen, medae_xgboost]
        }).round({'MAPE': 5, 'RMSE': 2, 'MedAE': 2})

        return render.DataGrid(errors)
    
    @render.text
    def model_comparison():
        df = pd.read_csv('~/Desktop/Data Science/AI Forex/complete_model_data.csv')
        df.set_index('Month', inplace=True)
        df = df.loc[(df['Entity'] == int(input.entity())) & (df['Currency Pair'] == input.currency_pair()) & (df['Account'] == input.account())]
        df_tg_fpa = df.copy()
        df = df[['Activity']]
        df_feature = df.copy()
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
        train = df[:split_point]
        test = df[split_point:]

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

        df = df_tg_fpa['1/1/2024':]
        fpa = df['FPA']
        time_gen = df['TimeGen']
        random_forest = random_forest_model.predict(X_test)
        xgboost = xgb_model.predict(X_test)

        mape_arima = mean_absolute_percentage_error(test['Activity'], arima)
        mape_fpa = mean_absolute_percentage_error(test['Activity'], fpa)
        mape_time_gen = mean_absolute_percentage_error(test['Activity'], time_gen)
        mape_random_forest = mean_absolute_percentage_error(test['Activity'], random_forest)
        mape_xgboost = mean_absolute_percentage_error(test['Activity'], xgboost)

        mse_arima = math.sqrt(mean_squared_error(test['Activity'], arima))
        mse_fpa = math.sqrt(mean_squared_error(test['Activity'], fpa))
        mse_time_gen = math.sqrt(mean_squared_error(test['Activity'], time_gen))
        mse_random_forest = math.sqrt(mean_squared_error(test['Activity'], random_forest))
        mse_xgboost = math.sqrt(mean_squared_error(test['Activity'], xgboost))

        medae_arima = median_absolute_error(test['Activity'], arima)
        medae_fpa = median_absolute_error(test['Activity'], fpa)
        medae_time_gen = median_absolute_error(test['Activity'], time_gen)
        medae_random_forest = median_absolute_error(test['Activity'], random_forest)
        medae_xgboost = median_absolute_error(test['Activity'], xgboost)

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

        return f'According to the error metrics, the best model for forecasting {input.currency_pair()} {input.account()} for Entity {input.entity()} 2023-2024 is {best_model}.'




    @render.data_frame
    def df_filtered():
        df = pd.read_csv('~/Desktop/Data Science/AI Forex/complete_model_data.csv')
        df = df.loc[(df['Entity'] == int(input.entity())) & (df['Currency Pair'] == input.currency_pair()) & (df['Account'] == input.account())]
        df = df.rename(columns={'Activity':'Actual'})
        df = df[['Month', 'FPA', 'Actual']]
        df.set_index(df['Month'].copy(), inplace=True)
        df = df['1/1/2023':]
        return render.DataGrid(df)
    
    @render.plot(alt='A line graph1')
    def plot1():
        df = pd.read_csv('~/Desktop/Data Science/AI Forex/complete_model_data.csv')
        df.set_index('Month', inplace=True)
        df = df.loc[(df['Entity'] == int(input.entity())) & (df['Currency Pair'] == input.currency_pair()) & (df['Account'] == input.account())]
        df = df['1/1/2023':]
        fig, ax = plt.subplots()
        ax.plot(df.index, df['Activity'], color='deepskyblue', label='Actual')
        ax.plot(df.index, df['FPA'], color='red', label='FPA')
        plt.xticks(ax.get_xticks()[::2], rotation = 45)
        plt.title(f'{input.currency_pair()} {input.account()} for Entity {input.entity()} 2023-2024')
        plt.xlabel('Reporting Month', fontsize=10)
        plt.ylabel('EURUSD 3101 Internal Sales USD', fontsize=10)
        plt.legend()
        return fig
    
    @render.plot(alt="A line graph2")
    def plot2():
        df = pd.read_csv('~/Desktop/Data Science/AI Forex/complete_model_data.csv')
        df.set_index('Month', inplace=True)
        df = df.loc[(df['Entity'] == int(input.entity())) & (df['Currency Pair'] == input.currency_pair()) & (df['Account'] == input.account())]
        df_tg_fpa = df.copy()
        df = df[['Activity']]
        df_feature = df.copy()
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

        train = df[:split_point]
        test = df[split_point:]

        # ARIMA
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
        

        # Random Forest Regressor
        random_forest_model = RandomForestRegressor(n_estimators=100, random_state=42)
        random_forest_model.fit(X_train, y_train)
        
        # XGBoost Regressor
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

        # Forecasts
        df = df_tg_fpa['1/1/2024':]
        time_gen = df['TimeGen']
        fpa = df['FPA']
        random_forest = pd.DataFrame(random_forest_model.predict(X_test))
        xgboost = pd.DataFrame(xgb_model.predict(X_test))

        fig, ax = plt.subplots()
        ax.plot(test.index, test['Activity'], color='red', label='Actual')
        ax.plot(arima.index, arima[0], color='salmon', label='ARIMA')
        ax.plot(fpa.index, fpa, color='blue', label='FP&A')
        ax.plot(time_gen.index, time_gen, color='gold', label='TimeGen1')
        ax.plot(random_forest.index, random_forest, color='forestgreen', label='Random Forest')
        ax.plot(xgboost.index, xgboost, color='cyan', label='XGBoost') 
        plt.xticks(rotation = 45, fontsize=8)
        plt.xlabel('Reporting Month', fontsize=10)
        plt.ylabel('EURUSD 3101 Internal Sales USD', fontsize=10)
        plt.legend()
        return fig

    @render.plot(alt='An error graph')
    def plot3():
        df = pd.read_csv('~/Desktop/Data Science/AI Forex/complete_model_data.csv')
        df.set_index('Month', inplace=True)
        df = df.loc[(df['Entity'] == int(input.entity())) & (df['Currency Pair'] == input.currency_pair()) & (df['Account'] == input.account())]
        df_tg_fpa = df.copy()
        df = df[['Activity']]
        df_feature = df.copy()
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
        train = df[:split_point]
        test = df[split_point:]

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
        
        df = df_tg_fpa['1/1/2024':]
        time_gen = df['TimeGen']
        fpa = df['FPA']
        random_forest = pd.DataFrame(random_forest_model.predict(X_test))
        xgboost = pd.DataFrame(xgb_model.predict(X_test))

        error_arima = abs(arima.values-test)
        error_timegen = abs(time_gen-df['Activity'])
        error_fpa = abs(fpa-df['Activity'])
        error_random_forest = abs(random_forest.values-test)
        error_xgboost = abs(xgboost.values-test)


        fig, ax = plt.subplots()
        ax.plot(test.index, error_arima, color='salmon', label='ARIMA')
        ax.plot(test.index, error_random_forest, color='darkgreen', label='Random Forest')
        ax.plot(test.index, error_xgboost, color='cyan', label='XGBoost Regressor')
        ax.plot(test.index, error_timegen, color='gold', label='TimeGen1')
        ax.plot(test.index, error_fpa, color='blue', label='FPA')
        plt.xticks(fontsize=8, rotation = 45)
        plt.xlabel('Reporting Month', fontsize=10)
        plt.ylabel('Error', fontsize=10)
        plt.title('Absolute Error')
        plt.legend()
        return fig

    @render.plot(alt="Another error graph")
    def plot4():
        df = pd.read_csv('~/Desktop/Data Science/AI Forex/complete_model_data.csv')
        df.set_index('Month', inplace=True)
        df = df.loc[(df['Entity'] == int(input.entity())) & (df['Currency Pair'] == input.currency_pair()) & (df['Account'] == input.account())]
        df_tg_fpa = df.copy()
        df = df[['Activity']]
        df_feature = df.copy()
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
        train = df[:split_point]
        test = df[split_point:]

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

        df = df_tg_fpa['1/1/2024':]
        time_gen = df['TimeGen']
        fpa = df['FPA']
        random_forest = pd.DataFrame(random_forest_model.predict(X_test))
        xgboost = pd.DataFrame(xgb_model.predict(X_test))

        error_arima = math.sqrt(mean_squared_error(test['Activity'], arima))
        error_fpa = math.sqrt(mean_squared_error(test['Activity'], fpa))
        error_time_gen = math.sqrt(mean_squared_error(test['Activity'], time_gen))
        error_random_forest = math.sqrt(mean_squared_error(test['Activity'], random_forest))
        error_xgboost = math.sqrt(mean_squared_error(test['Activity'], xgboost))

        fig, ax = plt.subplots()
        ax.bar(['Arima', 'Random Forest', 'XGBoost', 'TimeGen1', 'FPA'], [error_arima, error_random_forest, error_xgboost, error_time_gen, error_fpa])
        plt.xticks(fontsize=8)
        plt.xlabel('Reporting Month', fontsize=10)
        plt.ylabel('Error', fontsize=10)
        plt.title('Root Mean Squared Error')
        return fig

    @render.plot("Yet another error graph")
    def plot5():
        df = pd.read_csv('~/Desktop/Data Science/AI Forex/complete_model_data.csv')
        df.set_index('Month', inplace=True)
        df = df.loc[(df['Entity'] == int(input.entity())) & (df['Currency Pair'] == input.currency_pair()) & (df['Account'] == input.account())]
        df_tg_fpa = df.copy()
        df = df[['Activity']]
        df_feature = df.copy()
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
        train = df[:split_point]
        test = df[split_point:]

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

        df = df_tg_fpa['1/1/2024':]
        fpa = df['FPA']
        time_gen = df['TimeGen']
        random_forest = random_forest_model.predict(X_test)
        xgboost = xgb_model.predict(X_test)

        error_arima = mean_absolute_percentage_error(test['Activity'], arima)
        error_fpa = mean_absolute_percentage_error(test['Activity'], fpa)
        error_time_gen = mean_absolute_percentage_error(test['Activity'], time_gen)
        error_random_forest = mean_absolute_percentage_error(test['Activity'], random_forest)
        error_xgboost = mean_absolute_percentage_error(test['Activity'], xgboost)


        fig, ax = plt.subplots()
        ax.bar(['Arima', 'Random Forest', 'XGBoost', 'TimeGen1', 'FPA'], [error_arima, error_random_forest, error_xgboost, error_time_gen, error_fpa])
        plt.xticks(fontsize=8)
        plt.xlabel('Reporting Month', fontsize=10)
        plt.ylabel('Percent Error', fontsize=10)
        plt.title("Mean Absolute Percent Error")
        return fig

app = App(app_ui, server)
app.run()
