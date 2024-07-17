from shiny import App, render, ui
import numpy as np
import matplotlib.pyplot as plt

class metrics:
    def __init__(self, correlation):
        rng = np.random.default_rng()
        noise = rng.random(10)
        self.predicted = np.array([num for num in range(1, 11)])
        self.correlation = correlation
        if correlation == "Strong":
            self.actual = np.add(self.predicted, noise)
        elif correlation == "Weak":
            noise = 7 * noise - 4
            self.actual = np.add(self.predicted, noise)
        else:
            noise = 20 * noise - 10
            self.actual = np.add(self.predicted, noise)

    def mae(self):
        mae = np.abs(self.actual - self.predicted).mean()
        return mae
    
    def mape(self):
        mape = np.abs((self.actual - self.predicted) / self.actual).mean()
        return mape
    
    def mdape(self):
        mdape = np.median(np.abs((self.actual - self.predicted) / self.actual))
        return mdape
    
    def mse(self):
        mse = np.square(self.actual - self.predicted).mean()
        return mse
    
    def rmse(self):
        mse = self.mse()
        rmse = np.sqrt(mse)
        return rmse
    
    def nrmse_mean(self):
        rmse = self.rmse()
        nrmse_mean = rmse / self.actual.mean()
        return nrmse_mean
    
    def nrmse_minmax(self):
        rmse = self.rmse()
        nrmse_minmax = rmse / (self.actual.max() - self.actual.min())
        return nrmse_minmax

    def r_squared(self):
        y_bar = self.actual.mean()
        ss_tot = np.square(self.actual - y_bar).sum()
        ss_res = np.square(self.actual - self.predicted).sum()
        r_squared = 1 - (ss_res / ss_tot)
        return r_squared
    
    def print_spread_metric(self, type, val):
        val = str(val)
        return type + " = " + val
    
    def create_plot(self):
        matrix = np.array([self.mae(), self.mape(), self.mdape(), self.mse(), self.rmse(), self.nrmse_mean(), self.nrmse_minmax(), self.r_squared()])
        indices = np.array(["mae", "mape", "mdape", "mse", "rmse", "nrmse_mean", "nrmse_minmax", "r_squared"])

        def addlabels(x, y):
            for i in range(len(x)):
                plt.text(i, y[i], round(y[i], 3), ha='center',
                         bbox=dict(facecolor='#f37053', alpha=.8))

        plt.figure(figsize=(7, 7))
        plt.bar(indices, matrix, color='#154f7d')
        plt.xticks(rotation=45)
        plt.title("Various metrics for forecasting model")
        plt.xlabel("metric")
        plt.ylabel("value")
        addlabels(indices, matrix)

        fig = plt.gcf()
        plt.close(fig)
        return fig


app_ui = ui.page_fluid(
    ui.h1("Exploring Metrics for Forecasting Model"),
    ui.layout_sidebar(
        ui.panel_sidebar(
            ui.card(
        ui.card_header("Model Metrics"),
        ui.p("In this app, you have the ability to explore how various model metrics such as mean squared error and R-squared fluctuate depending on how well the model correlates with the data."),
        ui.input_select("correlation", "Correlation", choices=["Strong", "Weak", "None"], selected="Strong")    )
        ),
        ui.panel_main(
            ui.output_plot("plot")
        )
    ),
    ui.card(
        ui.card_header("An Explanation on Metrics"),
        ui.p("The metrics displayed above all summarize the same thing: how close your model's predictions are to actual values. Where they differ lies in how they penalize certain mistakes. For example, NRMSE or normalized root mean squared error tends to sharply drop when values are overestimated in comparison to underestimated. These tools become especially powerful when comparing between models.")
    ),
    ui.card(
        ui.p("This app was created with Python using Shiny and Numpy. Data was randomly generated according to correlation.")
    )
)

def server(input, output, session):
    @render.plot(alt="A bargraph")
    def plot():
        metric = metrics(input.correlation())
        return metric.create_plot()

app = App(app_ui, server)

if __name__ == '__main__':
    app.run()
