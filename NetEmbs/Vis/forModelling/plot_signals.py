# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
plot_signals.py
Created by lex at 2019-07-22.
"""
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import plotly.graph_objs as go
from plotly.offline import iplot
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


def filterData_v3(df,
                  query=[{"select": ["Sales 21 btw", "Sales 6 btw"],
                          "_with": {"FA_Name": "Revenue", "from": True}},
                         {"select": ["Collections"], "_with": None}],
                  on="GroundTruth"):
    result = list()
    for q in query:
        postfix = {"FA_Name": None, "flow": None}
        if q["select"] is None or q["select"] == ["ALL"] or q["select"] == "ALL":
            cur_df = df
        else:
            cur_df = df[df[on].isin(q["select"])]
        if q["_with"] is not None:
            for key, value in q["_with"].items():
                try:
                    cur_df = cur_df[cur_df[key] == value]
                    postfix[key] = str(value)
                except KeyError as e:
                    raise (f"{a} is not in a columns titles!")
        result.append(cur_df)
        result[-1].name = str(q["select"])[1:-1]
        if on == "label":
            result[-1].name += " cluster"
            if len(q["select"]) > 1:
                result[-1].name += "s"
        if postfix["FA_Name"] is not None:
            result[-1].name += f" – {postfix['FA_Name']}"
        if postfix["flow"] is not None:
            result[-1].name += f"({postfix['flow']})"
    if len(result) == 1:
        return result[0]
    else:
        return tuple(result)


def plotAmounts(DFs, aggtitle="Default signals", corr_score=None, filename="amounts_plot"):
    """Helper funciton to plot a few DataFrame in one plotly graph"""
    if corr_score is not None:
        aggtitle = aggtitle + ".     Correlation: " + str(round(corr_score, 3))
    if len(DFs) > 1:
        fig2 = go.Figure(data=[go.Scatter(x=df.index,
                                          y=df.amount,
                                          name=df.name
                                          ) for df in DFs],
                         layout=go.Layout(width=1200,
                                          height=400, showlegend=True, title=go.layout.Title(text=aggtitle),
                                          hovermode='closest',
                                          legend=dict(orientation="h", font=dict(size=18), xanchor='center', x=0.5,
                                                      y=-0.1),
                                          font=dict(size=18)))
    else:
        fig2 = go.Figure(data=go.Scatter(x=DFs.index,
                                         y=DFs.amount,
                                         name=DFs.name,
                                         layout=go.Layout(width=1200,
                                                          height=400, showlegend=True,
                                                          title=go.layout.Title(text=aggtitle), hovermode='closest')))
    # iplot(fig2)
    fig2.write_image(filename + ".pdf")


legend_postfix = {"W": ", weekly", "D": ", daily", "M": ", monthly", "2D": ", 2 days"}


def constructSignals_v2(df_all, shift=(0, 0), query=[{"select": [2], "_with": None}, {"select": [4], "_with": None}],
                        on="label", agg_period="W", title=None, legend=None, metric="MSE", scale_data=True):
    # Predicted labels
    agg_title = "Aggregated signals"
    left, right = filterData_v3(df_all, query=query, on=on)
    #     Make required shifts
    left_agg = left.shift(shift[0], freq="D")
    right_agg = right.shift(shift[1], freq="D")
    #     Aling TimeIndexes for correct aggregation.
    st_date = max(left_agg.index[0], right_agg.index[0])
    end_date = min(left_agg.index[-1], right_agg.index[-1])
    left_agg = left_agg[(left_agg.index >= st_date) & (left_agg.index <= end_date)]
    right_agg = right_agg[(right_agg.index >= st_date) & (right_agg.index <= end_date)]
    #     Makre required aggregation
    left_agg = left_agg.resample(agg_period).agg({"amount": sum})
    right_agg = right_agg.resample(agg_period).apply({"amount": sum})
    if legend is not None:
        left_agg.name = legend[0] + legend_postfix[agg_period]
        right_agg.name = legend[1] + legend_postfix[agg_period]
        agg_title += legend_postfix[agg_period]
    #     Add info about aggregation period to legen texts
    else:
        try:
            left_agg.name = left.name + legend_postfix[agg_period]
            right_agg.name = right.name + legend_postfix[agg_period]
            agg_title += legend_postfix[agg_period]
        except KeyError as e:
            left_agg.name = left.name + ", " + agg_period
            right_agg.name = right.name + ", " + agg_period
            agg_title += ", " + agg_period

    plotAmounts([left_agg, right_agg], agg_title, filename=title + "_" + agg_period,
                corr_score=left_agg.amount.corr(right_agg.amount))
    print(f"Correlation for given query and given shifts {shift} is \
          {left_agg.amount.corr(right_agg.amount)}")

    all_data = left_agg.join(right_agg, lsuffix="_X", rsuffix="_Y", how="inner")
    if scale_data:
        from sklearn.preprocessing import minmax_scale
        all_data["amount_Y"] = minmax_scale(all_data["amount_Y"])
        all_data["amount_X"] = minmax_scale(all_data["amount_X"])
    return all_data


def NRMSD(true_labels, pred_labels):
    mse = mean_squared_error(true_labels, pred_labels)
    return np.sqrt(mse) / (np.subtract(*np.percentile(true_labels, [75, 25])))


def evaluate_model(df, metric="RMSE", n_runs=5):
    scores = list()
    lr = LinearRegression()
    #     train, test = train_test_split(all_data, test_size=0.2, random_state=1)
    #     lr.fit(train.iloc[:, 0].values.reshape(-1, 1), train.iloc[:, 1].values.reshape(-1, 1))
    #     print("Coefficients in constructed linear regression model are: :", lr.coef_)
    #     tax_predicted = lr.predict(test[["amount_X"]])
    #     print(f"MAPE score is {MAPE(test[['amount_Y']], tax_predicted)}")
    #     plotScatter(train, test, labels=(left_agg.name, right_agg.name))
    if "amount_X" not in list(df) or "amount_Y" not in list(df):
        raise KeyError(f"Could not find the columns with X and Y in the given dataset... Titles are {list(df)}, while "
                       f"'amount_X' and 'amount_Y' required!")
    for r_s in range(n_runs):
        #         Make new split
        train, test = train_test_split(df, test_size=0.2, random_state=r_s)
        lr.fit(train.iloc[:, 0].values.reshape(-1, 1), train.iloc[:, 1].values.reshape(-1, 1))
        #         print("Coefficients in constructed linear regression model are: :", lr.coef_)
        tax_predicted = lr.predict(test[["amount_X"]])
        if metric == "MSE":
            scores.append(mean_squared_error(test[['amount_Y']], tax_predicted))
        elif metric == "RMSE":
            scores.append(np.sqrt(mean_squared_error(test[['amount_Y']], tax_predicted)))
        elif metric == "NRMSD":
            scores.append(NRMSD(test[['amount_Y']], tax_predicted))
    return scores
