# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
words_cloud.py
Created by lex at 2019-07-14.
"""
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud


# Count most frequent FA names in the given DataFrame OR FA names with the highest amount
def findMostCommonFAs_v2(df, labels_column="label", words_column="FA_Name", amount_column="amount", sort_mode="freq",
                         n_top=4, vis=False, folder=""):
    selected_size = df.ID.nunique()

    sns.set_context("paper", font_scale=2.3)
    if labels_column not in list(df):
        raise KeyError(
            f"Given column name {labels_column} is not presented in the given DataFrame! Only allows: {list(df)}!")
    if "from" not in list(df):
        raise KeyError(f"Please ensure that column 'from' is presented in your DataFrame!")
    for name, group in df.groupby(labels_column):
        print(f"Current cluster label is {name}, in selected zone it's "
              f"{round(group.ID.nunique() / selected_size,3) * 100}% of all samples")
        gr = group.groupby([words_column, "from"])
        counts = gr.size().to_frame(name='counts')
        all_stat = counts.join(gr.agg({amount_column: sum, 'Debit': lambda x: list(x), 'Credit': lambda x: list(x)})
            .rename(
            columns={amount_column: 'amount_sum', 'Debit': 'Debit_list', 'Credit': 'Credit_list'})) \
            .reset_index()
        if sort_mode == "freq":
            all_stat.sort_values(['counts', words_column], ascending=False, inplace=True)
        elif sort_mode == "amount":
            all_stat.sort_values(['amount_sum', words_column], ascending=False, inplace=True)
        #             Store all statistict for N_TOP values as dictionary for further visualization
        text = {"Left": [(x[0], x[2], x[3], x[5]) for x in all_stat[all_stat["from"] == True].values[:n_top]],
                "Right": [(x[0], x[2], x[3], x[4]) for x in all_stat[all_stat["from"] == False].values[:n_top]]}
        if vis:
            i = 0
            fig, axes = plt.subplots(2, 2)
        for key, data in text.items():
            if sort_mode == "freq":
                #             Take the most frequent FA names
                to_vis = [(str(item[0]), item[1]) for item in data]
            elif sort_mode == "amount":
                to_vis = [(str(item[0]), item[2]) for item in data]
            #             print(key, "--->", [item[:3] for item in data])
            if vis:
                #                 WordClouds
                axes[0, i].set_title(key, size=24)
                wc = WordCloud(background_color="white", width=800, height=400, max_font_size=84, min_font_size=14,
                               repeat=False, relative_scaling=0.8, max_words=100)
                if len(to_vis) > 0:
                    wc.generate_from_frequencies(dict(to_vis))
                else:
                    continue
                axes[0, i].axis("off")
                axes[0, i].imshow(wc, interpolation="bilinear")
                #                 Histograhm
                [sns.distplot(item[3], label=item[0], kde=False, bins=50, ax=axes[1, i], hist_kws={"range": (0, 1.0)})
                 for item in data if len(item[3]) > 10]
                axes[1, i].legend(frameon=False)
                axes[1, i].spines['right'].set_visible(False)
                axes[1, i].spines['top'].set_visible(False)
                axes[1, i].xaxis.set_ticks_position('bottom')
                axes[1, i].yaxis.set_ticks_position('left')
                axes[1, i].set_xlim((0, 1.0))
                i += 1
        if vis:
            import datetime
            plt.tight_layout()
            # plt.savefig("img/WordClouds/" + labels_column + str(list(df[labels_column].unique())) + str(datetime.datetime.now()) + ".png", dpi=140, pad_inches=0.01)
            plt.show()
