# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
mostCommonFAs.py
Created by lex at 2019-05-10.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter


def findMostCommonFAs(df, labels_column="label", words_column="FA_Name", amount_column="amount", sort_mode="freq",
                      n_top=2, vis=False):
    if vis:
        from wordcloud import WordCloud
    for name, group in df.groupby(labels_column):
        print("Current cluster label is ", name)
        cur_data = group.groupby(words_column, as_index=False)[amount_column].agg(sum)
        fa_amounts = dict(zip(cur_data[words_column].values, cur_data[amount_column].values))
        text = {"Left": [(item[0], item[1], fa_amounts[item[0]]) for item in
                         Counter(group[group["from"] == True][words_column].values).items()],
                "Right": [(item[0], item[1], fa_amounts[item[0]]) for item in
                          Counter(group[group["from"] == False][words_column].values).items()]}
        if vis:
            i = 1
            fig = plt.figure()
        for key, data in text.items():
            if sort_mode == "freq":
                #             Take the most frequent FA names
                output = sorted(data, key=lambda d: -d[1])
                to_vis = [(item[0], item[1]) for item in output]
            elif sort_mode == "amount":
                output = sorted(data, key=lambda d: -d[2])
                to_vis = [(item[0], item[2]) for item in output]
            print(key, "--->", output[:n_top])
            if vis:
                ax = fig.add_subplot(1, 2, i)
                i += 1
                ax.set_title(key)
                wc = WordCloud(background_color="white", repeat=False, relative_scaling=0.75, max_words=100)
                wc.generate_from_frequencies(dict(to_vis))
                ax.axis("off")
                ax.imshow(wc, interpolation="bilinear")
        if vis:
            plt.show()
