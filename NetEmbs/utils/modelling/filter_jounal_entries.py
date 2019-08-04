# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
filter_jounal_entries.py
Created by lex at 2019-08-04.
"""
import pandas as pd
from typing import Optional, List, Dict


def filter_data(df: pd.DataFrame, query: Optional[List[Dict]] = None,
                on: Optional[str] = "GroundTruth"):
    if query is None:
        query = [{"select": ["ALL"],
                  "_with": {"FA_Name": "Revenue"}},
                 {"select": ["ALL"], "_with": {"FA_Name": "Tax"}}]
    result = list()
    for q in query:
        postfix = {"FA_Name": None, "accountDesc": None, "flow": None}
        if q["select"] is None or q["select"] == ["ALL"] or q["select"] == "ALL":
            cur_df = df
            cur_df.name = "No processes information"
        else:
            cur_df = df[df[on].isin(q["select"])]
            if on == "GroundTruth":
                cur_df.name = "Expert label"
                if len(q["select"]) > 1:
                    cur_df.name += "s"
            elif on == "label":
                cur_df.name = "Cluster"
                if len(q["select"]) > 1:
                    cur_df.name += "s"
            cur_df.name += " " + str(q["select"])[1:-1]
        tmp_name = cur_df.name
        if q["_with"] is not None:
            for key, value in q["_with"].items():
                try:
                    cur_df = cur_df[cur_df[key] == value]
                    postfix[key] = str(value)
                except KeyError:
                    raise KeyError(f"{key} is not in a columns titles!")

        if postfix["FA_Name"] is not None:
            tmp_name += f" – {postfix['FA_Name']}"
        elif postfix["accountDesc"] is not None:
            tmp_name += f" – {postfix['accountDesc']}"
        if postfix["flow"] is not None:
            tmp_name += f"({postfix['flow']})"
        cur_df.name = tmp_name
        result.append(cur_df)
    if len(result) == 1:
        return result[0]
    else:
        return tuple(result)
