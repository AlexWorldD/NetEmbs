# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
Transaction.py
Created by lex at 2019-03-24.
"""

import sqlite3
from Simulation.CONFIG import *
import random
from Simulation.utils import randomString
from Simulation import CONFIG


class Transaction(object):

    def __init__(self, name, env):
        if PRINT:
            print("Transaction process ", name)
        self.conn = sqlite3.connect(DB_PATH)
        self.c = self.conn.cursor()
        self.env = env
        self.name = name

    def addRecord(self, name, fa_name, value, tid):
        # Name - Unique name of Financial Account, eg. Product A and Product B;
        # FA_Name - Group names such as Revenue, Tax etc.
        self.c.execute("INSERT INTO EntryRecords (TID, Name, FA_Name, Value) VALUES (?, ?, ?, ?)",
                       (tid, name, fa_name, value))
        self.conn.commit()
        return self.c.lastrowid

        # return 1

    def noise(self, base_amount, id, tid):
        noise = {"left": {"def": 0.0}, "right": {"def": 0.0}}
        # Add noise of type 2 when noisy financial accounts with very small amounts
        for _ in range(np.random.choice(ks_l, p=pds_l)):
            # noise_name = randomString(6)
            noise_name = np.random.choice(CONFIG.noisy_left)
            noise_amount = base_amount * random.uniform(0.0, NOISE_Type2["noise_amplitude"])
            # Add noise for credited side
            noise["left"][noise_name] = noise_amount
            self.addRecord(noise_name + "_" + str(id), noise_name, -noise_amount, tid)
        for _ in range(np.random.choice(ks_r, p=pds_r)):
            # noise_name = randomString(6)
            noise_name = np.random.choice(CONFIG.noisy_right)
            noise_amount = base_amount * random.uniform(0.0, NOISE_Type2["noise_amplitude"])
            # Add noise for credited side
            self.addRecord(noise_name + "_" + str(id), noise_name, noise_amount, tid)
            noise["right"][noise_name] = noise_amount
        return noise

    def noise2(self, base_amount, id, tid):
        noise = {"left": {"def": 0.0}, "right": {"def": 0.0}}
        # Add noise of type 2 when noisy financial accounts with very small amounts
        if random.random() < NOISE_Type2["freq"]:
            # TODO check other distribution for number of noisy FAs per BP
            for _ in range(int(random.uniform(0.0, NOISE_Type2["num_amplitude"]))):
                noise_name = randomString(6)
                noise_amount = base_amount * random.uniform(0.0, NOISE_Type2["noise_amplitude"])
                noise_side = np.random.choice(["left", "right"],
                                              p=[NOISE_Type2["proportion"], 1.0 - NOISE_Type2["proportion"]])
                if noise_side == "left":
                    # Add noise for credited side
                    noise["left"][noise_name] = noise_amount
                    self.addRecord(noise_name + "_" + str(id), noise_name, -noise_amount, tid)
                elif noise_side == "right":
                    self.addRecord(noise_name + "_" + str(id), noise_name, noise_amount, tid)
                    noise["right"][noise_name] = noise_amount
        return noise

    def new(self, postfix):
        self.c.execute("INSERT INTO JournalEntries (Time, Name, FA_Name) VALUES(?,?,?)",
                       (self.env.now, self.name + str(postfix), self.name))
        self.conn.commit()
        return self.c.lastrowid
        # return 1
