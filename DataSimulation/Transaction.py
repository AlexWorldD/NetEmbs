import sqlite3


class Transaction(object):

    def __init__(self, name, env):
        print "Transaction process %s " % (name)
        self.conn = sqlite3.connect('sample.db')
        self.c = self.conn.cursor()

        self.env = env

    def addRecord(self, name, valueCorrect, valueIncorrect, tid):
        self.c.execute("INSERT INTO EntryRecords (TID, Name, ValueCorrect, ValueIncorrect) VALUES (?, ?, ?, ?)", (tid, name, valueCorrect, valueIncorrect))
        self.conn.commit()
        return self.c.lastrowid

        # return 1

    def new(self):
        self.c.execute("INSERT INTO JournalEntries (Time, Name) VALUES(?,?)", (self.env.now, self.name))
        self.conn.commit()
        return self.c.lastrowid
        # return 1
