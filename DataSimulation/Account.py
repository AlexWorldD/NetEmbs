from Observer import *


class Account(Observable):
    # """An account is a container of monetary values"""
    container = None

    def __init__(self, env, asimpy, name, initialStock=None):
        Observable.__init__(self)

        print "Initialize container with name: %s" % (name)
        if initialStock != None:
            self.containerCorrect = asimpy.Container(env, init=initialStock)
            self.containerWrong = asimpy.Container(env)
        else:
            self.containerCorrect = asimpy.Container(env)
            self.containerWrong = asimpy.Container(env)
        self.env = env
        self.name = name

    def __str__(self):
        return "Account name: %s \t\t\t level correct: %d wrong: %d" % (
        self.name, self.containerCorrect.level, self.containerWrong.level)

    def __repr__(self):
        return "%s" % self.name


class PersonnelExpensesAccount(Account):
    def __init__(self, env, asimpy, name, initialStock=None):
        Account.__init__(self, env, asimpy, name, initialStock)
        self.payrollObserver = PersonnelExpensesAccount.PayrollObserver(self)
        self.purchaseObserver = PersonnelExpensesAccount.PurchaseObserver(self)

    def processPersonnelExpenses(self, payroll):
        [sal_c, sal_w, eb_c, eb_w, tax_c, tax_w] = payroll

        if sal_c > 0:
            yield self.containerCorrect.put(sal_c)

        if sal_w > 0:
            yield self.containerWrong.put(sal_w)

        self.setChanged()

        for obs in self.notifyObservers():
            yield obs

    def processPurchase(self, purchase):
        [tp, tp_w, oe, oe_w, ppe, ppe_w, pe, pe_w] = purchase

        if pe > 0:
            yield self.containerCorrect.put(pe)

        if pe_w > 0:
            yield self.containerWrong.put(pe_w)

        self.setChanged()

        for obs in self.notifyObservers():
            yield obs

    class PurchaseObserver(Observer):

        def __init__(self, outer):
            self.outer = outer

        def update(self, observable, args):
            for obs in self.outer.processPurchase(args):
                yield obs

    class PayrollObserver(Observer):

        def __init__(self, outer):
            self.outer = outer

        def update(self, observable, args):
            for obs in self.outer.processPersonnelExpenses(args):
                yield obs


class EBPayablesAccount(Account):

    def __init__(self, env, asimpy, name, initialStock=None):
        Account.__init__(self, env, asimpy, name, initialStock)
        self.payrollObserver = EBPayablesAccount.PayrollObserver(self)
        self.payrollDisbursementObserver = EBPayablesAccount.PayrollDisbursementObserver(self)

    def processPayroll(self, payroll):
        [sal_c, sal_w, eb_c, eb_w, tax_c, tax_w] = payroll

        if eb_c > 0:
            yield self.containerCorrect.put(eb_c)

        if eb_w > 0:
            yield self.containerWrong.put(eb_w)

        self.setChanged()

        for obs in self.notifyObservers():
            yield obs

    def processPayrollDisbursement(self, payroll):

        # small hack, cause of the time delay the amounts are two low if they come from the triggered event
        eb = self.containerCorrect.level
        eb_w = self.containerWrong.level

        if eb > 0:
            yield self.containerCorrect.get(eb)

        if eb_w > 0:
            yield self.containerWrong.get(eb_w)

        self.setChanged()

        for obs in self.notifyObservers():
            yield obs

    class PayrollDisbursementObserver(Observer):

        def __init__(self, outer):
            self.outer = outer

        def update(self, observable, args):
            for obs in self.outer.processPayrollDisbursement(args):
                yield obs

    class PayrollObserver(Observer):

        def __init__(self, outer):
            self.outer = outer

        def update(self, observable, args):
            for obs in self.outer.processPayroll(args):
                yield obs


class CashAccount(Account):

    def __init__(self, env, asimpy, name, initialStock=None):
        Account.__init__(self, env, asimpy, name, initialStock)
        self.collectionsObserver = CashAccount.CollectionsObserver(self)
        self.taxDisbursementObserver = CashAccount.TaxDisbursementObserver(self)
        self.payrollObserver = CashAccount.PayrollObserver(self)

    def processCollection(self, collection):

        [tr, tr_w] = collection

        if tr > 0:
            yield self.containerCorrect.put(tr)
        if tr_w > 0:
            yield self.containerWrong.put(tr_w)

        self.setChanged()

        for obs in self.notifyObservers():
            yield obs

    def processTaxDisbursement(self, tax):
        [tax_c, tax_w] = tax

        if tax_c > 0:
            yield self.containerCorrect.get(tax_c)

        if tax_w > 0:
            yield self.containerWrong.get(tax_w)

        self.setChanged()

        for obs in self.notifyObservers():
            yield obs

    def processPayroll(self, payroll):
        [eb, eb_w] = payroll

        if eb > 0:
            yield self.containerCorrect.get(eb)

        if eb_w > 0:
            yield self.containerWrong.get(eb_w)

        self.setChanged()

        for obs in self.notifyObservers():
            yield obs

    class PayrollObserver(Observer):

        def __init__(self, outer):
            self.outer = outer

        def update(self, observable, args):
            for obs in self.outer.processPayroll(args):
                yield obs

    class CollectionsObserver(Observer):

        def __init__(self, outer):
            self.outer = outer

        def update(self, observable, args):
            for obs in self.outer.processCollection(args):
                yield obs

    class TaxDisbursementObserver(Observer):

        def __init__(self, outer):
            self.outer = outer

        def update(self, observable, args):
            for obs in self.outer.processTaxDisbursement(args):
                yield obs


class TradeReceivablesAccount(Account):
    def __init__(self, env, asimpy, name, initialStock=None):
        Account.__init__(self, env, asimpy, name, initialStock)

        self.salesObserver = TradeReceivablesAccount.SalesObserver(self)
        self.collectionsObserver = TradeReceivablesAccount.CollectionsObserver(self)

    def salesOrder(self, lastTransactionDetails):
        [tr, tr_w, rev, rev_w, tax, tax_w, label] = lastTransactionDetails

        if tr > 0:
            yield self.containerCorrect.put(tr)
        if tr_w > 0:
            yield self.containerWrong.put(tr_w)
        self.setChanged()

        for obs in self.notifyObservers():
            yield obs

    def collectionsOrder(self, collection):
        [tr, tr_w] = collection
        if tr > 0:
            yield self.containerCorrect.get(tr)
        if tr_w > 0:
            yield self.containerWrong.get(tr_w)

        self.setChanged()

        for obs in self.notifyObservers():
            yield obs

    class SalesObserver(Observer):

        def __init__(self, outer):
            self.outer = outer

        def update(self, observable, args):
            for obs in self.outer.salesOrder(args):
                yield obs

    class CollectionsObserver(Observer):

        def __init__(self, outer):
            self.outer = outer

        def update(self, observable, args):
            for obs in self.outer.collectionsOrder(args):
                yield obs


class TaxPayablesAccount(Account):
    def __init__(self, env, asimpy, name, initialStock=None):
        Account.__init__(self, env, asimpy, name, initialStock)
        self.salesObserver = TaxPayablesAccount.SalesObserver(self)
        self.taxDisbursementObserver = TaxPayablesAccount.TaxDisbursementObserver(self)
        self.payrollObserver = TaxPayablesAccount.PayrollObserver(self)

    def processTaxExpenses(self, payroll):
        [sal_c, sal_w, eb_c, eb_w, tax_c, tax_w] = payroll

        if tax_c > 0:
            yield self.containerCorrect.put(tax_c)

        if tax_w > 0:
            yield self.containerWrong.put(tax_w)

        self.setChanged()

        for obs in self.notifyObservers():
            yield obs

    class PayrollObserver(Observer):

        def __init__(self, outer):
            self.outer = outer

        def update(self, observable, args):
            for obs in self.outer.processTaxExpenses(args):
                yield obs

    def salesOrder(self, lastTransactionDetails):
        [tr, tr_w, rev, rev_w, tax, tax_w, label] = lastTransactionDetails

        if tax > 0:
            yield self.containerCorrect.put(tax)
        if tax_w > 0:
            yield self.containerWrong.put(tax_w)
        self.setChanged()

        for obs in self.notifyObservers():
            yield obs

    def processTaxDisbursement(self, tax):
        # [tax_c, tax_w] = tax

        tax_c = self.containerCorrect.level * 0.8
        tax_w = self.containerWrong.level * 0.8

        if tax_c > 0:
            yield self.containerCorrect.get(tax_c)

        if tax_w > 0:
            yield self.containerWrong.get(tax_w)

        self.setChanged()

        for obs in self.notifyObservers():
            yield obs

    class TaxDisbursementObserver(Observer):

        def __init__(self, outer):
            self.outer = outer

        def update(self, observable, args):
            for obs in self.outer.processTaxDisbursement(args):
                yield obs

    class SalesObserver(Observer):

        def __init__(self, outer):
            self.outer = outer

        def update(self, observable, args):
            for obs in self.outer.salesOrder(args):
                yield obs


class PrepaidExpensesAccount(Account):
    def __init__(self, env, asimpy, name, initialStock=None):
        Account.__init__(self, env, asimpy, name, initialStock)

        self.purchaseObserver = OtherExpensesAccount.PurchaseObserver(self)

    def processPurchase(self, purchase):
        [tp, tp_w, oe, oe_w, ppe, ppe_w, pe, pe_w] = purchase

        if ppe > 0:
            self.containerCorrect.put(ppe)

        if ppe_w > 0:
            self.containerWrong.put(ppe_w)

        self.setChanged()

        for obs in self.notifyObservers():
            yield obs

    class PurchaseObserver(Observer):

        def __init__(self, outer):
            self.outer = outer

        def update(self, observable, args):
            for obs in self.outer.processPurchase(args):
                yield obs


class OtherExpensesAccount(Account):

    def __init__(self, env, asimpy, name, initialStock=None):
        Account.__init__(self, env, asimpy, name, initialStock)

        self.purchaseObserver = OtherExpensesAccount.PurchaseObserver(self)

    def processPurchase(self, purchase):
        [tp, tp_w, oe, oe_w, ppe, ppe_w, pe, pe_w] = purchase

        if oe > 0:
            self.containerCorrect.put(oe)

        if oe_w > 0:
            self.containerWrong.put(oe_w)

        self.setChanged()

        for obs in self.notifyObservers():
            yield obs

    class PurchaseObserver(Observer):

        def __init__(self, outer):
            self.outer = outer

        def update(self, observable, args):
            for obs in self.outer.processPurchase(args):
                yield obs


class DepreciationAccount(Account):
    def __init__(self, env, asimpy, name, initialStock=None):
        Account.__init__(self, env, asimpy, name, initialStock)
        self.depreciationObserver = DepreciationAccount.DepreciationObserver(self)

    def processDepreciation(self, depr):
        [depr, depr_w, fix, fix_w] = depr

        if depr > 0:
            yield self.containerCorrect.put(depr)

        if depr_w > 0:
            yield self.containerWrong.put(depr_w)

        self.setChanged()

        for obs in self.notifyObservers():
            yield obs

    class DepreciationObserver(Observer):
        def __init__(self, outer):
            self.outer = outer

        def update(self, observable, args):
            for obs in self.outer.processDepreciation(args):
                yield obs


class FixedAssetsAccount(Account):

    def __init__(self, env, asimpy, name, initialStock=None):
        Account.__init__(self, env, asimpy, name, initialStock)
        self.fixedAssetsObserver = FixedAssetsAccount.FixedAssetsObserver(self)
        self.depreciationObserver = FixedAssetsAccount.DepreciationObserver(self)

    def processDepreciation(self, depr):
        [depr, depr_w, fix, fix_w] = depr

        if fix > 0:
            yield self.containerCorrect.get(fix)

        if fix_w > 0:
            yield self.containerWrong.get(fix_w)

        self.setChanged()

        for obs in self.notifyObservers():
            yield obs

    class DepreciationObserver(Observer):
        def __init__(self, outer):
            self.outer = outer

        def update(self, observable, args):
            for obs in self.outer.processDepreciation(args):
                yield obs

    def processFixedAssets(self, assets):
        [fix, fix_w, tp, tp_w] = assets

        if fix > 0:
            yield self.containerCorrect.put(fix)

        if fix_w > 0:
            yield self.containerWrong.put(fix_w)

        self.setChanged()

        for obs in self.notifyObservers():
            yield obs

    class FixedAssetsObserver(Observer):
        def __init__(self, outer):
            self.outer = outer

        def update(self, observable, args):
            for obs in self.outer.processFixedAssets(args):
                yield obs


class TradePayablesAccount(Account):
    def __init__(self, env, asimpy, name, initialStock=None):
        Account.__init__(self, env, asimpy, name, initialStock)
        self.purchaseObserver = TradePayablesAccount.PurchaseObserver(self)
        self.purchaseInventoryObserver = TradePayablesAccount.PurchaseInventoryObserver(self)
        self.fixedAssetsObserver = TradePayablesAccount.FixedAssetsObserver(self)

    def purchaseInventoryOrder(self, orderCorrect, orderWrong):
        if orderCorrect > 0:
            yield self.containerCorrect.put(orderCorrect)
        if orderWrong > 0:
            yield self.containerWrong.put(orderWrong)
        self.setChanged()

        for obs in self.notifyObservers():
            yield obs

    def processPurchase(self, purchase):
        [tp, tp_w, oe, oe_w, ppe, ppe_w, pe, pe_w] = purchase

        if tp > 0:
            yield self.containerCorrect.put(tp)

        if tp_w > 0:
            yield self.containerWrong.put(tp_w)

        self.setChanged()

        for obs in self.notifyObservers():
            yield obs

    def processFixedAssets(self, assets):
        [fix, fix_w, tp, tp_w] = assets

        if tp > 0:
            yield self.containerCorrect.put(tp)

        if tp_w > 0:
            yield self.containerWrong.put(tp_w)

        self.setChanged()

        for obs in self.notifyObservers():
            yield obs

    class FixedAssetsObserver(Observer):
        def __init__(self, outer):
            self.outer = outer

        def update(self, observable, args):
            for obs in self.outer.processFixedAssets(args):
                yield obs

    class PurchaseObserver(Observer):

        def __init__(self, outer):
            self.outer = outer

        def update(self, observable, args):
            for obs in self.outer.processPurchase(args):
                yield obs

    class PurchaseInventoryObserver(Observer):

        def __init__(self, outer):
            self.outer = outer

        def update(self, observable, args):
            [c, w] = args
            for obs in self.outer.purchaseInventoryOrder(c, w):
                yield obs


class CostOfSalesAccount(Account):
    def __init__(self, env, asimpy, name, initialStock=None):
        Account.__init__(self, env, asimpy, name, initialStock)
        self.salesObserver = CostOfSalesAccount.SalesObserver(self)

    def processSales(self, transaction):
        [c, w] = transaction

        if c > 0:
            yield self.containerCorrect.put(c)
        if w > 0:
            yield self.containerWrong.put(w)
        self.setChanged()

        for obs in self.notifyObservers():
            yield obs

    class SalesObserver(Observer):
        def __init__(self, outer):
            self.outer = outer

        def update(self, observable, args):
            for obs in self.outer.processSales(args):
                yield obs


class InventoriesAccount(Account):

    def __init__(self, env, asimpy, name, initialStock=None):
        Account.__init__(self, env, asimpy, name, initialStock)

        if initialStock != None:
            self.containerCorrect = asimpy.Container(env, init=initialStock)
            self.containerWrong = asimpy.Container(env)

        self.purchaseObserver = InventoriesAccount.PurchaseObserver(self)
        self.salesObserver = InventoriesAccount.SalesObserver(self)

    def buyStock(self, numberOfStocks):

        [c, w] = numberOfStocks

        if c > 0:
            yield self.containerCorrect.put(c)

        # DOESN'T WORK YET
        # if w > 0:
        # 	yield self.containerWrong.put(w)

        self.setChanged()
        for obs in self.notifyObservers():
            yield obs

    def sellStock(self, salesTransaction):
        [c, w] = salesTransaction

        if c > 0:
            yield self.containerCorrect.get(c)

        # DOESN'T WORK YET
        # if w > 0 :
        # 	yield self.containerWrong.get(w)

        self.setChanged()
        for obs in self.notifyObservers():
            yield obs

    class PurchaseObserver(Observer):

        def __init__(self, outer):
            self.outer = outer

        def update(self, observable, args):
            for obs in self.outer.buyStock(args):
                yield obs

    class SalesObserver(Observer):

        def __init__(self, outer):
            self.outer = outer

        def update(self, observable, args):
            for obs in self.outer.sellStock(args):
                yield obs


class RevenueAccount(Account):

    def __init__(self, env, asimpy, name, initialStock=None):
        Account.__init__(self, env, asimpy, name, initialStock)

        self.salesObserver = RevenueAccount.SalesObserver(self)

    def processSales(self, lastTransactionDetails):
        [tr, tr_w, rev, rev_w, tax, tax_w, label] = lastTransactionDetails
        if rev > 0:
            yield self.containerCorrect.put(rev)
        if rev_w > 0:
            yield self.containerWrong.put(rev_w)

    class SalesObserver(Observer):
        def __init__(self, outer):
            self.outer = outer

        def update(self, observable, args):
            for obs in self.outer.processSales(args):
                yield obs
