class Eval:
    def __init__(self, label):
        self.label = label
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0

    def add(self, gold, predicted):
        if gold and predicted: self.tp += 1
        if gold and (not predicted): self.fn += 1
        if (not gold) and predicted: self.fp += 1
        if (not gold) and (not predicted): self.tn += 1

    @property
    def acc(self):
        return (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn)

    @property
    def pr(self):
        if (not self.tp) or (self.fp is None):
            return self.tp # zero or None...
        return self.tp / (self.tp + self.fp)

    @property
    def re(self):
        if (not self.tp) or (self.fn is None):
            return self.tp # zero or None...
        return self.tp / (self.tp + self.fn)

    @property
    def f(self):
        if (not self.pr) or (not self.re):
            return self.pr
        return (2 * self.pr * self.re) / (self.pr + self.re)

    def eval_str(self):
        return "{self.label:15s} Pr:{self.pr:.2f} Re:{self.re:.2f} F1:{self.f:.2f}".format(**locals())


def combine(evals, label="Total") -> Eval:
    result = Eval(label)
    for e in evals:
        for attr in ["tp", "tn", "fp", "fn"]:
            setattr(result, attr, getattr(result, attr) + getattr(e, attr))
    return result