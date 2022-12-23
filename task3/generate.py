import sys
import time
import random
import argparse
from collections import defaultdict
sys.setrecursionlimit(100000)

class PCFG(object):
    def __init__(self):
        self._rules = defaultdict(list)
        self._sums = defaultdict(float)
        self._tree_structure = False

    def add_rule(self, lhs, rhs, weight):
        assert(isinstance(lhs, str))
        assert(isinstance(rhs, list))
        self._rules[lhs].append((rhs, weight))
        self._sums[lhs] += weight

    @classmethod
    def from_file(cls, filename):
        grammar = PCFG()
        with open(filename) as fh:
            for line in fh:
                line = line.split("#")[0].strip()
                if not line: continue
                w,l,r = line.split(None, 2)
                r = r.split()
                w = float(w)
                grammar.add_rule(l,r,w)
        return grammar

    def is_terminal(self, symbol): return symbol not in self._rules

    def setTreesTructure(self, is_tree_structure):
        self._tree_structure = is_tree_structure

    def gen(self, symbol):
        if self.is_terminal(symbol): return symbol
        else:
            expansion = self.random_expansion(symbol)
            return_output = " ".join(self.gen(s) for s in expansion)

            if self._tree_structure:
               return_output = "(" + symbol + " " + return_output + ")"

        return return_output

    def random_sent(self):
        return self.gen("ROOT")

    def random_expansion(self, symbol):
        """
        Generates a random RHS for symbol, in proportion to the weights.
        """
        p = random.random() * self._sums[symbol]
        for r,w in self._rules[symbol]:
            p = p - w
            if p < 0: return r
        return r

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--number', type=str, default=1, help='Number of sentences')
    parser.add_argument('-t', action='store_true')
    args, unknown = parser.parse_known_args()
 
    return (args, unknown)

if __name__ == '__main__':
    args, f_name = parse_arguments()
    pcfg = PCFG.from_file(f_name[0])
    pcfg.setTreesTructure(args.t)

    for i in range(int(args.number)):
        print(f'{i + 1}. {pcfg.random_sent()}')
