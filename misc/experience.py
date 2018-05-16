from collections import namedtuple

Transition = namedtuple("Transition", ["s1", "m1", "a", "s2", "m2", "r"])
MacroTransition = namedtuple("MacroTransition", ["s1", "m1", "i", "a", "s2", "m2", "r", "vpred"])
