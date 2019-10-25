# Author: Yu-Chia Chen <yuchaz@uw.edu>
# LICENSE: Simplified BSD

from __future__ import division, print_function, absolute_import
import time
from functools import wraps
from matplotlib import rc

def isnotebook():
    try:
        from IPython import get_ipython
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except (NameError, ImportError):
        return False      # Probably standard Python interpreter


if isnotebook():
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm
    import matplotlib
    matplotlib.use('agg')


def load_latex_preamble():
    import matplotlib
    matplotlib.rcParams['text.latex.preamble'] = [
        r'\usepackage{amssymb}',
        r'\usepackage{bm}',
        r'\newcommand{\vect}[1]{\mathbf{\bm{#1}}}',
        r'\usepackage{siunitx}'
    ]
load_latex_preamble()


import seaborn as sns
from seaborn import xkcd_rgb as xkcd

colors_dict = ['purple', 'goldenrod', 'scarlet', 'lawn green', 'windows blue',
               'purplish pink', 'orange', 'teal', 'denim', 'tomato red']
marker_list = ['o', '^', 'p', '*', 'P', 'X', 'D']

def setup_color_palettes():
    sns.set(font_scale=1.0)
    sns.set_style('whitegrid',
                 {'font.family':['serif'],
                  'font.serif':['Times New Roman']})

    sns.set_palette(sns.xkcd_palette(colors_dict))
    rc('text', usetex=True)


def current_palettes():
    sns.palplot(sns.color_palette())

def color_hex(idx):
    return xkcd[colors_dict[idx % len(colors_dict)]]


def makrer_loop(idx):
    return marker_list[idx % len(marker_list)]


def time_counter():
    try:
        return time.perf_counter()
    except AttributeError:  # for python 2
        return time.time()


def timec(prefix=None):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time_counter()
            res = func(*args, **kwargs)
            end = time_counter()
            nonlocal prefix
            if prefix is None:
                prefix = 'Time'
            print('{} spent {:.2f}s'.format(prefix, end - start))
            return res
        return wrapper
    return decorator
