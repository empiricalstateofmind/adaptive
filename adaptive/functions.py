from collections import defaultdict
from inspect import isfunction
import numpy as np

def degree(simulation):
    """ Returns degree distribution of a simulation"""
    return simulation.A.sum(axis=1)

def mean_degree(simulation):
    """"""
    return degree(simulation).mean()

def opinion(simulation):
    """"""
    return simulation.S 

def mean_opinion(simulation):
    """"""
    return opinion(simulation).mean()

def excess_opinion(simulation):
    """"""
    return -simulation.S * (simulation.A @ simulation.S)

def mean_excess_opinion(simulation):
    """"""
    return excess_opinion(simulation).mean()

def active_links(simulation):
    """"""

    alinks = ((simulation.S[:,np.newaxis] * simulation.A * simulation.S[:,np.newaxis].T)==-1).sum()
    total_edges = simulation.A.sum() #double check both these for double counting!
    if total_edges == 0: 
        return 0
    else:
        return alinks / total_edges

def process_timeseries(simulation, time_range, functions):
    """
    Calculates properties of a simulation over a given time range.
    
    Input:
        simulation (AdaptiveVoter):
        time_range (np.array):
        functions (function or list):

    Returns:
    """

    store = defaultdict(dict)
    sp = 0

    if isfunction(functions):
        functions = [functions]

    simulation.build(0)

    for ix,t in enumerate(time_range):

        changed = simulation.build(t)

        for func in functions:
            store[func.__name__][t] = func(simulation)
        
    return store

