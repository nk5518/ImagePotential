# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 16:35:11 2020

@author: non_k
"""

#%%
from time import perf_counter, sleep
from functools import wraps

#%%
def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = perf_counter()
        value = func(*args, **kwargs)
        end = perf_counter()
        runtime = end - start
        print(f'{func.__name__} : {runtime} s')
        return value 
    return wrapper 


def debugger(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        args_repr = [repr(a) for a in args]
        kwargs_repr = [f'{k} = {v!r}' for k, v in kwargs.items()]
        signature = ', '.join(args_repr + kwargs_repr)
        print(f'calling {func._name_}({signature})')
        value = func(*args, **kwargs)
        print(f'{func._name_} returned {value!r}')
        return value
    return wrapper 


def delay(func, t_delay = 1):
    @wraps(func)
    def wrapper(*args, **kwargs):
        sleep(t_delay)
        return func(*args, **kwargs)
    return wrapper


def cache(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        cache_key = args + tuple(kwargs.items())
        if cache_key not in wrapper.cache:
            wrapper.cache[cache_key] = \
                func(*args, **kwargs)
        return wrapper.cache[cache_key]
    wrapper.cache = dict()
    return wrapper 

