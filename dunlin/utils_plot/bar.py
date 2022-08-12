import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy             as np
import pandas            as pd

def plot_bar(ax, df, xlabel='', ylabel='', width=0.4, bottom=0, 
             horizontal=False, stacked=False, **bar_args):
    if horizontal:
        if stacked:
            return plot_barh_stack(ax, df, xlabel, ylabel, width, bottom, **bar_args)
        else:
            return plot_barh_nostack(ax, df, xlabel, ylabel, width, bottom, **bar_args)
    else:
        if stacked:
            return plot_bar_stack(ax, df, xlabel, ylabel, width, bottom, **bar_args)
        else:
            return plot_bar_nostack(ax, df, xlabel, ylabel, width, bottom, **bar_args)

def bar_args_per_columns(bar_args, column):
    bar_args_ = {k: v[column] if hasattr(v, 'items') else v for k, v in bar_args.items()}        
    return bar_args_

def format_ylabel(ylabel, column):
    if callable(ylabel):
        label = ylabel(column)
    elif hasattr(ylabel, 'format'):
        label = ylabel.format(column)
    else:
        label = column
    
    return label
    
def format_xlabel(xlabel, df):
    if callable(xlabel):
        xticks = [xlabel(i) for i in df.index]
    elif hasattr(xlabel, 'format'):
        xticks = [xlabel.format(i) for i in df.index]
    else:
        xticks = list(df.index)
    
    return xticks
    
def plot_bar_stack(ax, df, xlabel='', ylabel='', width=0.4, bottom=0, rot=0, **bar_args):
    nrows, ncols = df.shape
    xcoords      = np.array(list(range(nrows)), dtype=np.float64)
    bottom       = np.zeros(nrows) + bottom
    result       = []

    #Iterate and plot
    for column in df.columns:
        #Format the y labels
        label = format_ylabel(ylabel, column)
        
        #Extract column-based bar args
        bar_args_ = bar_args_per_columns(bar_args, column)
        
        #Plot
        heights  = df[column].values
        temp     = ax.bar(xcoords, 
                          heights, 
                          bottom=bottom, 
                          label=label, 
                          width=width,
                          **bar_args_
                          )
        bottom  += heights
        
        #Update result
        result.append(temp)
        
    #Format the x ticks
    xticks = format_xlabel(xlabel, df)
    
    ax.xaxis.set_major_locator(mtick.MaxNLocator(nbins=nrows))
    ax.set_xticks(xcoords, xticks, rotation=rot)
    
    return result

def plot_bar_nostack(ax, df, xlabel='', ylabel='', width=0.4, bottom=0, rot=0, **bar_args):
    nrows, ncols = df.shape
    xcoords      = np.array(list(range(nrows)), dtype=np.float64)*ncols
    result       = []
    xcache       = []
    
    for column in df.columns:
        xcache.append(list(xcoords))
        
        #Format the y labels
        label = format_ylabel(ylabel, column)
        
        #Extract column-based bar args
        bar_args_ = bar_args_per_columns(bar_args, column)
        
        #Plot
        heights  = df[column].values
        temp     = ax.bar(xcoords, 
                          heights, 
                          label=label, 
                          bottom=bottom, 
                          width=width, 
                          **bar_args_
                          )
        xcoords += width

        result.append(temp)
        
    #Format the x ticks
    xticks = format_xlabel(xlabel, df)
    
    xlocs = np.mean(xcache, axis=0)
    ax.xaxis.set_major_locator(mtick.MaxNLocator(nbins=nrows))
    ax.set_xticks(xlocs, xticks, rotation=rot)

    return result

def plot_barh_nostack(ax, df, xlabel='', ylabel='', width=0.4, bottom=0, rot=0, **bar_args):
    nrows, ncols = df.shape
    xcoords      = np.array(list(range(nrows)), dtype=np.float64)*ncols
    result       = []
    xcache       = []
    
    for column in df.columns:
        xcache.append(list(xcoords))
        
        #Format the y labels
        label = format_ylabel(ylabel, column)
        
        #Extract column-based bar args
        bar_args_ = bar_args_per_columns(bar_args, column)
        
        #Plot
        heights  = df[column].values
        temp     = ax.barh(xcoords, 
                           heights, 
                           label=label, 
                           left=bottom, 
                           height=width, 
                           **bar_args_
                           )
        xcoords += width

        result.append(temp)
        
    #Format the x ticks
    xticks = format_xlabel(xlabel, df)
    
    xlocs = np.mean(xcache, axis=0)
    ax.yaxis.set_major_locator(mtick.MaxNLocator(nbins=nrows))
    ax.set_yticks(xlocs, xticks, rotation=rot)

    return result

def plot_barh_stack(ax, df, xlabel='', ylabel='', width=0.4, bottom=0, rot=0, **bar_args):
    nrows, ncols = df.shape
    xcoords      = np.array(list(range(nrows)), dtype=np.float64)
    result       = []
    
    for column in df.columns:
        #Format the y labels
        label = format_ylabel(ylabel, column)
        
        #Extract column-based bar args
        bar_args_ = bar_args_per_columns(bar_args, column)
        
        #Plot
        heights  = df[column].values
        temp     = ax.barh(xcoords, 
                           heights, 
                           label=label, 
                           left=bottom, 
                           height=width, 
                           **bar_args_
                           )
        bottom  += heights

        result.append(temp)
        
    #Format the x ticks
    xticks = format_xlabel(xlabel, df)
    
    ax.yaxis.set_major_locator(mtick.MaxNLocator(nbins=nrows))
    ax.set_yticks(xcoords, xticks, rotation=rot)

    return result

