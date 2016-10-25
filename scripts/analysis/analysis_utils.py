#!/usr/bin/env python

import re
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from glob import glob as glb
from collections import OrderedDict

def get_meta(l, d):
    """get meta data"""
    layer_re = re.compile('.+ Creating layer (.+)')
    iters_re = re.compile('Testing for\s+(\d+)\s+iterations.')
    th_re = re.compile('Number of OpenMP threads: (\d+)')
    batch_size_re = re.compile('batch_size: (\d+)')
    batch_size_re2 = re.compile('shape .*?dim: (\d+)',re.DOTALL)
    m = layer_re.findall(l)
    if(m is not []): layers = m

    m = iters_re.search(l)
    if(m is not None): d['iterations'] = int(m.group(1))    

    m = th_re.search(l)
    if(m is not None): d['threads'] = int(m.group(1))    

    m = batch_size_re.search(l)
    if(m is not None):
        d['batch size'] = int(m.group(1))
    else:
        m = batch_size_re2.search(l)
        if(m is not None):
            d['batch size'] = int(m.group(1))
    
    d['layers'] = set(layers)
    return d


def get_data(l, entry):
    """get the time of each layer and the total"""
    #total timing
    fd_re = re.compile('.+Average Forward pass:\s+(.+)\s+ms.')
    bd_re = re.compile('.+Average Backward pass:\s+(.+)\s+ms.')
    total_re = re.compile('.+Total Time:\s+(.+)\s+ms.')
    m = fd_re.search(l)
    if(m is not None): entry['avg forward'] = float(m.groups(1)[0])/1e3
    m = bd_re.search(l)
    if(m is not None): entry['avg backward'] = float(m.groups(1)[0])/1e3
    m = total_re.search(l)
    if(m is not None):
        entry['total time'] = float(m.groups(1)[0])/1e3
        entry['time per iteration'] = entry['total time']/entry['iterations']
    # layers timing
    ltime_re = re.compile('.+\s+(\w+)\s+(forward|backward):\s*(.+)\s+ms.')
    m = ltime_re.findall(l)
    layers_data = OrderedDict()
    for lname,direction,val in m:
        layers_data[lname+' '+direction] = float(val)/1e3  
    entry.update(layers_data)
    # layers memory footprint
    for layer in entry['layers']:
        lmem_re = re.compile('Creating layer\s+('+layer+').*Memory required for data:\s+(\d+)',re.DOTALL)
        m = lmem_re.search(l)
        if(m is not None): entry[m.groups(0)[0]+' memory'] = int(m.groups(0)[1])

def get_df(flist):
    """Parse the files in a dataframe"""
    data = []
    for f in flist:
        with open(f, 'r') as fp:
            entry = OrderedDict()
            txt = fp.read() #.split('\n')
            get_meta(txt, entry)
            get_data(txt, entry)
            entry['file path'] =f
            entry['file name'] =f.split('/')[-1]
            data.append(pd.DataFrame([entry], columns=entry.keys()))
    df = pd.concat(data)
    df.reset_index(inplace=True)
    return df

def normalize_batches(df):
    """Normalize the time by the batch size"""
    df_norm = df.copy()
    time_fields = [s for s in df.columns.values if('ward' in s or 'time' in s)]
    for f in time_fields:
        df_norm.loc[:, f] = df.loc[:, f]/df.loc[:, 'batch size']
    return df_norm

def normalize_time(df):
    """Normalize the time by the total time"""
    df_norm = df.copy()
    time_fields = [s for s in df.columns.values if('ward' in s  or 'time per iteration' in s)]
    for f in time_fields:
        df_norm.loc[:, f] = 100.0*df.loc[:, f]/df.loc[:, 'time per iteration']
    del df_norm['total time']
    return df_norm

def group_small_entries(df, threas):
    """Sum insignificant entries in one column that fall
    below the 'threas' percentile of the total time"""
    df_filt = df.copy()
    df_filt['others'] = 0.0
    df_norm_time = normalize_time(df)
    time_fields = [s for s in df.columns.values if('ward' in s)]
    for f in time_fields:
        if(not all(df_norm_time[f].apply(lambda x: x > threas))):
            df_filt['others'] = df_filt['others'] + df_filt[f]
            del df_filt[f]
    return df_filt

def plot_batch_scaling(df, threas, threads=1, arch='', res_path=''):
    """Plot batch scaling from a data frame table"""
    df.drop_duplicates('batch size',inplace=True)
    df_norm_batch = normalize_batches(df)
    df_filt = group_small_entries(df_norm_batch, threas)

    layers_cols = [s for s in df_filt.columns.values if('ward' in s and not 'avg' in s)]
    layers_cols = layers_cols + ['others']
    df_filt.index = df_filt['batch size']
    plt_data = pd.DataFrame(df_filt[layers_cols],index=df_filt['batch size'], columns=layers_cols)

    plt_data= plt_data.sort_index()

    ax = plt_data.plot(kind='bar', stacked=True, rot=0)
    ax.set_ylabel('Stacked time per batch size per iter. (seconds)')
    ax.set_xlabel('Batch size')
    ax.set_title(arch+' batch scaling with '+str(threads)+' threads')
    ax.set_ylim(ymin=0)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(os.path.join(res_path, arch+'_batch scaling_'+str(threads)+'th.jpg'), format='jpg',bbox_inches='tight', dpi=900)
    return ax

def plot_thread_scaling(df, threas, batch_size=1, arch='', res_path=''):
    """Plot thread scaling from a data frame table"""
    df_filt = group_small_entries(df, threas)

    layers_cols = [s for s in df_filt.columns.values if('ward' in s and not 'avg' in s)]
    layers_cols = layers_cols + ['others']
    df_filt.index = df_filt['threads']
    plt_data = pd.DataFrame(df_filt[layers_cols],index=df_filt['threads'], columns=layers_cols)

    plt_data= plt_data.sort_index()

    ax = plt_data.plot(marker='o')
    ax.set_xscale('log',basex=2)
    ax.set_yscale('log')
    ax.set_ylabel('Time per iter. (seconds)')
    ax.set_xlabel('Threads #')
    ax.set_title(arch+' thread scaling with '+str(batch_size)+' batch size')
    ax.set_ylim(ymin=0)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(os.path.join(res_path, arch+'_thread_scaling_'+str(batch_size)+'_batch_size.jpg'), format='jpg',bbox_inches='tight', dpi=900)

    return ax

def plot_comparative(files_loc, threshold, res_path='', title_postfix='', sort_data=True):
    """Plot comparative results of the data frame records"""
    df = get_df(glb(files_loc))
    df_filt = group_small_entries(df, threshold)
    layers_cols = [s for s in df_filt.columns.values if('ward' in s and not 'avg' in s)]
    layers_cols = layers_cols + ['others']    
    df_filt.index = df_filt['file name']
    plt_data = pd.DataFrame(df_filt[layers_cols].transpose(),columns=df_filt['file name'], index=layers_cols)

    if(sort_data):
        plt_data= plt_data.sort_index()
        plt_data.sort_values(by=plt_data.columns.values[0], inplace=True, ascending=False)

    ax = plt_data.plot(kind='bar')
    scaling = plt_data.iloc[:,0]/plt_data.iloc[:,1]
    ax2 = scaling.plot(secondary_y=True, color='k', marker='o', rot=90)
    ax2.set_ylabel('Scaling factor from 1st to 2nd column')
    ax.set_ylabel('Time per iter. (seconds)')
    ax.set_xlabel('File name')
    ax.set_title('Comparative layers time '+title_postfix)
    ax.set_ylim(ymin=0)
    ax.set_xlim(xmin=ax.get_xlim()[0]-0.25, xmax=ax.get_xlim()[1]+0.25)

    # add the total time in the legends
    df['time per iteration no data'] =  df['time per iteration'] - df['data forward'] - df['data backward']
    exp_time = dict(df[['file name', 'time per iteration no data']].values)
    handles, labels = ax.get_legend_handles_labels()
    new_labels = []
    for l in labels:
        new_labels.append(l+' '+str(exp_time[l])+'s/iter (no data)')
    ax.legend(handles, new_labels)#, loc='center left', bbox_to_anchor=(1.15, 0.5))

    plt.savefig(os.path.join(res_path,'comparative_layers_time'+title_postfix+'.jpg'), format='jpg',bbox_inches='tight', dpi=900)
    return ax

def plot_pie(files_loc, threshold, res_path='', title_postfix='', sort_data=True):
    """Plot a time breakdown pie chart from a data frame record"""
    df = get_df(glb(files_loc))
    df_filt = group_small_entries(df, threshold)
    df_filt = normalize_time(df_filt)

    layers_cols = [s for s in df_filt.columns.values if('ward' in s and not 'avg' in s)]
    layers_cols = layers_cols + ['others']    
    df_filt.index = df_filt['file name']
    plt_data = pd.DataFrame(df_filt[layers_cols].transpose(),columns=df_filt['file name'], index=layers_cols)

    if(sort_data):
        plt_data= plt_data.sort_index()
        plt_data.sort_values(by=plt_data.columns.values[0], inplace=True, ascending=False)
    ser = pd.Series(plt_data.iloc[:,0].values, index=plt_data.index.values)
    ax = ser.plot.pie()
    ax.legend(loc='center left', bbox_to_anchor=(1.15, 0.5))
    ax.set_title('Time breakdown '+title_postfix)
    plt.savefig(os.path.join(res_path,'time_breakdown'+title_postfix+'.jpg'), format='jpg',bbox_inches='tight', dpi=900)
    return ax


def plot_all(f_wildcard, threshold=1.0, res_path=''):
    """filter and split the files from the provided wildcard
    and plot the thread/batch scaling figures"""
    flist = glb(f_wildcard)
    axis_l = dict()
    for arch in ['hsw', 'knl']:
        axis_l[arch] = dict()
        flist_arch = [f for f in flist if arch in f]
        df = get_df(flist_arch)
        
        grps = df.groupby(['batch size']).groups
        plt_grp = [(gk,gv) for gk, gv in grps.iteritems() if len(gv) >1]
        for batch_size, entries in plt_grp:
            if(len(df.loc[entries,'threads'].unique()) == 1): continue
            th_plts = df.loc[entries,:]
            axis_l[arch][batch_size] = plot_thread_scaling(th_plts, threshold, batch_size=batch_size, arch=arch, res_path=res_path)

        grps = df.groupby(['threads']).groups
        plt_grp = [(gk,gv) for gk, gv in grps.iteritems() if len(gv) >1]
        for th, entries in plt_grp:
            if(len(df.loc[entries,'batch size'].unique()) == 1): continue
            batch_plts = df.loc[entries,:]
            axis_l[arch][th] = plot_batch_scaling(batch_plts, threshold, threads=th, arch=arch, res_path=res_path)
    return axis_l

