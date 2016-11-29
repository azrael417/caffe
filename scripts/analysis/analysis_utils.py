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
    m = layer_re.findall(l)
    if(m is not []): d['layers'] = m

    iters_re = re.compile('Testing for\s+(\d+)\s+iterations.')
    m = iters_re.search(l)
    if(m is not None): d['iterations'] = int(m.group(1))    

    th_re = re.compile('Number of OpenMP threads: (\d+)')
    m = th_re.search(l)
    if(m is not None): d['threads'] = int(m.group(1))    

    batch_size_re = re.compile('batch_size: (\d+)')
    batch_size_re2 = re.compile('shape .*?dim: (\d+)',re.DOTALL)
    m = batch_size_re.search(l)
    if(m is not None):
        d['batch size'] = int(m.group(1))
    else:
        m = batch_size_re2.search(l)
        if(m is not None):
            d['batch size'] = int(m.group(1))

    prof_re = re.compile('.+ Profiling Layer: (.+)')
    m = prof_re.search(l)
    if(m is not None): d['profiled layer'] = m.group(1)

    flops_re = re.compile('--->Total FLOPs = (\d+)')
    m = flops_re.search(l)
    if(m is not None): d['total flops'] = int(m.group(1))
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
        layers_data[lname+' '+direction+' avg time'] = float(val)/1e3  
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
    time_fields = [s for s in df.columns.values if('time' in s)]
    for f in time_fields:
        df_norm.loc[:, f] = df.loc[:, f]/df.loc[:, 'batch size']
    return df_norm

def normalize_time(df):
    """Normalize the time by the total time"""
    df_norm = df.copy()
    time_fields = [s for s in df.columns.values if(('ward avg time' in s) or ('time per iteration' in s) or ('others' in s))]
    for f in time_fields:
        df_norm.loc[:, f] = 100.0*df.loc[:, f]/df.loc[:, 'time per iteration']
    del df_norm['total time']
    return df_norm

def group_small_entries(df, threshold):
    """Sum insignificant entries in one column that fall
    below the 'threshold' percentile of the total time"""
    df_filt = df.copy()
    df_filt['others time'] = 0.0
    df_norm_time = normalize_time(df)
    time_fields = [s for s in df.columns.values if('ward avg time' in s)]
    for f in time_fields:
        if(not all(df_norm_time[f].apply(lambda x: x > threshold))):
            df_filt['others time'] = df_filt['others time'] + df_filt[f]
            del df_filt[f]
    return df_filt

def plot_batch_scaling(df, threshold, threads=1, arch='', res_path=''):
    """Plot batch scaling from a data frame table"""
    df.drop_duplicates('batch size',inplace=True)
    df_norm_batch = normalize_batches(df)
    df_filt = group_small_entries(df_norm_batch, threshold)

    layers_cols = [s for s in df_filt.columns.values if('ward avg time' in s)]
    layers_cols = layers_cols + ['others time']
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

def plot_thread_scaling(df, threshold, batch_size=1, arch='', res_path=''):
    """Plot thread scaling from a data frame table"""
    df_filt = group_small_entries(df, threshold)

    layers_cols = [s for s in df_filt.columns.values if('ward avg time' in s)]
    layers_cols = layers_cols + ['others time']
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

    df['time per iteration'] = df['time per iteration'] - df['data forward avg time']
    df['time per iteration'] = df['time per iteration'] - df['data backward avg time']
    df.drop('data forward avg time', axis=1, inplace=True)
    df.drop('data backward avg time', axis=1, inplace=True)
    
    
    df_filt = group_small_entries(df, threshold)
    layers_cols = [s for s in df_filt.columns.values if('ward avg time' in s)]
    layers_cols = layers_cols + ['others time']    
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
    ax.set_xlabel('Layer name')
    ax.set_title('Comparative layers time '+title_postfix)
    ax.set_ylim(ymin=0)
    ax.set_xlim(xmin=ax.get_xlim()[0]-0.25, xmax=ax.get_xlim()[1]+0.25)

    # add the total time in the legends
    exp_time = dict(df[['file name', 'time per iteration']].values)
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
    
    df['time per iteration'] = df['time per iteration'] - df['data forward avg time']
    df['time per iteration'] = df['time per iteration'] - df['data backward avg time']
    df.drop('data forward avg time', axis=1, inplace=True)
    df.drop('data backward avg time', axis=1, inplace=True)

    df_filt = group_small_entries(df, threshold)
    df_filt = normalize_time(df_filt)

    layers_cols = [s for s in df_filt.columns.values if('ward avg time' in s)]
    layers_cols = layers_cols + ['others time']    
    df_filt.index = df_filt['file name']
    plt_data = pd.DataFrame(df_filt[layers_cols].transpose(),columns=df_filt['file name'], index=layers_cols)
    plt_data = plt_data[plt_data.index != 'data forward avg time']
    plt_data = plt_data[plt_data.index != 'data backward avg time']

    layers_names = {s:s.replace(' avg time','') for s in plt_data.index.values}
    plt_data.rename(index=layers_names, inplace=True)
    
    if(sort_data):
        plt_data= plt_data.sort_index()
        plt_data.sort_values(by=plt_data.columns.values[0], inplace=True, ascending=False)
    ser = pd.Series(plt_data.iloc[:,0].values, index=plt_data.index.values)
    ax = ser.plot.pie()
    ax.legend(loc='center left', bbox_to_anchor=(1.15, 0.5))
    ax.set_title('Time breakdown '+title_postfix)
    plt.savefig(os.path.join(res_path,'time_breakdown'+title_postfix+'.jpg'), format='jpg',bbox_inches='tight', dpi=900)
    return ax

def plot_flops(sde_files_loc, time_file_loc, res_path='', title_postfix='', sort_data=False, threshold=1.0):
    """Plot the flop rate given a set of results from SDE for each layer and the time experiement"""
    # get the timing results of the test case
    df = get_df(glb(time_file_loc))

    layers = list(df['layers'].tolist()[0])
    expanded_layers = []
    for l in layers:
        expanded_layers.append(l+' forward')
        expanded_layers.append(l+' backward')
    plt_df = pd.DataFrame(index=expanded_layers, columns=['Time', 'GFlops'])

    layers_time_cols = {s.split(' avg')[0]:s for s in df.columns.values if('ward avg time' in s)}
    for k,v in layers_time_cols.iteritems():
        plt_df.loc[k,'Time'] = df[v].values[0]
        
    # get the flops from the SDE tests
    for f in glb(sde_files_loc):
        with open(f, 'r') as fp:
            entry = dict()
            txt = fp.read()
            get_meta(txt, entry)
            if ('profiled layer' in entry.keys()):
                plt_df.loc[entry['profiled layer'], 'GFlops'] = entry['total flops']/1e9
            else:
                del plt_df[entry['profiled layer'], 'GFlops']
                print "Could not parse: ", f

    plt_df = plt_df[plt_df.index != 'data forward']
    plt_df = plt_df[plt_df.index != 'data backward']
    plt_df['time percent'] = 100.0*plt_df.Time/plt_df.Time.sum()

    plt_filt = plt_df.loc[plt_df.loc[:,'time percent']>threshold,:]

    ax = plt_filt.GFlops.plot(kind='bar')
    ax.set_title('Layers flop rate'+' (filtered layers <'+str(threshold)+'% time)')
    ax.set_ylabel('GFLOP/s')
    ax.set_xlabel('Layer')
    plt.savefig(os.path.join(res_path,'flops_rate.jpg'), format='jpg',bbox_inches='tight', dpi=900)
    return ax

def plot_all(f_wildcard, threshold=1.0, res_path=''):
    """filter and split the files from the provided wildcard
    and plot the thread/batch scaling figures"""
    flist = glb(f_wildcard)
    axis_l = dict()
    for arch in ['hsw', 'knl']:
        axis_l[arch] = dict()
        flist_arch = [f for f in flist if arch in f]
        if(flist_arch==[]): continue
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

def plot_roofline_points(points_list,res_path='',title_prefix='', labels_markers={}):
    """Plot the KNL roofline for a set of data points
        points_list=[(Label, AI, GFlop/s),]"""
    from matplotlib import colors
    def get_gflops_per_core(cpu_speed,num_cores,vector_length,fma):
        fma_fac=2 if fma else 1
        return cpu_speed*num_cores*vector_length*fma_fac

    #Xeon Phi KNL, datasheet
    cpu_speed_ghz=1.4
    num_cores=68
    vector_length=32
    peak_bandwidth_gbs=90
    peak_bandwidth_hbm_gbs=425
    l1_cache_per_core_kb=32/2
    l2_cache_per_core_kb=512/2
    l3_cache_per_cpu_mb=0
    peak_gflops_ixeknl=get_gflops_per_core(cpu_speed_ghz,num_cores,vector_length,True)

    xopt = np.arange(0.1,250,0.1)
    yopt = [np.min([val*peak_bandwidth_hbm_gbs,peak_gflops_ixeknl]) for val in xopt]
    ynohbm = [np.min([val*peak_bandwidth_gbs,peak_gflops_ixeknl]) for val in xopt]
    #ytlponly = [np.min([val*peak_bandwidth_hbm_gbs,get_gflops_per_core(cpu_speed_ghz,num_cores,1,False)]) for val in xopt]
    #ytlpilp = [np.min([val*peak_bandwidth_hbm_gbs,get_gflops_per_core(cpu_speed_ghz,num_cores,2,False)]) for val in xopt]
    #ynofma = [np.min([val*peak_bandwidth_hbm_gbs,get_gflops_per_core(cpu_speed_ghz,num_cores,vector_length,False)]) for val in xopt]
    yfmanosimd = [np.min([val*peak_bandwidth_hbm_gbs,get_gflops_per_core(cpu_speed_ghz,num_cores,2,True)]) for val in xopt]
    yopt75p = [np.min([val*peak_bandwidth_hbm_gbs,peak_gflops_ixeknl*0.75]) for val in xopt]

    #removed malicious dpi=1024 argument
    plt.figure(num=None, figsize=(20, 27), facecolor='w', edgecolor='k')
    ax=plt.subplot(2,1,1)
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(50)

    area = np.pi * 17**2

    #ranges:
    minx= float('inf')
    miny= float('inf')
    maxx= float('-inf')
    maxy= float('-inf')
    for label, x, y in points_list:
        minx = min(minx, x)
        miny = min(miny, y)
        maxx = max(maxx, x)
        maxy = max(maxy, y)
    plt.xlim(10**(-1),max(xopt))
    plt.ylim(min(10**(1),miny),10**4)

    ax.tick_params(axis='both', which='major', size=15)
    ax.tick_params(axis='both', which='minor', size=15)
    ax.tick_params(axis='both', which='major', labelsize=50)

    #lines
    line0,=plt.plot(xopt, yopt75p, '-', linewidth=3,color='k',label='75% of optimal SP')
    line1,=plt.plot(xopt, yopt, '-', linewidth=3,color='green',label='optimal SP (HBM)')
    line2,=plt.plot(xopt, ynohbm, '-.', linewidth=5,color='k',label='optimal SP (DDR)')
    #line3,=plt.plot(xopt, ynofma, '-', linewidth=3,color='orange',label='no FMA')
    line3m,=plt.plot(xopt, yfmanosimd, '--', linewidth=3,color='blue',label='FMA no SIMD')
    #line4,=plt.plot(xopt, ytlpilp, '-', linewidth=3,color='red',label='TLP with ILP')
    #line4m,=plt.plot(xopt, ytlponly, '--', linewidth=3,color='red',label='TLP only')

    banned_colors = ['yellow','gold','pink','mistyrose','navajowhite','palegreen','greenyellow',
                     'seashell','papayawhip','blanchedalmond','chartreuse','white', 'ivory', 'lawngreen',
                    'lime', 'burlywood', 'mediumspringgreen', 'peachpuff', 'springgreen', 'aquamarine']
    
    c_list = [(k,v) for k,v in colors.cnames.iteritems() if all(k!=b for b in banned_colors)]
    for ind, (label, x, y) in enumerate(points_list):
        marker='v'
        for k,v in labels_markers.iteritems():
            if(k in label):
                marker=v
                break
#        label = label+' '+c_list[ind][0]  # print colors to exclude the undesired
        ax.scatter(x,y,marker=marker, linewidth=5,s=area,alpha=0.5, label=label, color=c_list[ind][1])

    for tick in ax.xaxis.get_major_ticks():
        tick.set_pad(20)
    for tick in ax.yaxis.get_major_ticks():
        tick.set_pad(15)

    plt.title(title_prefix, size=50, y=1.01)
    ax.set_xscale('log',basex=10)
    ax.set_yscale('log',basey=10)
    ax.set_xlabel('Arithmetic Intensity Flops/Byte', size=50)
    ax.set_ylabel('GFlops/s', size=50)
    
    ax.legend(scatterpoints=1, prop={'size':20}, loc='center left', bbox_to_anchor=(1, 0.5),title='layers end w/ time %')
    ax.get_legend().get_title().set_fontsize('20')

    plt.savefig(os.path.join(res_path,'roofline.jpg'), format='jpg',bbox_inches='tight', dpi=900)
    return ax



def generate_roofline_sde(time_file_loc, sde_files_loc, likwid_file_loc, res_path=''
                      ,sort_data=False, title_prefix='', threshold=1.):
    df = get_df(glb(time_file_loc))
    """Generate roofline figure from SDE, LIKWID, and timing measurements"""

    layers = list(df['layers'].tolist()[0])
    expanded_layers = []
    for l in layers:
        expanded_layers.append(l+' forward')
        expanded_layers.append(l+' backward')
    plt_df = pd.DataFrame(index=expanded_layers, columns=['Time', 'GFlops', 'GB memory volume'])

    layers_time_cols = {s.split(' avg')[0]:s for s in df.columns.values if('ward avg time' in s)}
    for k,v in layers_time_cols.iteritems():
        plt_df.loc[k,'Time'] = df[v].values[0]

    # Parse LIKWID memory test
    f = glb(likwid_file_loc)[0]
    with open(f, 'r') as fp:
        likwid_entry = dict()
        txt = fp.read()
        get_meta(txt, likwid_entry)

        # Get the measurement type
        group_re = re.compile('TABLE,Region.*Metric,(\w*)')
        m = group_re.search(txt)
        if(m is not None): df['HW perf group'] = m.group(1)

        for layer in layers:
            for di in ['forward', 'backward']:
                exp_layer = layer+' '+di
                mem_vol_re = re.compile(layer+'_'+di+'.*?Memory data volume \[MBytes\],(\d+(.\d+)?)',re.DOTALL)
                m = mem_vol_re.search(txt)
                if(m is not None):
                    plt_df.loc[exp_layer, 'GB memory volume'] = float(m.group(1))/likwid_entry['iterations']/1e3

    # get the flops from the SDE tests
    for f in glb(sde_files_loc):
        with open(f, 'r') as fp:
            entry = dict()
            txt = fp.read()
            get_meta(txt, entry)
            if ('profiled layer' in entry.keys()):
                plt_df.loc[entry['profiled layer'], 'GFlops'] = entry['total flops']/1e9
            else:
                del plt_df[entry['profiled layer'], 'GFlops']
                print "Could not parse: ", f


    # Filter the data
    plt_df = plt_df[plt_df.GFlops != 0.0]
    plt_df = plt_df[plt_df.index != 'data forward']
    plt_df = plt_df[plt_df.index != 'data backward']

    # Add derived metrics of interest
    plt_df['AI'] = plt_df['GFlops']/plt_df['GB memory volume']
    plt_df['GFlop/s'] = plt_df.GFlops/plt_df.Time
    plt_df['time percent'] = 100.0*plt_df.Time/plt_df.Time.sum()

    plt_filt = plt_df.loc[plt_df.loc[:,'time percent']>threshold,:]

    if(sort_data): plt_filt = plt_filt.sort_index()
    plt_filt.index = plt_filt.index.str.cat(plt_filt['time percent'].map('{:02.0f}'.format), sep=' ')
    data_points = zip(plt_filt.index.tolist(),plt_filt.AI.tolist(),plt_filt['GFlop/s'].tolist())
    #for d in data_points: print d
    ax = plot_roofline_points(data_points, res_path=res_path, title_prefix=title_prefix+' (filtered layers <'+str(threshold)+'%)'
                       , labels_markers={'conv':'.', 'fc':'^', 'norm=':'x', 'pool':'*', 'loss':'v'})
    return ax, plt_df


def generate_roofline_likwid(time_file_loc, likwid_file_loc, res_path=''
                      ,sort_data=False, title_prefix='', threshold=1.):
    df = get_df(glb(time_file_loc))
    """Generate roofline figure from flops/mem events using LIKWID, and timing measurements"""

    layers = list(df['layers'].tolist()[0])
    expanded_layers = []
    for l in layers:
        expanded_layers.append(l+' forward')
        expanded_layers.append(l+' backward')
    plt_df = pd.DataFrame(index=expanded_layers, columns=['Time', 'GFlops', 'GB memory volume'])

    layers_time_cols = {s.split(' avg')[0]:s for s in df.columns.values if('ward avg time' in s)}
    for k,v in layers_time_cols.iteritems():
        plt_df.loc[k,'Time'] = df[v].values[0]

    # Parse LIKWID memory and Flops test
    f = glb(likwid_file_loc)[0]
    with open(f, 'r') as fp:
        likwid_entry = dict()
        txt = fp.read()
        get_meta(txt, likwid_entry)

        # Get the measurement type
        group_re = re.compile('TABLE,Region.*Metric,(\w*)')
        m = group_re.search(txt)
        if(m is not None): df['HW perf group'] = m.group(1)

        for layer in layers:
            for di in ['forward', 'backward']:
                exp_layer = layer+' '+di

                # Likwid formatted output
                mem_vol_re = re.compile(layer+'_'+di+'.*?Memory data volume \[MBytes\] STAT\s+\|\s+([-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+)?)',re.DOTALL)
                m = mem_vol_re.search(txt)
                if(m is not None):
                    plt_df.loc[exp_layer, 'GB memory volume'] = float(m.group(1))/likwid_entry['iterations']/1e3
                else:
                    # Likwid CSV output
                    mem_vol_re = re.compile(layer+'_'+di+'.*?Memory data volume \[MBytes\],(\d+(.\d+)?)',re.DOTALL)
                    m = mem_vol_re.search(txt)
                    if(m is not None):
                        plt_df.loc[exp_layer, 'GB memory volume'] = float(m.group(1))/likwid_entry['iterations']/1e3
                    else:
                        print "memory volume could not find ", exp_layer

                # Likwid formatted output
                flops_re = re.compile(layer+'_'+di+'.*?MFLOP \(SP AVX512 FMA assumed\) STAT\s+\|\s+([-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+)?)',re.DOTALL)
                m = flops_re.search(txt)
                if(m is not None):
                    plt_df.loc[exp_layer, 'GFlops'] = float(m.group(1))/likwid_entry['iterations']/1e3
                else:
                    # Likwid CSV output
                    flops_re = re.compile(layer+'_'+di+'.*?MFLOP \(SP AVX512 FMA assumed\) STAT,(\d+(.\d+)?)',re.DOTALL)
                    m = flops_re.search(txt)
                    if(m is not None):
                        plt_df.loc[exp_layer, 'GFlops'] = float(m.group(1))/likwid_entry['iterations']/1e3
                    else:
                        print "GFlops could not find ", exp_layer

    # Filter the data
    plt_df = plt_df[plt_df.GFlops != 0.0]
    plt_df = plt_df[plt_df.index != 'data forward']
    plt_df = plt_df[plt_df.index != 'data backward']

    # Add derived metrics of interest
    plt_df['AI'] = plt_df['GFlops']/plt_df['GB memory volume']
    plt_df['GFlop/s'] = plt_df.GFlops/plt_df.Time
    plt_df['time percent'] = 100.0*plt_df.Time/plt_df.Time.sum()

    plt_filt = plt_df.loc[plt_df.loc[:,'time percent']>threshold,:]

    if(sort_data): plt_filt = plt_filt.sort_index()
    plt_filt.index = plt_filt.index.str.cat(plt_filt['time percent'].map('{:02.0f}'.format), sep=' ')
    data_points = zip(plt_filt.index.tolist(),plt_filt.AI.tolist(),plt_filt['GFlop/s'].tolist())
    #for d in data_points: print d
    ax = plot_roofline_points(data_points, res_path=res_path, title_prefix=title_prefix+' (filtered layers <'+str(threshold)+'%)'
                       , labels_markers={'conv':'.', 'fc':'^', 'norm=':'x', 'pool':'*', 'loss':'v'})
    return ax, plt_df