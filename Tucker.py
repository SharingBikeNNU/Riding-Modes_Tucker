#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 15:42:24 2019

@author: caiboqin
"""

import pandas as pd
import numpy as np
import tensorly as tl
from tensorly.decomposition import non_negative_tucker
import geohash
import matplotlib.pyplot as plt
import geopandas

def generateShp(dataset, outPath="outshp/out.shp"):
    geodf=geopandas.read_file("bj/bj.shp")
    geodf=geodf.merge(pd.read_csv(dataset), how='left', left_on='Id', right_on='id')
    geodf.crs={'init' :'epsg:4326'}
    geodf.to_file(outPath)

def Tucker(datain, _x, _y, _z):
    # cell size and data range of coordinates
    xmax,xmin,ymax,ymin=117.073,115.668,40.465,39.465
    xsize,ysize=round((xmax-xmin)*200),round((ymax-ymin)*200)

    # import data and decode geohashed location
    df=pd.read_csv(datain)#,date_parser='starttime')
    loc=[geohash.decode_exactly(i) for i in df['geohashed_start_loc']]
    df['start_x'],df['start_y']=[i[1] for i in loc],[i[0] for i in loc]

    ## prepare a new dataset for group calculation
    df['datetime'] = pd.to_datetime(df['starttime'])
    dt=pd.DataFrame({'ts':df['datetime'],'id':df['orderid'], 'x':df['start_x'], 'y':df['start_y']})
    #dt=dt.query("ts >= '{}' and ts <= '{}'".format('2018-03-08 00:00:00', '2018-04-10 00:00:00'))
    dt=dt.set_index(['ts'])

    dt['date']=[(month-5)*31+day-10 for month,day in zip(dt.index.month,dt.index.day)]
    dt['hour']=dt.index.hour

    dt=dt[dt['x']<=xmax]
    dt=dt[dt['x']>=xmin]
    dt=dt[dt['y']<=ymax]
    dt=dt[dt['y']>=ymin]

    x=((np.array(dt['x'])-xmin)*ysize)//1
    y=((np.array(dt['y'])-ymin)*ysize)//1
    dt['loc']=y*xsize+x  # transform x,y to cell id

    # join data
    idx=[(x,y) for x in range(24) for y in [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14]]
    dt_join=pd.DataFrame({'idx':idx})
    dt_new = dt.groupby(['loc','hour','date']).count()['id']

    group_count=dt_new.unstack(['loc'])
    group_count['idx']=group_count.index
    group_count=pd.merge(dt_join,group_count,how='left',on='idx')
    group_count=group_count.set_index(['idx'])
    group_count=group_count.fillna(0)

    # generate tensor to decomposition
    X=np.array(group_count)
    #max_cell=np.max(b,axis=0)
    #m=np.tile(max_cell,(15*24,1))
    #b=b/m
    X=X.reshape(24,14,-1)#15327
    X=X.transpose([2,0,1])#张量转置

    # tensor decomposition
    X = tl.tensor(X)
    core, factors = non_negative_tucker(X, rank=[_x, _y, _z])#non_negative_

    # plot
    for i in range(factors[1].shape[1]):
        plt.plot(factors[1][:,i])
    plt.ylabel('Mode Value')
    plt.xlabel('Hour')
    plt.savefig('pic/M1.png', dpi=300)
    plt.show()

    for i in range(factors[2].shape[1]):
        plt.plot(factors[2][:,i])
    plt.ylabel('Mode Value')
    plt.xlabel('Day')
    plt.savefig('pic/M2.png', dpi=300)
    plt.show()

    space_out=pd.DataFrame(factors[0],index=group_count.columns)
    space_out['id']=space_out.index
    space_out.to_csv('space.csv',index=False)

    generateShp('space.csv')
    return factors[1], factors[2]

if __name__=="__main__":
    Tucker('indata/train.csv',6,3,2)