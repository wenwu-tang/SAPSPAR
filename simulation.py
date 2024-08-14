import pandas as pd
import geopandas as gpd
import sys
import numpy as np
from scipy.stats import norm
from scipy.stats import uniform
import os

'''
This script is for the simulation of link travel time of a road network. Wuhan road network was used here as an example

Road network data (Wuhan here) were downloaded from https://figshare.com/articles/dataset/Urban_Road_Network_Data/2061897 
It output three files, an ASCII OD matrix, a lookup table (OD matrix row/column ID -> Node ID), and a csv file for the edge list with travel time

It first join type and maxspeed from ESRI Shapfile to the edgelist table. Then join a speed lookup table(.csv) to 
Wuhan edge list (.csv, joined with road type) so that each edge with maxspeed as 0 will be assigned a value as 
per the lookup table. The output is an edgelist with speed and a corresponding OD matrix of travel time.

To run it:

python <directory to this file> <city name (capitalize first letter)> <directory to job folder> <distribution> <rate>

e.g., python ./pre_processing.py  Wuhan /PATH/job normal 0.1

The outputs will be saved at current working directory (e.g., pwd in Linux)
'''

def generate_random_value(distribution='normal', time=20.5, rate_to_max=0.1, rv_p=0.1):
    if distribution == 'normal':
        loc = time + 0.5 * rate_to_max * time  # mean time between min and max time along x axis
        scale = (loc - time) / 2  # set min to mean as two sigma -> so 5% chance it will be lower than min time (1-3 sigma probability: 68, 95, 99.7)
        
        rv = norm.ppf(rv_p, loc, scale)
        rv_pdf = norm.pdf(rv_p, loc, scale)
    elif distribution == 'uniform':
        loc = time  # start point (min time) along x axis
        scale = time * rate_to_max  # range along x axis

        rv = uniform.ppf(rv_p, loc, scale)
        rv_pdf = uniform.pdf(rv_p, loc, scale)
    return rv


# read input
city=sys.argv[1]
base_dir=sys.argv[2]
distribution=sys.argv[3]
rate=float(sys.argv[4])

print(f'City: {city}')
print(f'distribution: {distribution}')
print(f'User configured rate: {rate}')

lookup_dir=os.path.join(base_dir,'data','speed_lookup_table.csv')
data_dir = os.path.join(base_dir,'data/Wuhan')
edgelist_dir=os.path.join(data_dir,f'{city}_Edgelist.csv')
shapefile_dir=os.path.join(data_dir,f'{city}_Links.shp')


# join shapefile type and maxspeed to table
shpdf = gpd.read_file(shapefile_dir)
csvdf = pd.read_csv(edgelist_dir)
mergeddf = shpdf.merge(csvdf, left_on="OBJECTID", right_on="EDGE")
edge_speed = mergeddf[["XCoord", "YCoord", "START_NODE", "END_NODE", "EDGE", "LENGTH", "type", "maxspeed"]]


lookup=pd.read_csv(lookup_dir)
edge=edge_speed
#get edge without speed, join with lookup table, assign value to maxspeed, and delete the new column speed
indices_to_join=edge[edge['maxspeed']==0].index
edge_to_join=edge.loc[indices_to_join].join(lookup.set_index('type'),on='type')
edge_to_join.loc[:,'maxspeed']=edge_to_join['speed']
edge_to_join=edge_to_join.drop(columns=['speed'])

#get edges with speed
edge_w_speed=edge.drop(indices_to_join)

#append all back together
edge=pd.concat([edge_w_speed,edge_to_join]).sort_index()
edge['time_sec']=edge['LENGTH']/(edge['maxspeed']*1000/3600)
edge['perturbed_time']=edge.apply(lambda row: generate_random_value(distribution=distribution,time=row['time_sec'],rate_to_max=rate,rv_p=np.random.rand()),axis=1)
# print(edge['perturbed_time'])
#save to csv
edge_out_dir=os.path.join(base_dir,'output',f'{city}_edgelist_out.csv')
edge.to_csv(edge_out_dir,index=False)
print(f'edgelist saved at {edge_out_dir}')

#convert to OD matrix
node_id=pd.unique(edge['START_NODE'])
node_id=np.sort(node_id)
# print(node_id)
# print(np.where(node_id==1)[0])
n=len(node_id)
print(f'# nodes: {n}')

print(f'Saving OD matrix. This can take a while...')
OD_outdir=os.path.join(base_dir,'output','ODmatrix.txt')
# np.savetxt(OD_outdir,OD)
print(f'OD matrix saved at {OD_outdir}')

# create node_ID
df=pd.DataFrame()
df['node_ID']=node_id
df.reset_index(inplace=True)
df=df.rename(columns={'index':'OD_ID'})
node_id_dir=os.path.join(base_dir,'output','node_ID.csv')
df.to_csv(node_id_dir,index=False)
print(f'nodeID lookup table saved at {node_id_dir}')
