# SAPSPAR
SAPSPAR (**Stochastic All-Pair Shortest Path Routing**) is a parall computing framework and software implementation for computing all-pair shortest paths (APSP) within stochastic road networks accelerated using Graphics Processing Units (GPUs). The computing of APSP within stochastic road networks is extremely computationally considerable. SAPSPAR here provides a solution that makes this computing feasible. 


# Data and source code for stochastic all-pair shortest path routing (SAPSPAR) 

## under main folder

**simulation.py**:			Python source code for the simulation of link travel time (road segment level) of a road network

**path_floyd.cpp**: 		C++ source code for CPU-based all-pair shortest path calculation

**path_resample_sub.cpp**: C++ source code for CPU-based path travel time estimation via Monte Carlo-based resampling, including a row-wise domain decomposition

**path_floyd.cu**: 		CUDA source code for GPU-based parallel all-pair shortest path calculation

**path_resample.cu**: 		CUDA source code for GPU-based parallel travel time resampling using Monte Carlo approach, includidng two steps: path extraction and Monte Carlo-based resampling of travel time 

## under ./data folder

**params.txt**: 	parameter file for all-pair shortest path calculation

**wuhanOD.txt**: 	example input OD matrix (using Wuhan city of China here) of distance or travel time for shortest path calculation; about 1.5G, within data.zip (provide a link to the zip file; too big)

**next.txt**: 		example input OD matrix of predecessors for travel time resampling using Monte Carlo approach; about 2.5G, within data.zip (provide a link to the zip file; too big)

**data.zip**: 		zip file includidng wuhanOD.txt and next.txt, one of the OD matrices of travel time and predecessors (provide a link)

**cdf.txt**: 		probability look up table (cumulative probability)
**./Wuhan**: 		shapefiles and edge files of road network of Wuhan

