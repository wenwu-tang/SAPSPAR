###################################### C/C++ Source Code for All Pair Shortest Path Routing using Floyd-Warshall Algorithm #######################

#include <iostream>
#include <fstream>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define STAR 99999999
#define CUDA

#ifdef CUDA
#include <cuda_runtime.h>
#include <helper_cuda.h>
#endif

using namespace std;

float* m,*m1,*m2;
int size=0;
string dataName;
int isPathExtract=1;
int *aPath;// store IDs of nodes
int stopFlag=1;//0: stop
int gpuID=0; //ID of the GPU used for computation

void readParams(string fn);
void printParams();
void readData(string fn);
void writeData(string fn,float*s);
void extractPathAll();
void extractPath(int from,int to);
void extractNode(int from,int to,int cur);
void printPath(int*pa);

//CPU-based Floyd_Warshall algorithm
int *next_m; //store next node for shorest path
void floyd_warshall(float*d,int*pa);
void extractPathAll_floyd(int*pa);
void extractPath_floyd(int*pa, int from,int to);
void printNext(int*pa,string fn);
void printDist(float*pa,string fn);

#ifdef CUDA
float *m_gpu,*m1_gpu,*m2_gpu;
int grid_size;
int block_size=128;
int size_perthread=1;
int* stopFlag_cpu;
int* stopFlag_gpu;

// CUDA-based Floyd Warshall algorithm
int *next_m_gpu;
//kernel of initialization function for Floyd Warshall algorithm
__global__ void floyd_warshall_initialize_gpu(float*d,int*pa,int size,int size_perthread);
//kernel of Floyd Warshall algorithm for all-pair shortest path routing 
__global__ void floyd_warshall_gpu(float*d,int*pa,int size, int size_perthread,int k);

unsigned int timer=0;
float computime;
void startGPU();
void endGPU();
#endif

int main(int argc, char* argv[]);

int main(int argc, char* argv[])
{
	int i;
	time_t start,end,end1; double diff;
	if(argc>1){
		//set gpu ID instead of using default.
		gpuID=atoi(argv[1]);
	}

	cout<<"read parameters..."<<endl;
	readParams("./params.txt");
	printParams();
	
	m=new float[size*size];
	next_m=new int[size*size];

	cout<<"read data..."<<endl;
	readData(dataName);

	if(isPathExtract){
		aPath=new int[size];
		memset(aPath,-1,sizeof(int)*size); 
	}
#ifdef CUDA
	cout<<"init GPU..."<<endl;
	cout<<"use GPU "<<gpuID<<endl;
	cudaSetDevice(gpuID);
	startGPU();

	cudaError_t err = cudaSuccess;
	cudaEvent_t start1,stop1;
	cudaEventCreate(&start1);
	cudaEventCreate(&stop1);
	cudaEventRecord(start1);
	
#endif
	string fn;
	time(&start);

#ifndef CUDA
	floyd_warshall(m,next_m);
#endif
#ifdef 	CUDA
	//initialize for CUDA-based Floyd Warshall algorithm on GPU
	floyd_warshall_initialize_gpu<<<grid_size,block_size>>>(m_gpu,next_m_gpu,size,size_perthread);
	//run FW algorithm on GPU
	for(int k=0;k<size;k++){
		floyd_warshall_gpu<<<grid_size, block_size>>>(m_gpu,next_m_gpu,size,size_perthread,k);		
	}
#endif
	time(&end1);
	diff=difftime(end1,start);

#ifdef CUDA
	//export OD matrix of distance/travel time //can be commented out as needed
	err = cudaMemcpy(m, m_gpu, size*size*sizeof(float), cudaMemcpyDeviceToHost);
        printDist(m,"./dist_floyd.txt");//export od matrix of distance or time.
	
	//export OD matrix of predecessors //can be commented out as needed
	err = cudaMemcpy(next_m, next_m_gpu, size*size*sizeof(int), cudaMemcpyDeviceToHost);
	printNext(next_m,"./next_floyd.txt"); //export od matrix of predescessor

#endif
	cout<<"Done..."<<endl;
	//timing
	time(&end);
	diff=difftime(end,start);
	cout<<"Total computing Time: "<<diff<<endl;

#ifdef CUDA
	cout<<"end GPU..."<<endl;
	cudaEventRecord(stop1);
	cudaEventSynchronize(stop1);
	cudaEventElapsedTime(&computime,start1,stop1);
	cudaEventDestroy(start1);
	cudaEventDestroy(stop1);
	cout<<"GPU time: "<<computime<<";"<<start1<<"->"<<stop1<<endl;
	printf("GPU TIME: %f\n",computime);
	endGPU();

#endif
	if(m!=NULL) delete [] m;
	if(isPathExtract){
		if(aPath!=NULL) delete [] aPath;
		if(next_m!=NULL) delete [] next_m;
	}
	return 0;

}
void readParams(string fn){
	char str[200];
	int val;
	ifstream f;
	f.open(fn);
	f>>str>>str;
	dataName=str;
	f>>str>>val;
	size=val;
	f>>str>>val;
	niter=val;
	f>>str>>val;
	block_size=val;	
	f.close();

}
void printParams(){
	cout<<"File Name:\t"<<dataName<<endl;
	cout<<"#Nodes:\t"<<size<<endl;
	cout<<"#Max_iterations:\t"<<niter<<endl;
	cout<<"Block Size:\t"<<block_size<<endl;
}
void readData(string fn){
	ifstream f;
	f.open(fn);
	int i,j,n;
	float val;
	n=size;
	for(i=0;i<n;i++){
		for(j=0;j<n;j++){
			f>>val;
			if(val==-1) val=STAR;
			m[i*size+j]=val;	
		}
	}
	f.close();
}
void writeData(string fn,float*s){
	ofstream f;
	f.open(fn);
	int i,j;
	int n;
	n=size;

	for(i=0;i<n;i++){
		for(j=0;j<n;j++){
			f<<s[i*size+j]<<"\t";
		}
		f<<endl;
	}
	f.close();
}
void extractPathAll(){
	int i,j;

	for(i=0;i<size;i++){
                for(j=0;j<size;j++){
                        extractPath(i,j);
                        printPath(aPath);
                        memset(aPath,-1,sizeof(int)*niter+1);
                }
        }

}
void printPathAll(){
	int i,j;
	int* pa;
	for(i=0;i<size;i++){
                for(j=0;j<size;j++){
                        cout<<i+1<<"\t"<<j+1<<"\t";
                        //extractPath(i,j);
						pa=&pathAll[(i*size+j)*(niter+1)];
                        printPath(pa);
                        //memset(aPath,-1,sizeof(int)*niter+1);
                }
        }
}

//extract a specific path from the cube for a OD pair, stored into aPath
void extractPath(int from,int to){
	extractNode(from,to,niter);	
}
void printPath(int*pa){
	int i;
	int *pa1=pa;
	int n=niter_stop;
	for(i=0;i<n+1;i++){
	//	cout<<pa1[i]+1<<"\t";
		if(pa1[i]!=-1){
			cout<<pa1[i]+1;
			if(i!=n)
				cout<<"->";
		}
	
	}
	cout<<endl;
}

#ifdef CUDA
void startGPU(){
	// Error code to check return values for CUDA calls
    	cudaError_t err = cudaSuccess;
	err = cudaMalloc((void **)&m_gpu, size*size*sizeof(float));
    	if (err != cudaSuccess)
    	{
        	fprintf(stderr, "Failed to allocate device m_gpu (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
    	}	
	err = cudaMalloc((void **)&next_m_gpu, size*size*sizeof(int));
        if (err != cudaSuccess)
        {
                fprintf(stderr, "Failed to allocate device next_m_gpu (error code %s)!\n", cudaGetErrorString(err));
                exit(EXIT_FAILURE);
        }
	
	//copy memory from host to device
	cout<<"copy data from host memory to device memory..."<<endl;

	err = cudaMemcpy(m_gpu, m, size*size*sizeof(float), cudaMemcpyHostToDevice);
    	if (err != cudaSuccess)
    	{
        	fprintf(stderr, "Failed to copy m from host to device (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
    	}
	err = cudaMemcpy(next_m_gpu, next_m, size*size*sizeof(float), cudaMemcpyHostToDevice);

        if (err != cudaSuccess)
        {
                fprintf(stderr, "Failed to copy next_m from host to device (error code %s)!\n", cudaGetErrorString(err));
                exit(EXIT_FAILURE);
        }

	//calculate grid size
	grid_size=size*size/(block_size*size_perthread)+(size*size%(block_size*size_perthread)==0?0:1);;
}
void endGPU(){
	// Free device global memory
	cudaError_t err = cudaSuccess;
	err = cudaFree(m_gpu);

    	if (err != cudaSuccess)
    	{
        	fprintf(stderr, "Failed to free device m_gpu (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
    	}
}
#endif

//CPU-based Floyd Warshall algorithm  
void floyd_warshall(float*d,int*pa){
        int k,i,j;
        int n=size;
        for(i=0;i<n;i++){
                for(j=0;j<n;j++){
                        if(i==j)
                                pa[i*n+j]=-1;
                        else if(abs((d[i*n+j]-STAR))>0.00000001)                      
                                pa[i*n+j]=j;
                        else
                                pa[i*n+j]=-2;
                }
        }

        for(k=0;k<n;k++){
                for(i=0;i<n;i++){
                        for(j=0;j<n;j++){
                                if(d[i*n+j]>d[i*n+k]+d[k*n+j]){
                                        d[i*n+j]=d[i*n+k]+d[k*n+j];
                                        pa[i*n+j]=pa[i*n+k];
                                }
                        }
                }
        }
}


void extractPathAll_floyd(int*pa){
        int i,j;
        int n=size;

        for(i=0;i<n;i++){
                for(j=0;j<n;j++){
						//cout<<i<<"\t"<<j<<endl;
                        extractPath_floyd(pa,i,j);
                }
        }
}

void extractPath_floyd(int*pa, int from,int to){
        int n=size;
        int counter=1;
        int cur=from;
        if(pa[from*size+to]==-2)
                return;
        aPath[0]=from;
        cur=from;
        while(cur!=to){
                cur=pa[cur*n+to];
                aPath[counter]=cur;
                counter++;
        }

        int i;
        for(i=0;i<counter;i++){
                cout<<aPath[i]+1<<"\t";
        }
        cout<<endl;
        return;
}

void printNext(int*pa,string fn){
        ofstream f;
        f.open(fn);
        int i,j;
        int n=size;
        for(i=0;i<n;i++){
                for(j=0;j<n;j++){
                        f<<pa[i*n+j]<<"\t";
                }
                f<<endl;
        }
}
void printDist(float*pa,string fn){
        ofstream f;
        f.open(fn);
        int i,j;
        int n=size;
        for(i=0;i<n;i++){
                for(j=0;j<n;j++){
                        f<<pa[i*n+j]<<"\t";
                }
                f<<endl;
        }
}

#ifdef CUDA
__global__ void floyd_warshall_initialize_gpu(float*d,int*pa,int size,int size_perthread){
	int idx=blockIdx.x*blockDim.x+threadIdx.x;
	if(idx>=size*size) return;
	int n=size;
	int ii,i,j,id;
	for(ii=0;ii<size_perthread;ii++){
		id=idx*size_perthread+ii;
		i=id/size;
		j=id%size;
		if(i==j)
			pa[i*n+j]=-1;
		else if(abs((d[i*n+j]-STAR))>0.00000001)
			pa[i*n+j]=j;
		else
			pa[i*n+j]=-2;
	}
}
__global__ void floyd_warshall_gpu(float*d,int*pa,int size, int size_perthread,int k){
	int idx=blockIdx.x*blockDim.x+threadIdx.x;
	if(idx>=size*size) return;
	int n=size;
	int ii,i,j,id;
	for(ii=0;ii<size_perthread;ii++){
		id=idx*size_perthread+ii;
		i=id/size;
		j=id%size;
		if(d[i*n+j]>d[i*n+k]+d[k*n+j]){
                	d[i*n+j]=d[i*n+k]+d[k*n+j];
                        pa[i*n+j]=pa[i*n+k];
                }
		
	}
}

#endif
