################################# CUDA Source Code for GPU-based Path Travel Time Estimation using Monte Carlo Approach ##########################

#include <iostream>
#include <fstream>
#include <string.h>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define STAR 100000000
#define CUDA

#ifdef CUDA
#include <cuda_runtime.h>
#include <helper_cuda.h>
#define Max_Block_Size 1024
#define Max_Len_Path 400  //maximal path length
#define RANDOM_A 1664525
#define RANDOM_B 1013904223
#endif

using namespace std;

float* m,*m1,*m2;
int size=0;
string dataName;
int isPathExtract=1;
int* next_m;

int sub_start,sub_end;
string probaFile;
float*probaTable;
int sizeTable=100000;
void readProbaFile(string fn);
int niter=100;  // maximum #iterations 
int niter_stop;// #iters when iterations stop
int *aPath;// store IDs of nodes
int stopFlag=1;//0: stop
float* transpose(float*source,float* des, int nr,int nc);
void readParams(string fn);
void printParams();
float* readMatrix(string fn,float*s);
void readData(string fn);
void writeData(string fn,float*s);
void initialize_next(float *d,int*pa);
void printPath(int*pa);
void printNext(int*pa,char const* fn);
void extractPathAll_floyd(int*pa);
void extractPath_floyd(int*pa, int from,int to);
int* readNext(string fn,int*next);//load matrix of predecessors
float* readBinData(string fn,float*s,int n);
void writeBinData(string fn,float*s,int n);

#ifdef CUDA
float *m_gpu,*m1_gpu,*m2_gpu;
int* pathAll;//(size*nrows_sub*n); //n_rows_sub: every n_partition rows  for large data
int*pathAll_gpu;
int nrows_sub=200;// #rows on OD matrix for path extraction; predefined 
int max_length_path=400;//maximum path length in terms of #nodes // predefine

int grid_size;
int block_size=128;  //blockDim.x
int block_size2=8;   //blockDim.y
int size_perthread=1; 
int* stopFlag_cpu;
int* stopFlag_gpu;
int* nextm_gpu;//predecessor matrix

int* pathlen;
int* pathlen_gpu;

int*seed_random;
float* meanM;
float* stdM;
int*d_seed_random;
float*d_probaTable;
float*d_meanM;
float*d_stdM;

unsigned int timer=0;
float computime;
float computime1;// for path extraction
void startGPU();
void endGPU();
//kernel of path travel time estimation via Monte Carlo-based resampling
__global__ void extractPathAll_path_gpu(int*N,int*len,float*dist,int size,int begin,int nrows,int maxLen,int size_perthread,float*probaTable,int sizeTable,int*rand, float rate_perturb,float*meanM,float*stdM);
__device__ void extractPath_path_gpu(int*N,int*len,float*dist,int size,int from, int to,int begin,int nrows,int maxLen,float*probaTable,int sizeTable,int*rand, float rate_perturb,float*meanM,float*stdM);

#endif

int main(int argc, char* argv[]);

int main(int argc, char* argv[])
{
	int i;
	time_t start,end,end1; double diff;
	int device_id=0;
	float rate=0.1; //rate of perurbation; passed from arguments
	if(argc>=3){
		sub_start=atoi(argv[1]);
		sub_end=atoi(argv[2]);
		device_id=atoi(argv[3]);
		rate=atof(argv[4]);
	}

	cout<<"read parameters..."<<endl;
	readParams("./params.txt");
	printParams();

	cout<<"read data..."<<endl;
	m=new float[size*size];
	readData(dataName);
	
	if(isPathExtract){
		next_m=new int[size*size];
                next_m=readNext("./next.txt",next_m);
		aPath=new int[max_length_path];
		meanM=new float[size*size];
		stdM=new float[size*size];
		probaTable=new float[100000];
		readProbaFile("./cdf.txt");
	}
#ifdef CUDA
	cout<<"init GPU..."<<endl;
	cudaSetDevice(device_id);
	startGPU();
	cudaError_t err = cudaSuccess;
	cudaEvent_t start1,stop1;
	cudaEventCreate(&start1);
	cudaEventCreate(&stop1);
	cudaEvent_t start2;
	cudaEventCreate(&start2);
	
	dim3 grid(grid_size,1,1), block(block_size,block_size2,1);	
#endif
	string fn;
	niter=size;
	time(&start);
	cout<<"Resample path travel time..."<<endl;

#ifdef CUDA
	unsigned int totalSize1;
	int begin;
	begin=0;
	cout<<"grid size: "<<grid_size<<endl;
	cout<<"block size: "<<block_size<<endl;

	cudaEventRecord(start1);
	computime=0;
	float prevTime=0;
	
	cout<<"start-end: "<<sub_start<<";"<<sub_end<<endl;
	for(i=sub_start;i<sub_end;i++){
		cout<<"part: "<<i+1<<"\t";
		prevTime=computime;
		extractPathAll_path_gpu<<<grid_size,block_size>>>(nextm_gpu,pathlen_gpu,m_gpu,size,begin,nrows_sub,max_length_path,size_perthread,d_probaTable,sizeTable,d_seed_random,rate,d_meanM,d_stdM);
		begin=begin+nrows_sub;

		cudaEventRecord(stop1);
        cudaEventSynchronize(stop1);
        cudaEventElapsedTime(&computime,start1,stop1);
		cout<<"compute time: "<<computime<<"\t"<<computime-prevTime<<endl;
	}

#endif
	cout<<"Done path travel time resampling..."<<endl;
	//timing
	time(&end);
	diff=difftime(end,start);
	cout<<"Total computing Time: "<<diff<<endl;

#ifdef CUDA
	cudaEventRecord(stop1);
	cudaEventSynchronize(stop1);
	cudaEventElapsedTime(&computime,start1,stop1);
	//path extraction time
	cudaEventElapsedTime(&computime1,start2,stop1);

	cudaEventDestroy(start1);
	cudaEventDestroy(stop1);
	cudaEventDestroy(start2);
	cout<<"GPU time: "<<computime<<endl;
	cout<<"GPU Time for Path Extraction: "<<computime1<<endl;
	printf("GPU TIME: %f\n",computime);
	
	//copy od matrix of mean and std from gpu global memory back to host memory
	err = cudaMemcpy(meanM, d_meanM, size*size*sizeof(float), cudaMemcpyDeviceToHost);
	err = cudaMemcpy(stdM, d_stdM, size*size*sizeof(float), cudaMemcpyDeviceToHost);

	//writeData("./mean.txt",meanM);//export to ascii text file
	//writeData("./std.txt",stdM); //export to ascii text file
	writeBinData("./mean.bin",meanM,size); //export to binary file
	writeBinData("./std.bin",stdM,size); //export to binary file

	endGPU();
#endif
	if(isPathExtract){
		if(next_m!=NULL) delete [] next_m;
		if(m!=NULL) delete [] m;
		if(aPath!=NULL) delete [] aPath;
		if(probaTable!=NULL) delete [] probaTable;
		if(meanM!=NULL) delete [] meanM;
		if(stdM!=NULL) delete [] stdM;
		if(seed_random!=NULL) delete [] seed_random;
	}
	return 0;
}
float* transpose(float*source,float* des, int nr,int nc){
	int i,j;
	for(i=0;i<nr;i++){
		for(j=0;j<nc;j++){
			des[i*nc+j]=source[j*nr+i];
		}
	}
	return des;
}
void readParams(string fn){
	char str[200];
	int val;
	ifstream f;
	f.open(fn.c_str());
	f>>str>>str;
	dataName=str;
	f>>str>>val;
	size=val;
	f>>str>>val;
	niter=val;
	f>>str>>val;
	block_size=val;	
	f>>str>>val;
	block_size2=val;	
	f.close();

}
void printParams(){
	cout<<"File Name:\t"<<dataName<<endl;
	cout<<"#Nodes:\t"<<size<<endl;
	cout<<"#Max_iterations:\t"<<niter<<endl;
	cout<<"Block Size1:\t"<<block_size<<endl;
	cout<<"Block Size2:\t"<<block_size2<<endl;
}

float* readMatrix(string fn,float* s){
        ifstream f;
        f.open(fn.c_str());
        int i,j,n;
        float val;
        n=size;
        for(i=0;i<n;i++){
                for(j=0;j<n;j++){
                        f>>val;
                        s[i*n+j]=val;
                }
        }


        f.close();
        return s;
}

int* readNext(string fn,int*next){
	ifstream f;
	f.open(fn.c_str());
	int i,j,n=size;
	int val;
	for(i=0;i<n;i++){
		for(j=0;j<n;j++){
			f>>val;
			next[i*size+j]=val;
		}
	}
	f.close();

	return next;
}

void readData(string fn){
	ifstream f;
	f.open(fn.c_str());
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
	f.open(fn.c_str());
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
void initialize_next(float *d,int*pa){
        int i,j;
        int n=size;
        for(i=0;i<n;i++){
                for(j=0;j<n;j++){
                        if(i==j)
                                pa[i*n+j]=-1;//self
                        //else
                        //      pa[i*n+j]=j;
                        else if (abs((d[i*n+j]-STAR))>0.00000001)
                                pa[i*n+j]=j;
                        else
                                pa[i*n+j]=-2;//not connected
                }
        }
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
void printNext(int*pa,char const* fn){
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
        f.close();
}
void extractPathAll_floyd(int*pa){
        int i,j;
        int n=size;

        for(i=0;i<n;i++){
                if(i==2)
                        int stop=1;
                cout<<i<<endl;
		for(j=0;j<n;j++){
                      if(i==24&&j==0)
                                int stop1=1;
			cout<<i<<"\t"<<j<<endl;
                       extractPath_floyd(pa,i,j);
                }
        }
}


void extractPath_floyd(int*pa, int from,int to){
        int n=size;
        int counter=1;
        int cur=to; //reverse
        if(pa[from*size+to]==-2||pa[from*size+to]==-1)
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
                cout<<aPath[i]<<"\t";
        }
        cout<<endl;
        return;
}

#ifdef CUDA
void startGPU(){
	// Error code to check return values for CUDA calls
    	cudaError_t err = cudaSuccess;

	err = cudaMalloc((void **)&nextm_gpu, size*size*sizeof(int));
	cout<<"nextm_gpu: "<<nextm_gpu<<endl;
	cout<<"err: "<<err<<endl;
	cout<<"copy data from host memory to device memory..."<<endl;
	err = cudaMemcpy(nextm_gpu, next_m, size*size*sizeof(int), cudaMemcpyHostToDevice);

	err = cudaMalloc((void **)&m_gpu, size*size*sizeof(float));

        cout<<"copy distance data from host memory to device memory..."<<endl;
        err = cudaMemcpy(m_gpu, m, size*size*sizeof(float), cudaMemcpyHostToDevice);

	err = cudaMalloc((void **)&d_seed_random, size*size*sizeof(int));
	
	seed_random=new int[size*size];
	int i;
	for(i=0;i<size*size;i++){
		seed_random[i]=rand();
	}

	//need create seed_random
        cout<<"copy data from host memory to device memory..."<<endl;
        err = cudaMemcpy(d_seed_random, seed_random, size*size*sizeof(int), cudaMemcpyHostToDevice);

	err = cudaMalloc((void **)&d_probaTable, sizeTable*sizeof(float));

        cout<<"copy data from host memory to device memory..."<<endl;
        err = cudaMemcpy(d_probaTable, probaTable, sizeTable*sizeof(float), cudaMemcpyHostToDevice);
	
	err = cudaMalloc((void **)&d_meanM, size*size*sizeof(float));

        cout<<"copy distance data from host memory to device memory..."<<endl;
        err = cudaMemcpy(d_meanM, meanM, size*size*sizeof(float), cudaMemcpyHostToDevice);

	err = cudaMalloc((void **)&d_stdM, size*size*sizeof(float));

        cout<<"copy distance data from host memory to device memory..."<<endl;
        err = cudaMemcpy(d_stdM, stdM, size*size*sizeof(float), cudaMemcpyHostToDevice);

	cout<<"d_meanM: "<<d_meanM<<endl;
	cout<<"d_meanM: "<<d_stdM<<endl;

	//calculate grid size
	grid_size=size*size/(block_size*size_perthread)+(size*size%(block_size*size_perthread)==0?0:1);
	
	//StartTimer();
}
void endGPU(){
	//EndTimer();
	// Free device global memory
	cudaError_t err = cudaSuccess;
	cout<<"nextm_gpu: "<<nextm_gpu<<endl;
	err= cudaFree(nextm_gpu);
	
	err=cudaFree(m_gpu);
	err=cudaFree(d_probaTable);
	err=cudaFree(d_seed_random);
	err=cudaFree(d_meanM);
	err=cudaFree(d_stdM);

        if (err != cudaSuccess)
        {
                fprintf(stderr, "Failed to free device nextm_gpu (error code %s)!\n", cudaGetErrorString(err));
                exit(EXIT_FAILURE);
        }

}

__global__ void extractPathAll_path_gpu(int*N,int*len,float*dist,int size,int begin,int nrows,int maxLen,int size_perthread,float*probaTable,int sizeTable,int*rand,float rate_perturb,float*meanM,float*stdM){
	int idx=blockIdx.x*blockDim.x+threadIdx.x;
        int i,j;
        if(idx>=size*size) return;
		//if(idx>=nrows*size) return;
        int ii,id;

        for(ii=0;ii<size_perthread;ii++){
                id=idx*size_perthread+ii;
                i=id/size;
                j=id%size;
				i=begin+i;
				if(i>=size) return; //row ID exceeds #rows
				extractPath_path_gpu(N,len,dist,size,i,j,begin,nrows,maxLen,probaTable,sizeTable,rand,rate_perturb,meanM,stdM);
        }
}
__device__ void extractPath_path_gpu(int*N,int*len,float*dist,int size,int from, int to,int begin,int nrows,int maxLen,float*probaTable,int sizeTable,int* rand, float rate_perturb,float*meanM,float*stdM){
	int idx=blockIdx.x*blockDim.x+threadIdx.x;
	if(idx>size*size) return;

	int cur,cur1;
    int counter=1;//counter=0 for path length
	int path[Max_Len_Path];//store nodes of a path
	//len matrix now is cumulative length
	float sumD=0;
	float sumD2=0;//sum of square x
	float val;
        int randvalue;
	float base=pow(2.0f,32.0f),p;
	cur=from;
	cur1=cur;
	path[0]=cur;//first node

	if(N[from*size+to]<0)//-1/-2 will return
		return;
	while(cur!=to){
		cur=N[cur*size+to];
		path[counter]=cur;
        counter++;
    }
	//comment out the following for Monte Carlo resampling of travel time
	//return;
	//start to resample travel time using a Monte Carlo apporach
	//nMC Monte Carlo
	int nMC=1000;//set to 1000
	int i,j;
	int nid1,nid2;
	int upper=97500,lower=2500;
	float d;//distance
	int pos;
	float X;
	float pathD=0;
	float location;
	float scale;
	float r=0.1;//rate of change  
	r=rate_perturb;//get it from argument
	p=0.0;//initialize
	randvalue=rand[from*size+to];

	for(i=0;i<nMC;i++){
		pathD=0;
		for(j=0;j<counter-1;j++){
			//perturb each edge
			nid1=path[j];	
			nid2=path[j+1];
			d=dist[nid1*size+nid2];
			while(p<=0.025||p>=0.975){	//for normal distribution 5% cut off	
				randvalue=RANDOM_A*randvalue+RANDOM_B;
				randvalue=(int)randvalue%(int)base;
				p=randvalue*4.656612e-10;//normlized random value
			}
			pos=p*100000;
			X=probaTable[pos];
			location=d+0.5*d*r;
			scale=0.25*d*r;
			X=X*scale+location;
			pathD+=X;
		}	
		sumD+=pathD;
		sumD2+=pathD*pathD;
	}
	rand[from*size+to]=randvalue;//copy back
	//derive mean and std
	float mean=sumD/(float)nMC;
	float std=sqrt(sumD2/(float)nMC-mean*mean);
	//write back to global memory
	meanM[from*size+to]=mean;
	stdM[from*size+to]=std;
}

#endif

void readProbaFile(string fn){
        ifstream f;
        f.open(fn);

        int i;
        float x,y;
        int v1,v2,v3;
        for(i=0;i<sizeTable;i++){
                f>>x;//>>v2;//>>v3>>x>>y;
                probaTable[i]=x;
        }
        f.close();
}

float* readBinData(string fn,float*s,int n){
        ifstream f;
        f.open(fn.c_str(),ios::in | ios::binary);
        f.read((char*) s,n*n*sizeof(float));
        f.close();
        return s;
}
void writeBinData(string fn,float*s,int n){
        ofstream f;
        f.open(fn.c_str(), ios::out | ios::binary);
        if(!f) {
                cout<<"Cannot open file.";
                return ;
        }
        f.write((char*)s,n*n*sizeof(float));
        f.close();
}


