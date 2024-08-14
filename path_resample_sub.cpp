################################## C++ source code for CPU-based path travel time estimation via Monte Carlo-based resampling ##########################
#include <iostream>
#include <fstream>
#include <string.h>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define STAR 100000000

using namespace std;

float* m,*m1,*m2;
int size=0;
string dataName;
int isPathExtract=1;
int* next_m;
float*mean_m; //od matrix of path mean after perturbation
float*std_m; //od matrix of path standard deviation after perturbation
string probaFile;
float *proba;
int size_proba=100000; 
void readProbaFile(string fn);

int max_length_path=300;//maximum path length in terms of #nodes // predefine
int block_size=128;  //blockDim.x
int block_size2=8;   //blockDim.y

int* lookupTable=NULL;
void readNodeData(string fn);
int niter=100;  // maximum #iterations 
int niter_stop;// #iters when iterations stop
int *aPath;// store IDs of nodes
int stopFlag=1;//0: stop

int*OD_list;//list of OD pairs
int n_ODs; // number of OD pairs to be handled
string ODFile;
string pathFile; //local IDs
string pathFile1;//node IDs
int nSeg=0;//number of path segments in total
int sub_start,sub_end;//start and end index of row-wise domain decomposition 

float* transpose(float*source,float* des, int nr,int nc);
void readParams(string fn);
void printParams();
float* readMatrix(string fn,float*s);
void readData(string fn);
void writeData(string fn,float*s);
void initialize_next(float *d,int*pa);
void extractPathAll();
void printPath(int*pa);
void printNext(int*pa,char const* fn);
//path extraction and travel time estimation using Monte Carlo-based resampling
void extractPathAll_floyd(int*pa);
void extractPath_floyd(int*pa, int from,int to,ofstream &file,ofstream &file1);
void printPathAll1(int*pa,int*len,int size,int totalSize);
void printPathAll2(int*pa,int size,int begin,int nrows,string fn);
//load next matrix
int* readNext(string fn,int*next);

int main(int argc, char* argv[]);

int main(int argc, char* argv[])
{
	int i;
	int origin,dest;
	time_t start,end,end1; double diff;
	sub_start=0; sub_end=21568;
	if(argc>1){
		sub_start=atoi(argv[1]);
                sub_end=atoi(argv[2]);

	}
	cout<<"read parameters..."<<endl;
	readParams("./params.txt");
	printParams();

	m=new float[size*size];
	cout<<"read data..."<<endl;
	readData(dataName);
	cout<<"done reading distance matrix"<<endl;
	if(isPathExtract){
		next_m=new int[size*size];
                next_m=readNext("./next.txt",next_m);
		aPath=new int[max_length_path];
		lookupTable=new int[size];
		readNodeData("./node_ID.txt");
		mean_m=new float[size*size];
		std_m=new float[size*size];
		
		proba=new float[100000];
		readProbaFile("./cdf.txt");
	}
	
	int iter=1;
	string fn;
	int nstep=niter/2;
	//niter=log(size);
	niter=size;
	time(&start);

	cout<<"Extract Path..."<<endl;
	//printNext(next_m,"next_repeat_gpu.txt");
	//sequential algorithm to extract path and estimate path travel time here
	extractPathAll_floyd(next_m);
	cout<<"Done extracting path..."<<endl;
	//timing
	time(&end);
	diff=difftime(end,start);
	cout<<"Total computing Time: "<<diff<<endl;

	cout<<"Write to files..."<<endl;
	
	//writeData("./meanPathDist.txt",mean_m);
	//writeData("./stdPathDist.txt",std_m);

	if(isPathExtract){
		if(m!=NULL) delete [] m;
		if(next_m!=NULL) delete [] next_m;
		if(aPath!=NULL) delete [] aPath;
		if(mean_m!=NULL) delete [] mean_m;
		if(std_m!=NULL) delete [] std_m;
		if(proba!=NULL) delete [] proba;
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
                        printPath(pa);
                }
        }
}

void printPath(int*pa){
	int i;
	int *pa1=pa;
	int n=niter_stop;
	for(i=0;i<n+1;i++){
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
        int i,j,k;
        int n=size;

	ofstream f;
	ofstream f1;
	for(i=sub_start;i<sub_end;i++){
		cout<<i<<endl;
		for(j=0;j<n;j++){
			extractPath_floyd(pa,i,j,f,f1);

		}
	}
	return;

}


void extractPath_floyd(int*pa, int from,int to,ofstream &file,ofstream &file1){
        int n=size;
        int counter=1;
        int cur=to; //reverse
		int i,j;
        if(pa[from*size+to]==-2||pa[from*size+to]==-1)
            return;
        aPath[0]=from;
        cur=from;
        while(cur!=to){
            cur=pa[cur*n+to];
            aPath[counter]=cur;
			if(cur==aPath[counter-1])
				break;
            counter++;
        }
	//comment out the following return code for resampling travel time estimation
	//return;

	//resample path travel time 
	int nid1,nid2;
	float dist;
	int upper, lower;
	lower=2500;
	upper=97500;
	int pos;
	float Xs,X;
	float location,scale;
	float r=0.1; //perturbation rate (pass it from the main function)
	float sum_dist=0;// the distance of the path
	float sum_pathDist=0;//summation of all path distance
	float avg_pathDist=0;
	float std_pathDist=0;
	float sum_square_pathDist=0;
	float pathDist[10000]; //store path distance of 10000 perturbations
	int nMC=1000;//#Monte Carlo runs
	for(i=0;i<nMC;i++){
		sum_dist=0;
		for(j=0;j<counter-1;j++){
			nid1=aPath[j];
			nid2=aPath[j+1];	
			dist=m[nid1*size+nid2];//get the distance/travel time from distance matrix
			pos=rand()%(upper-lower+1)+lower;	
			Xs=proba[pos];
			location=dist+0.5*dist*r;		
			scale=0.25*dist*r;
			X=Xs*scale+location;
			sum_dist+=X;
			pathDist[i]=sum_dist;
		}	
		sum_pathDist+=sum_dist;
		sum_square_pathDist+=sum_dist*sum_dist;
	}

	avg_pathDist=sum_pathDist/(float) nMC;
	
	//calculate standard deviation
	std_pathDist=sqrt(sum_square_pathDist/(float)nMC-avg_pathDist*avg_pathDist);
	mean_m[from*size+to]=avg_pathDist;
	std_m[from*size+to]=std_pathDist;	
	nSeg+=counter;

    return;
}
void printPathAll2(int*pa,int size,int begin,int nrows,string fn){
	int i,j,ii,id;
	int len;
	ofstream f;
	f.open(fn.c_str());
	for(i=0;i<nrows;i++){
		for(j=0;j<size;j++){
			id=i*size+j;
			len=pa[id*max_length_path];
			f<<pa[id*max_length_path]<<"\t";
			for(ii=1;ii<=len;ii++){
				f<<pa[id*max_length_path+ii]<<"\t";
			}
			f<<endl;
				
		}
	}

	f.close();
}

void printPathAll1(int*pa,int*len,int size,int totalSize){
	
	int i,j,k,length;
	int ii,start,end;
	ofstream f;
	f.open("path_repeat_gpu.txt");
	for(i=0;i<size;i++){
		for(j=0;j<size;j++){
			ii=i*size+j;
			start=len[ii];
			if(ii!=size*size-1)
				end=len[ii+1];					
			else //the last one
				end=totalSize;
			length=end-start;
			//cout<<"start-end: "<<start<<"; "<<end<<endl;
			f<<i<<"\t";
			for(k=start;k<end;k++){
				f<<pa[k]<<"\t";		
			}	
			f<<endl;
		}
	}	
	f.close();
}
void readNodeData(string fn){
        ifstream f;
        f.open(fn);

        int i;
        float x,y;
        int v1,v2,v3;
        for(i=0;i<size;i++){
                f>>v1>>v2;//>>v3>>x>>y;
                lookupTable[i]=v2; 
        }

        f.close();

}

void readProbaFile(string fn){
        ifstream f;
        f.open(fn);

        int i;
        float x,y;
        int v1,v2,v3;
        for(i=0;i<size_proba;i++){
                f>>x;//>>v2;//>>v3>>x>>y;
		proba[i]=x;
        }

        f.close();

}
