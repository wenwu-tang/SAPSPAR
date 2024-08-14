################################# C++ Source code for CPU-based all-pair shortest path routing algorithm ################################
#include <iostream>
#include <fstream>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#define STAR 99999999

using namespace std;

float* m,*m1,*m2;
int size=0;
string dataName;
int isPathExtract=1;
int niter=50;//maximum number of iterations
int niter_stop;
int *aPath;// store IDs of nodes
int stopFlag=1;//0: stop

void readParams(string fn);
void printParams();
void readData(string fn);
void writeData(string fn,float*s);
void printPath();
//Floyd-Warshall algorithm
int *next_m; //store next node for shorest path
void floyd_warshall(float*d,int*pa);
void extractPathAll_floyd(int*pa);
void extractPath_floyd(int*pa, int from,int to,ofstream &f, int flag);
void printNext(int*pa,string fn);


int main(int argc, char* argv[]);

int main(int argc, char* argv[])
{
	int i;
	time_t start,end,start1,end1; double diff;

	cout<<"read parameters..."<<endl;
	readParams("./params.txt");
	printParams();
	
	m=new float[size*size];
	next_m=new int[size*size];

	cout<<"read data..."<<endl;
	readData(dataName);

	cout<<"allocated memory"<<endl;
	
	if(isPathExtract){
		//size of aPath is #nodes for floy-warshall algorithm
		aPath=new int[size];
		//aPath={-1};//use -1 as a flag 
		memset(aPath,-1,sizeof(int)*size); 
	}
	
	time(&start);
	cout<<"Floyd-Warshall Algorithm..."<<endl;
	floyd_warshall(m,next_m);	

        time(&end1);
        diff=difftime(end1,start);
        cout<<"Computing time for Floyd-Warshall operations: "<<diff<<endl;	
	//to export od matrix of predecessors, enable the following line
	printNext(next_m,"./next_floyd.txt");
	//enable the following to export od matrix of distance or travel time
	writeData("./dist_floyd.txt",m);

	//extract paths (as needed); otherwise, comment them out.
	cout<<"Extract path..."<<endl;
	time(&start1);
	extractPathAll_floyd(next_m);
	cout<<"Done extracting path"<<endl;
	
	//timing
	time(&end);
	diff=difftime(end,start1);
	cout<<"Path Extraction Time: "<<diff<<endl;

	diff=difftime(end,start);
	cout<<"Total Computing Time: "<<diff<<endl;

	//release memory
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
    //f>>str>>val;
    //block_size=val;
	f.close();

}
void printParams(){
    cout<<"File Name:\t"<<dataName<<endl;
    cout<<"#Nodes:\t"<<size<<endl;
    cout<<"#Max_iterations:\t"<<niter<<endl;
    //cout<<"Block Size:\t"<<block_size<<endl;
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
void printPath(){
	int i,n;
	n=niter_stop;
	for(i=0;i<n+1;i++){
		if(aPath[i]!=-1){
			cout<<aPath[i]+1;
			if(i!=n)
				cout<<"->";
		}
	}
	cout<<endl;
}
//// Floyd-Warshall algorithm

void floyd_warshall(float*d,int*pa){
	int k,i,j;
	int n=size;
	//initialize 
	for(i=0;i<n;i++){
		for(j=0;j<n;j++){
			if(i==j) 
				pa[i*n+j]=-1;//self
			else if(abs((d[i*n+j]-STAR))>0.00000001)			
				pa[i*n+j]=j;// start from 0
			else
				pa[i*n+j]=-2;//not connected
		}
	}

	//O(n3) loop
	for(k=0;k<n;k++){
		//cout<<k+1<<endl;
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
	int flag=0; //1: print 2 file
	ofstream f;
	if(flag){
		f.open("./path_floyd.txt");
	}	
	for(i=0;i<n;i++){
		for(j=0;j<n;j++){
			extractPath_floyd(pa,i,j,f,flag);
		}
	}
	if(flag){
		f.close();
	}
}

void extractPath_floyd(int*pa, int from,int to,ofstream & f,int flag){
	int n=size;
	int counter=1;
	int cur=from;
	if(pa[from*size+to]==-2)//not connected
		return;
	aPath[0]=from;
	cur=from;
	while(cur!=to){
		cur=pa[cur*n+to];
		aPath[counter]=cur;
		counter++;
	}

	//print
	if(flag){
	int i;
	
	for(i=0;i<counter;i++){
		f<<aPath[i]<<"\t";
	}	
	f<<endl;

	}
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

 
