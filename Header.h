#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define rows 100            //Number of dataset
#define cols 10             //Number of attributes
#define neurons 7           //Number of neuron for hidden layer
#define MAX 20000           //Maximum number of iteration
#define MAEtol 0.18         //means avg error tolerance
#define rateLearn 0.05      //learning rate
#define traindivider 90     //Number of training data sets
#define testdivider 10      //Number of test data sets
#define cutoff 9            //differentiate from x input and d input
#define GNUPLOT "/usr/local/Cellar/gnuplot/5.4.0_1/bin/gnuplot -p"  //Filepath for gnuplot
char DATASET_FILEPATH[100]; //String for user input filepath

/**********Variables************/
//Arrays for input
float **data;                           //pointer ** towards the array
struct arrayinputs
{
float traintruex[traindivider][cutoff]; //2D array for the training attributes
float traintruey[traindivider][1];      //2D array for the training y 
float testtruex[10][cutoff];            //2D array for the testing attributes
float testtruey[10][1];                 //2D array for the testing y
};

//Main nueron (hidden layer to output) variables for calculation
struct neuronmain
{
float mainweight[neurons];              //Weights for main neuron
float mainb;                            //Mainbias
float mainLR[traindivider];             //Linear regression for main nueron
float MAE[MAX];                         //MAE for each iteration
float MSE[4];                           //MMSE for each iteration
float mainguessy[traindivider];         //guess y, after the sigmoid activation function    
int   confusematrix[8];                 //contains training and testing for TP,TN,FP,FN
} mainweight[neurons]={0},mainb = {0};

struct hiddenlayers
{
//hidden layer (input to hidden layer) arrays for calculation
float hiddenweights[neurons][cutoff];       //Weights for hidden nueron
float hiddenLR[neurons][traindivider];      //Linear regression for each dataset
float hiddenbias[neurons];                  //hidden bias
float hiddenguessy[neurons][traindivider];  //Output of hiddenlayer
} 
temphiddenweight[neurons][cutoff]={0},
hiddenweights[neurons][cutoff]={0},
hiddenbias[neurons]={0},
hiddenguessy[neurons][traindivider]={0};

/**********Function************/
//Input
float **arrData();                      //read file
float datasets();                       //store all the files into different arrays
void initialize();                      //initialize weight and bias

//Calculation
int Braintrain();                       //Contains all the calculation function for training set
int Braintest();                        //Contains all the calculation function for training set

/*Main nueron calculation functions*/
void mainlinearregression(int a);       //Calculation linear regression in the main neuron
void mainsigmoidactivation(int a);      //Calculation sigmoid activation in the main neuron 
void CalculateMAE(int count);           //Calculate MAE for each iteration

/*Hidden nueron calculation functions*/
void Hiddenlayer(int dataset, float arr1[traindivider][cutoff]);
void hiddenlinearregression(int dataset, int neuron, float arr1[traindivider][cutoff]);  //Calculation linear regression in the hidden neuron
void hiddensigmoidactivation(int dataset, int neuron, float arr1[traindivider][cutoff]); //Calculation sigmoid activation in the hidden neuron  

/*General calculation function*/
void parameterupdate();                                 //Does parameter update for main and hidden neuron
void MMSE(int a, int c, float arr1[traindivider][1]);   //Calculate MMSE     
void confusionmatrix(int a, float arr1[rows][1]);       //Calculate confusion matrix

//Output
void printdata(int numberofiteration);                  //Print MAE,MMSE,Confusion matrix        
void plotGraph(int numberofiteration);                  //Plot graph for training set