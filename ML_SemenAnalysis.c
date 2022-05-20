#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "Header.h"

typedef struct arrayinputs ArrayInputs;
typedef struct neuronmain NeuronMain;
typedef struct hiddenlayers HiddenLayers;

ArrayInputs ArrayData;
NeuronMain MainNeuron;
HiddenLayers HiddenNeuron;


int main()
{       
    int count;
    clock_t begin = clock();
    
    /*******Input******/
    data = arrData(); 
    datasets();

    /*******Intialize parameter*******/
    initialize();
    
    /*******Caculation******/
    count = Braintrain();
    Braintest();    

    /******* Output ********/
    printdata(count);
    // plotGraph(count);

    clock_t end = clock();
    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("Time spent: %.2f\n", time_spent);
    return 0;
}

float **arrData()
{
    float **data = (float **)malloc(rows * cols); //2D array row and column
    printf("\nPlease enter the file path for your dataset: ");
    scanf("%s", DATASET_FILEPATH);

    FILE *file = fopen(DATASET_FILEPATH, "r"); //open file from location with read only
    if(file == NULL){
        printf("File could not be opened!");
        exit(1);
    }

    int i, j;
    for (i = 0; i < rows; i++) //assign of the rows
    {
        data[i] = (float *)malloc(cols * sizeof(float *)); // assign memory to the cols
        for (j = 0; j < cols; j++)                         // assigning individual cols
        {
            fscanf(file, "%f,", &data[i][j]); //adding the data into the 2D array
        }
    }
    
    fclose(file); //close file
    return data;
}

float datasets() 
{
    int i,j;
    //Storing training data attributes into array traintruex//
    for (i = 0; i < traindivider; i++) 
    {
        for (j = 0; j < cutoff; j++)
        {
            ArrayData.traintruex[i][j] = data[i][j];
        }
    }
    
    //Storing training data output into array traintruey//
    for (i = 0; i < traindivider; i++)
    {
        ArrayData.traintruey[i][1]=data[i][9];
    }

    //Storing testing data attributes into array testtruex//
    for (i = 0; i < 10; i++)
    {
        for (j = 0; j < cutoff; j++)
        {
            ArrayData.testtruex[i][j] = data[i+traindivider][j];
        } 
    }
    
    //Storing testing data output into array testtruey//
    for (i = 0; i < 10; i++)
    {
        ArrayData.testtruey[i][1]=data[i+traindivider][9];
    }
    return 0;
}

void initialize()
{
    int i,j;
    //Initialize weights and bias with random values ranging from -1 to 1//
    for (i = 0; i < neurons; i++)
    {
    MainNeuron.mainweight[i] = (float)rand()/RAND_MAX*2.0-1.0;
    }
    MainNeuron.mainb = (float)rand()/RAND_MAX*2.0-1.0;

    for (j = 0; j < neurons; j++)
    {
        for (i = 0; i < cutoff; i++)
        {
            HiddenNeuron.hiddenweights[j][i]=(float)rand()/RAND_MAX*2.0-1.0;
        }
        HiddenNeuron.hiddenbias[j]=(float)rand()/RAND_MAX*2.0-1.0;
    }
}

int Braintrain()
//Calculation for training set//
    {
    int i;
    int iteration=1;

    //Loops till MAE is smaller then MAE tolerance
    do                                  
    {
        /*Input to hidden layer calculation*/
        Hiddenlayer(traindivider,ArrayData.traintruex);                        

        /*Hidden layer to output calculation*/
        for (i = 0; i < traindivider; i++)
        {
            mainlinearregression(i);        //Calculate linear regression
            mainsigmoidactivation(i);       //Calculate sigmoid activation function
        }
        
        /*Calculate MMSE for iteration one of training set*/
        if (iteration<2)
        {MMSE(0,traindivider,ArrayData.traintruey);}  //Calculate untrained MMSE for training set
        
        CalculateMAE(iteration);            //Calculate MAE of the iteration
        parameterupdate();                  //Calculate new weight and bias

        iteration++;                        //Increase iteration by 1
    } while (MainNeuron.MAE[iteration-1] > MAEtol);    //Check if MAE is smaller than MAE tolerance of 0.25, iteration -1 as we just did iteration++   
   
    MMSE(1,traindivider,ArrayData.traintruey);            //Calculate trained MMSE for training set
    confusionmatrix(traindivider,ArrayData.traintruey);   //Calculate confusion matrix of training data
    return iteration;                           //Return iteration to know how many iteration there were
    }

int Braintest()
{
    /*Calculate MMSE for train test*/
    int i;
    
    /*Input to hidden layer calculation*/
    Hiddenlayer(testdivider,ArrayData.testtruex);

        for (i = 0; i < testdivider; i++)
        {
            mainlinearregression(i);        //Calculate linear regression
            mainsigmoidactivation(i);       //Calculate sigmoid activation 
        }

    MMSE(3,testdivider,ArrayData.testtruey);          //Calculate MMSE of trained testing data set
    confusionmatrix(testdivider,ArrayData.testtruey); //Calculate confusion matrix of training/testing data

    /*Calculate MMSE for untrain test*/
    initialize();                           //Change weight and bias prior to parameter update
                                            //Rand values stays the same unless Srand.

    Hiddenlayer(testdivider,ArrayData.testtruex);     //Calculate hidden layer values

        for (i = 0; i < testdivider; i++)
        {
            mainlinearregression(i);        //Calculate linear regression
            mainsigmoidactivation(i);       //Calculate sigmoid activation
        }      
    MMSE(2,testdivider,ArrayData.testtruey);          //Calculate MMSE of untrained testing data set
    return 0;
}

void Hiddenlayer(int dataset, float arr1[traindivider][cutoff])
{
    /*Calculates output for hidden layer*/
    int i,j;
    for (j = 0; j < neurons; j++)           //Changes the array to store Linear regression and sigmoid activation based on neuron
    {
    for (i = 0; i < dataset; i++)      
        {
        hiddenlinearregression(i,j,arr1);        //Calculate linear regression of the dataset
        hiddensigmoidactivation(i,j,arr1);       //Calculate sigmoid activation of the dataset
        }
    }
}

void hiddenlinearregression(int dataset, int neuron, float arr1[traindivider][cutoff])
/********************************************************/
// dataset   : The current dataset of training/testing set
// neuron    : The current neuron
/********************************************************/
{
    int i;
    float temp=0,tempLR=0; 

    /*calculating Linear regression*/
    {
        for (i = 0; i < cutoff; i++)    //loop to calculate w*x for all 9 attributes
    {
        temp = HiddenNeuron.hiddenweights[neuron][i] * arr1[dataset][i];
        tempLR += temp;
    }
    HiddenNeuron.hiddenLR[neuron][dataset] = tempLR + HiddenNeuron.hiddenbias[neuron];             //add bias
    }
}

void hiddensigmoidactivation(int dataset, int neuron, float arr1[traindivider][cutoff])
/********************************************************/
// dataset   : The current dataset of training/testing set
// nueron    : The current nueron
/********************************************************/
{
    float sigmoid=0;

    //Calculate sigmoid and store into guessy array//
    sigmoid = 1 / (1 + exp(-HiddenNeuron.hiddenLR[neuron][dataset]));
    HiddenNeuron.hiddenguessy[neuron][dataset]=sigmoid;
}

void mainlinearregression(int a)
{
    /********************************************************/
    // a   : The current dataset of training/testing set
    /********************************************************/
    int i;
    float temp=0;

    for (i=0; i<neurons; i++)
    {
        temp += MainNeuron.mainweight[i] * HiddenNeuron.hiddenLR[i][a];
    }
    MainNeuron.mainLR[a] = temp + MainNeuron.mainb;
}

void mainsigmoidactivation(int a)
{
    float sigmoid=0;
    sigmoid = 1 / (1 + exp(-MainNeuron.mainLR[a]));
    MainNeuron.mainguessy[a]=sigmoid;
}

void CalculateMAE(int count)
/********************************************************/
// count    : count is iteration
/********************************************************/
{
    int i;
    float temp=0;
    
    //Calculating MAE for current dataset//
    for (i = 0; i < traindivider; i++)          //Sum up all the MAE
    {
    temp += fabs(MainNeuron.mainguessy[i]-ArrayData.traintruey[i][0]);
    }
    MainNeuron.MAE[count] = temp/traindivider;             //Get average MAE
}

void MMSE(int a, int c, float arr1[traindivider][1])
{
/********************************************************/
// a    : determine the array element to store results
// c    : determine number of training or testing dataset
// arr1 : arr1 is y of training/testing   
/********************************************************/
    int i;
    float mmse=0;
    
    for(i=0; i < c; i++)                            //Sum all MMSE dataset
        {    
        mmse += pow(MainNeuron.mainguessy[i]-arr1[i][0],2);    
        }                    
    MainNeuron.MSE[a]=mmse/c;                                  //Get average for all MMSE
}

void parameterupdate()
{
/****************Calculating main nueron deltaeoverw and deltaeoverb************************/
    float yminusd=0;
    float arrErroroutput[traindivider]={0};
    float erroroutput=0;
    float erroroutput1=0;
    float deltaeoverw[neurons]={0};
    float deltaeoverb=0;
    int i, j;

    for (i = 0; i < traindivider; i++)
    {
        yminusd=(MainNeuron.mainguessy[i]-ArrayData.traintruey[i][0]);
        arrErroroutput[i] = yminusd * ((exp(MainNeuron.mainLR[i]) / pow(1 + exp(MainNeuron.mainLR[i]), 2)));
        erroroutput += arrErroroutput[i];
    }  
    deltaeoverb = erroroutput/traindivider;

    for (j = 0; j < neurons; j++)
    {
        for (i = 0; i < traindivider; i++)
    {
        erroroutput1 += arrErroroutput[i]*HiddenNeuron.hiddenguessy[j][i];
    }
    deltaeoverw[j] = erroroutput1/traindivider;
    erroroutput1=0; 
    }
       
/****************Calculating hidden layer deltaeoverw and deltaeoverb************************/
    float hiddenerroroutput[neurons]={0};
    float hiddenerroroutputneuron[neurons][traindivider]={0};
    float hiddendeltaeoverw[neurons][cutoff]={0};
    float hiddendeltaeoverb[neurons]={0};
    int r;
    float temp=0, temp1=0, temp2=0;
    erroroutput1=0;

for (j = 0; j < neurons; j++)
{
    for (i = 0; i < traindivider; i++)
    {
    temp= MainNeuron.mainweight[j]*((exp(HiddenNeuron.hiddenLR[j][i]) / pow(1 + exp(HiddenNeuron.hiddenLR[j][i]), 2)));
    hiddenerroroutputneuron[j][i] = arrErroroutput[i]*temp;
    erroroutput1 += hiddenerroroutputneuron[j][i];
    }
    hiddendeltaeoverb[j]=erroroutput1/traindivider;
    erroroutput1=0;
}

for (r = 0; r < neurons; r++)
    {
    for (j = 0; j < cutoff; j++)
        {
        for (i = 0; i < traindivider; i++)
            {
            temp1 += hiddenerroroutputneuron[r][i]*ArrayData.traintruex[i][j];
            }
            hiddendeltaeoverw[r][j]=temp1/traindivider;
            temp1=0;  
        }
    }

/*********************Update main neuron weights and bias********************************/
    for (i = 0; i < neurons; i++)
    {
        MainNeuron.mainweight[i] = MainNeuron.mainweight[i]- rateLearn*deltaeoverw[i];
    }
        MainNeuron.mainb = MainNeuron.mainb - rateLearn*deltaeoverb;

/*********************Update hidden layer weights and bias********************************/
    for (j = 0; j < neurons; j++)
    {
        for (i = 0; i < cutoff; i++)
        {
            HiddenNeuron.hiddenweights[j][i]= HiddenNeuron.hiddenweights[j][i] - rateLearn*hiddendeltaeoverw[j][i];
        }
        HiddenNeuron.hiddenbias[j]= HiddenNeuron.hiddenbias[j] - rateLearn*hiddendeltaeoverb[j];    
    }
}

void confusionmatrix(int a, float arr1[rows][1])
/********************************************************/
// a    : number of dataset
// arr1 : y for training/testing dataset 
/********************************************************/
{
    int i=0, TP=0, TN=0, FP=0, FN=0, temp=0;
    float temp1;

    /*Compare to see if */
    for (i = 0; i < a; i++)
    {
        temp=MainNeuron.mainguessy[i]>0.30?1:0;    
        if ((temp == 1) && (arr1[i][0] == 1))
        {
            TP++;    
        }
        if ((temp == 0) && (arr1[i][0] == 0))
        {
            TN++;    
        }
        if ((temp == 0) && (arr1[i][0] == 1))
        {
            FN++;    
        }
        if ((temp == 1) && (arr1[i][0] == 0))
        {
            FP++;    
        }
    }
    if (a>89)   //Store for training set
    {   
        MainNeuron.confusematrix[0]=TP;
        MainNeuron.confusematrix[1]=TN;
        MainNeuron.confusematrix[2]=FN;
        MainNeuron.confusematrix[3]=FP;
    }
    else        //Store for testing set
    {
        MainNeuron.confusematrix[4]=TP;
        MainNeuron.confusematrix[5]=TN;
        MainNeuron.confusematrix[6]=FN;
        MainNeuron.confusematrix[7]=FP;
    }
}

void printdata(int numberofiteration)
/********************************************************/
// numberofiteration    : no of iteration for training set
/********************************************************/
{
    int i;
    for (i = 1; i < numberofiteration; i++) //Print training set MAE values
    {
     printf("Iteration %d , MAE Values: %f\n", i, MainNeuron.MAE[i]);   
    }
    printf("MAE Values stopped: %f\n",MainNeuron.MAE[numberofiteration-1]);

    //Print training and testing set MMSE values
    printf("\nuntrained training set MMSE: %f\n", MainNeuron.MSE[0]);
    printf("Trained training set MMSE: %f\n", MainNeuron.MSE[1]);
    printf("untrained testing set MMSE: %f\n", MainNeuron.MSE[2]);
    printf("Trained testing set MMSE: %f\n", MainNeuron.MSE[3]);        

    //Print training confusion matrix values
    printf("\nTraining set\nTrue postive:%d\n", MainNeuron.confusematrix[0]);
    printf("True negative:%d\n", MainNeuron.confusematrix[1]);
    printf("False negative:%d\n", MainNeuron.confusematrix[2]);
    printf("False positive:%d\n", MainNeuron.confusematrix[3]);

    //Print training confusion matrix values
    printf("\nTesting set\nTrue postive:%d\n", MainNeuron.confusematrix[4]);
    printf("True negative:%d\n", MainNeuron.confusematrix[5]);
    printf("False negative:%d\n", MainNeuron.confusematrix[6]);
    printf("False positive:%d\n", MainNeuron.confusematrix[7]);
}

void plotGraph(int numberofiteration)
{
   int i, j=0;

    FILE * fp;
    fp = fopen("training.txt", "w"); //Creates or opens new txt file to plot MAE
    for (i = 1; i < numberofiteration; i++)
    {
        fprintf(fp, "%d\t%lf\n", i, MainNeuron.MAE[i]);
    }
    fclose(fp);

    FILE *pipe_gp = popen(GNUPLOT, "w");
    fputs("set autoscale x\n",pipe_gp);
    fputs("set autoscale y\n",pipe_gp);
    fputs("plot 'training.txt' lt 2 lw -1\n",pipe_gp);
    pclose(pipe_gp);
}