#include<bits/stdc++.h>
using namespace std;
typedef vector<string> LINE;
class Calculation{
    
    public:
    vector<double> meanData;
    vector<double> stdData;
    vector<vector<double>> inputData;
    vector<vector<double>>trainingData;
    vector<vector<double>>testingData;
    int i,j;
    int ReadData();
    int splitingData();
    double calcMean();
    double calcstdDev();
    double LogisticModel();
    int showMeanData();
    int showStdDevData();
    int showTrainingData();
    int ShowTestingData();
    int showLoadedValues();
};


int main(){
    Calculation o1;
    o1.ReadData();
    o1.splitingData();
    double accuracy=o1.LogisticModel();
     // o1.showLoadedValues();
    // o1.ShowTestingData();
    // o1.showTrainingData();
    cout<<endl;
    cout<<"After Traing and Testing of Dataset the accuracy obtained is";
    cout<<accuracy<<"%"<<endl;
    // o1.calcMean();
    // o1.calcstdDev();
    // o1.showMeanData();
    // o1.showStdDevData();
     return 0;
}

// Loading the data
int Calculation::ReadData(){
     string line;
	 int pos;
	vector<LINE> array;
    int i,j;
	ifstream in("EmpAttritionData.csv");
	if(!in.is_open())
	{
		cout << "Failed to open file" << endl;
		return 1;
	}
	while( getline(in,line) )
	{
		LINE ln;
		while( (pos = line.find(',')) >= 0)
		{
			string field = line.substr(0,pos);
			line = line.substr(pos+1);
			ln.push_back(field);
		}
		array.push_back(ln);
	}
  

  for(i=1;i<array.size();i++)
  {
    vector<double>inputRow;
    for(j=0;j<array[i].size();j++)
    {
      double input;
      input = std::stod(array[i][j]);
      inputRow.push_back(input);

    }
    inputData.push_back(inputRow);
  }
  cout<<"Data Loaded"<<endl;
  cout<<"After Loading of Data"<<endl;
   cout<<"No. of rows are="<<inputData[0].size()<<" No. of columns are ="<<inputData.size()<<endl;
  return 0;
}


// Spliting the dataset into training and testing
int Calculation::splitingData()
{
    // training data
    for(int i=0;i<1025;i++)
 {
    vector<double>row1;
    for(int j=0;j<inputData[i].size();j++)
    {
        row1.push_back(inputData[i][j]);
    }
    trainingData.push_back(row1);
 }
//  testing data
for(i=1025;i<inputData.size();i++)
 {
    vector<double>row2;
    for(j=0;j<inputData[i].size();j++)
    {
        row2.push_back(inputData[i][j]);
    }
    testingData.push_back(row2);
 }
 cout<<"dataSplitted"<<endl;
 cout<<"After Splitting of data";
 cout<<"No. of rows in training data are :"<<trainingData[0].size()<<endl;
 cout<<"No. of columns in training data are :"<<trainingData.size()<<endl;
 cout<<"No. of rows in testing data are :"<<testingData[0].size()<<endl;
 cout<<"No. of columns in testing data are :"<<testingData.size()<<endl;

    return 0;
}




double Calculation::LogisticModel()
{
    // trainingPhase
 double b0 = 0, b1 = 0,b2 = 0, b3 = 0,b4 = 0,b5 = 0,b6 = 0,b7 = 0, b8 = 0;
 vector<double> error;
 double err;
 double alpha=0.01;
 double e=2.71828;
 for(int i=1;i<5;i++)
    {
        for(int j=0;j<trainingData.size();j++)
        {
            // cout<<data[j][0]<<" "<<data[j][1]<<" "<<data[j][2]<<" "<<data[j][3]<<" "<<data[j][4];
            // cout<<endl;
           double p = -(b0 + b1 * trainingData[j][1]+ b2* trainingData[j][2]+b3*trainingData[j][3]+b4*trainingData[j][4]+ b5*trainingData[j][5]+ b6*trainingData[j][6]+ b7*trainingData[j][7]+b8*trainingData[j][8]);
            double pred  = 1/(1+ pow(e,p));
            // cout<<pred<<endl;
            err = trainingData[j][0]-pred;
            // cout<<err<<endl;
            b0 = b0 - alpha * err* 1.0;   
            b1 = b1 + alpha * err * pred*(1-pred) *trainingData[j][1];
            b2 = b2 + alpha * err * pred*(1-pred) * trainingData[j][2];
            b3 = b3 + alpha * err *pred *(1-pred)* trainingData[j][3];
            b4 = b4 + alpha * err * pred * (1-pred)*trainingData[j][4];
            b5 = b5 + alpha * err * pred * (1-pred)*trainingData[j][5];
            b6 = b6 + alpha * err * pred * (1-pred)*trainingData[j][6];
            b7 = b7 + alpha * err * pred * (1-pred)*trainingData[j][7];
            b8 = b8 + alpha * err * pred * (1-pred)*trainingData[j][8];
            error.push_back(err);
        }
        


    }
     sort(error.begin(),error.end());
        cout<<"Final Values are: "<<"B0="<<b0<<" "<<"B1="<<b1<<" "<<"B2="<<b2<<"B3="<<b3<<"B4="<<b4<<"B5="<<b5<<"B6="<<b6<<"B7="<<b7<<"B8="<<b8 <<"error="<<error[0];
        cout<<endl;

 // testingPhase
 double count=0;
 for(int i=0;i<testingData.size();i++){
 double pred1=-(b0+b1*testingData[i][1]+b2*testingData[i][2]+b3*testingData[i][3]+b4*testingData[i][4]+b5*testingData[i][5]+b6*testingData[i][6]+b7*testingData[i][7]+b8*testingData[i][8]); //make prediction
 double pred2 = 1/(1+pow(e,pred1));
 // cout<<"The value predicted by the model= "<<pred2<<endl;
 if(pred2>0.5)
 pred2=1;
 else
 pred2=0;

 if(pred2==testingData[i][0])
 count++;
 }
 cout<<"training and testing done";
//  cout<<count<<endl;
 return count/testingData.size()*100;
 
}

int Calculation::showLoadedValues()
{
    int i,j;
    for(i=0;i<inputData.size();i++)
    {
        for(j=0;j<inputData[i].size();j++)
        {
            cout<<inputData[i][j]<<"  ";
        }
        cout<<endl;
    }
    return 0;
}


int Calculation::showTrainingData()
{
    int i,j;
    cout<<"Training Data:"<<endl;
    for(i=0;i<trainingData.size();i++)
    {
        for(j=0;j<trainingData[i].size();j++)
        {
            cout<<trainingData[i][j]<<"  ";
        }
        cout<<endl;
    }
    return 0;
}


int Calculation::ShowTestingData()
{
    int i,j;
    cout<<"Testing Data :"<<endl;
    for(i=0;i<testingData.size();i++)
    {
        for(j=0;j<testingData[i].size();j++)
        {
            cout<<testingData[i][j]<<"  ";
        }
        cout<<endl;
    }
    return 0;
}

double Calculation::calcMean()
{
    int i,j;
    double sum1=0,mean;
    for(i=0;i<inputData[0].size();i++)
    {
        for (j=0;j<inputData.size();j++)
        {
            sum1 += inputData[i][j];
        }
        mean = sum1/inputData.size();
        meanData.push_back(mean);
        
    }
    cout<<"success";
  return 0;
}


double Calculation::calcstdDev()
{
    cout<<"Hello to calcstdDev";
    int rows = inputData.size();
    int cols = inputData[0].size();
    float sum=0;
    float stdDev;
    int i,j;
    for(j=0;j<cols;j++)
    { 
      sum=0;
      for(i=0;i<rows;i++)
      {
       //if(data[i][1]==1)
        sum+=pow(inputData[j][i]-meanData[j],2);
      }
     stdDev= sqrt(sum/rows);
     stdData.push_back(stdDev);
    }
 //cout<<rows<<" "<<cols;
   cout<<"stdDev calculated"<<endl;
   return 0;
}


int Calculation::showMeanData()
{
    int i;
    cout<<"Mean are :"<<endl;
    for(i=0;i<meanData.size();i++)
    {
        cout<<meanData[i]<<"  ";
    }

    return 0;
}

int Calculation::showStdDevData()
{
    int i;
    cout<<"Std Deviations are : "<<endl;
    for(i=0;i<stdData.size();i++)
    {
        cout<<stdData[i]<<"  ";
    }
    return 0;
}
