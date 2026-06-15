#include "gelu_omp.h"

#include <cmath>
#include <omp.h>

using namespace std;

vector<float> GeluOMP(const vector<float>& input) 
{
    const size_t inputSize = input.size();
    vector<float> ouput(inputSize);

    const float multVal1 = 2*sqrt(2/M_PI); 
    const float multVal2 = 0.044715;

    #pragma omp parallel for
    for (size_t i = 0; i < inputSize; ++i)
    {
        float tanhArg = multVal1 * (input[i] + multVal2*input[i]*input[i]*input[i]);
        float expVal  = exp(tanhArg);
        float tanhVal = (expVal - 1)/(expVal + 1);
        ouput[i] = input[i]/2 * (1 + tanhVal);
    }

    return ouput;
}