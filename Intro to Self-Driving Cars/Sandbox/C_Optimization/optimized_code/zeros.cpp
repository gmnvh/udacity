#include "pch.h"
#include "headers/zeros.h"

using namespace std;

vector < vector <float> > zeros(int height, int width)
{
	return vector < vector <float> >  (height, vector<float> (width, 0.0F));
}