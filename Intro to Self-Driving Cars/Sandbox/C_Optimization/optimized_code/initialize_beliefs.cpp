#include "pch.h"
#include "headers/initialize_beliefs.h"

using namespace std;

// OPTIMIZATION: pass large variables by reference
vector< vector <float> > initialize_beliefs(vector< vector <char> > grid)
{
    // initialize variables for new grid
    int height, width;

    height = grid.size();
    width = grid[0].size();

    return vector < vector <float> >(height, vector<float>(width, 1.0 / ((float)height * width)));
}