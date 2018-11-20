#include "pch.h"
#include "headers/normalize.h"
#include "headers/zeros.h"
using namespace std;

// OPTIMIZATION: Pass variable by reference
vector< vector<float> > normalize(vector< vector <float> > &grid)
{
    float sum = 0;
    int i, j, height, width;

    height = grid.size();
    width = grid[0].size();
    vector< vector<float> > newGrid = zeros(height, width);

    /* Calculate the sum of all elements of the grid */
    for (i = 0; i < height; i++)
    {
        for (j = 0; j < width; j++)
        {
            sum = sum + grid[i][j];
        }
    }

    /* Update newGrid with normalized value */
    for ( i = 0; i < height; i++)
    {
        for (j = 0; j < width; j++)
        {
            newGrid[i][j] = (grid[i][j] / sum);
        }
    }

    return newGrid;
}
