#include "pch.h"
#include "headers/blur.h"

using namespace std;

// OPTIMIZATION: Pass large variable by reference
vector < vector <float> > blur(vector < vector < float> > &grid, float blurring)
{

	// OPTIMIZATION: window, DX and  DY variables have the 
    // same value each time the function is run.
  	// It's very inefficient to recalculate the vectors
    // every time the function runs. 
    // 
    // The const and/or static operator could be useful.
  	// Define and declare window, DX, and DY using the
    // bracket syntax: vector<int> foo = {1, 2, 3, 4} 
    // instead of calculating these vectors with for loops 
    // and push back
	int height;
	int width;

	height = grid.size();
	width = grid[0].size();

    vector < vector <float> > newGrid(height, vector<float>(width, 0.0F));
    float center_prob, corner_prob, adjacent_prob;

    center_prob = 1.0 - blurring;
    corner_prob = blurring / 12.0;
    adjacent_prob = blurring / 6.0;

    /* Initialize window mask that should be convoluted with the grid */
    vector< vector<float> > window = { {corner_prob,   adjacent_prob, corner_prob  },
                                       {adjacent_prob, center_prob,   adjacent_prob},
                                       {corner_prob,   adjacent_prob, corner_prob  } };

    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            for (int window_row = -1; window_row < 2; window_row++)
            {
                for (int window_col = -1; window_col < 2; window_col++)
                {
                    float mult = window[window_row + 1][window_col + 1];
                    int new_i = (i + window_row + height) % height;
                    int new_j = (j + window_col + width) % width;

                    newGrid[new_i][new_j] += mult * grid[i][j];
                }
            }
        }

    }

    return newGrid;
}
