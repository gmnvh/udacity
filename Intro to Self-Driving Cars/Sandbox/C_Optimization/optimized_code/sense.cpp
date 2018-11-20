#include "pch.h"
#include "headers/sense.h"

using namespace std;

// OPTIMIZATION: Pass larger variables by reference
vector< vector <float> > sense(char color, vector< vector <char> > &grid, vector< vector <float> > beliefs,  float p_hit, float p_miss) 
{
    int hit;
    int i, j;

    for (i=0; i < grid.size(); i++) {
        for (j=0; j < grid[0].size(); j++) {
            hit = (grid[i][j] == color);
            beliefs[i][j] = beliefs[i][j] * hit * p_hit + (1 - hit) * p_miss * beliefs[i][j];
        }
    }
    return beliefs;
}
