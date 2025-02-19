#include "pch.h"
#include <iostream>
#include "vector"
#include "matrix.h"

Matrix::Matrix()
{
	vector< vector<float> > initial_grid (10, vector<float> (5, 0.5));
	grid = initial_grid;

	rows = grid.size();
	cols = grid[0].size();
}

Matrix::Matrix(vector< vector<float> > initial_grid)
{
	grid = initial_grid;

	rows = grid.size();
	cols = grid[0].size();
}

void Matrix::setGrid(vector< vector<float> > new_grid)
{
	grid = new_grid;
	rows = grid.size();
	cols = grid[0].size();
}

vector< vector<float> > Matrix::getGrid()
{
	return grid;
}

vector<float>::size_type Matrix::getRows()
{
	return rows;
}

vector<float>::size_type Matrix::getCols()
{
	return cols;
}

Matrix Matrix::matrix_transpose() {
	vector< vector<float> > new_grid;
	vector<float> row;

	for (int i = 0; i < cols; i++) {
		row.clear();

		for (int j = 0; j < rows; j++) {
			row.push_back(grid[j][i]);
		}
		new_grid.push_back(row);
	}

	return Matrix(new_grid);
}

Matrix Matrix::matrix_addition(Matrix toadd)
{
	if ((rows != toadd.getRows()) || (cols != toadd.getCols())) {
		throw std::invalid_argument("matrices are not the same size");
	}

	vector<vector<float>> rsp = grid;
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			rsp[i][j] = grid[i][j] + toadd.getGrid()[i][j];
		}
	}

	return Matrix(rsp);
}

void Matrix::matrix_print()
{
	for (int i = 0; i < rows; i++)
	{
		std::cout << i << "| ";
		for (int j = 0; j < cols; j++)
		{
			std::cout << grid[i][j] << " ";
		}
		std::cout << "\r\n";
	}
}


