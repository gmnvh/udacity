#ifndef MATRIX_H
#define MATRIX_H

#pragma once
#include <vector>

// Header file for the Matrix class
using namespace std;

class Matrix
{
private:

	vector< vector<float> > grid;
	vector<float>::size_type rows;
	vector<float>::size_type cols;

public:

	Matrix();
	Matrix(vector< vector<float> >);

	void setGrid(vector< vector<float> >);
	vector< vector<float> > getGrid();
	vector<float>::size_type getRows();
	vector<float>::size_type getCols();

	Matrix matrix_transpose();
	Matrix matrix_addition(Matrix);
	void matrix_print();
};

#endif /* MATRIX_H */