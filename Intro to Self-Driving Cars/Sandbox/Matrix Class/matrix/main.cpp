#include "pch.h"
#include <iostream>
#include "matrix.h"

int main()
{
	std::cout << "Matrix example!\n";
	Matrix myMatrix = Matrix();
	Matrix myMatrix2 = Matrix();

	Matrix rsp = myMatrix.matrix_addition(myMatrix2);
	myMatrix.matrix_print();
	rsp.matrix_print();
}
