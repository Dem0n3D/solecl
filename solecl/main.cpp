#include <fstream>
#include <iostream>
#include <string>
#include <sstream>

#include <QVector>
#include <QDebug>

#include "gauss.h"

using namespace std;

int main(int argc, char *argv[])
{
    int N;
    //cout << "Enter N: ";
    //cin >> N;
    N = 1000;
    stringstream fname;
    fname << "matrix/matrix" << N << ".txt";
    ifstream f(fname.str().c_str());

    QVector< QVector<float> > A(N, QVector<float>(N+1));
    //float *A = new float[N*(N+1)];
    for(int i = 0; i < N; i++) {
        qDebug() << "read:" << i;
        for(int j = 0; j < N+1; j++) {
            //f >> A[i*N+j];
            f >> A[i][j];
        }
    }

    int r;

    qDebug() << Gauss(A, N, &r);
    //qDebug() << GaussCL(A, N, NULL, &r);

    qDebug() << r;
}
