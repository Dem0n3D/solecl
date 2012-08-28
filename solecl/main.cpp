#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <math.h>

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

    for(int i = 0; i < N; i++) {
        qDebug() << "read:" << i;
        for(int j = 0; j < N+1; j++) {
            f >> A[i][j];
        }
    }

    int r1, r2;

    QVector<float> x1 = Gauss(A, N, &r1);
    QVector<float> x2 = GaussCL(A, N, NULL, &r2);

    float sum = 0;
    for(int i = 0; i < N; i++) {
        sum += fabs(x1[i]-x2[i]);
    }

    qDebug() << r1 << r2 << sum;
}
