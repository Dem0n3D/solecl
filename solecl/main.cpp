#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <math.h>

#include <QVector>
#include <QDebug>

#include "util.h"
#include "gauss.h"
#include "jacobi.h"
#include "zeidel.h"

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

    float *x1 = new float[N];
    float *x2 = new float[N];

    float D;

    r1 = Zeidel(A, N, x1);
    r2 = ZeidelCL(A, N, NULL, x2);

    float max = 0;
    for(int i = 0; i < N; i++) {
        max = (max > fabs(x1[i]-x2[i])) ? max : fabs(x1[i]-x2[i]);
    }

    outX(x1, N);
    outX(x2, N);

    qDebug() << r1 << r2 << max;
}
