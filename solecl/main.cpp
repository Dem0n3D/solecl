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
#include "square.h"

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

    f.close();

    int t0, t1, t2, t3, t4;

    QVector< QVector<float> > AtA(N, QVector<float>(N+1, 0));

    QCLContext *context = new QCLContext();
    QCLBuffer buffAtA;

    if(!context->create(QCLDevice::GPU)) {
        qFatal("Could not create OpenCL context");
    }

    t0 = multTranspCL(A, AtA, N, context, &buffAtA);

    QVector<float> x1(N, 0);
    QVector<float> x2(N, 0);

    t1 = SquareCL(AtA, N, x1, context);
    t2 = GaussCL(AtA, N, x2, context);

    qDebug() << t1 << maxError(AtA, x1);
    qDebug() << t2 << maxError(AtA, x2);
}
