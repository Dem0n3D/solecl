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

    int t0, t1, t2, t3, t4;

    float *C = new float[N*(N+1)];

    QCLContext *context = new QCLContext();
    QCLBuffer buffC;

    if(!context->create(QCLDevice::GPU)) {
        qFatal("Could not create OpenCL context");
    }

    t0 = multTransp(A, C, N, context, &buffC);

    float *x1 = new float[N];
    float *x2 = new float[N];

    float D;

    t1 = Zeidel(A, N, x1, 0.1, &t4);
    t2 = ZeidelCL2(buffC, N, context, x2, 0.1);
    t3 = GaussCL(buffC, N, context, x2);

    float max = 0;
    for(int i = 0; i < N; i++) {
        max = (max > fabs(x1[i]-x2[i])) ? max : fabs(x1[i]-x2[i]);
    }

    outX(x1, N);
    outX(x2, N);

    qDebug() << t1+t4 << t2 << t3 << t0 << t1 << t4 << max;
}
