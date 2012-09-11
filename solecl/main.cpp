#include <fstream>
#include <iostream>
#include <iomanip>
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
    cout << "Enter N: ";
    cin >> N;

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

    QCLContext *context = new QCLContext();
    if(!context->create(QCLDevice::GPU)) {
        qFatal("Could not create OpenCL context");
    }

    int maxUnits;
    clGetDeviceInfo(context->defaultDevice().deviceId(), CL_DEVICE_MAX_COMPUTE_UNITS, 4, &maxUnits, NULL);

    int t0, t1, t2, t3, t4;

    QVector<float> x1(N, 0);
    QVector<float> x2(N, 0);

    //QVector< QVector<float> > AtA(N, QVector<float>(N+1, 0));
    //QCLBuffer buffAtA;
    //t0 = multTranspCL(A, AtA, N, context, &buffAtA);
    //t1 = Zeidel(AtA, N, x1, 0.1);

    QCLBuffer buffA = context->createBufferDevice(N*(N+1)*sizeof(float), QCLMemoryObject::ReadWrite);

    QCLVector<float> xcl = context->createVector<float>(N, QCLMemoryObject::ReadWrite);

    stringstream fname2;
    fname2 << "result/" << N << ".txt";
    ofstream res;
    qDebug() << fname2.str().c_str();
    res.open(fname2.str().c_str(), ios_base::app);
    res << "+==================================================+\n\n";
    res << setprecision(4);

    int test_count = (N > 1000) ? 1 : 5;

    int avg, i1;

    int maxSize = context->defaultDevice().maximumWorkItemsPerGroup();

    for(int i = (N > 4000) ? 32 : 0; i <= (N+1)/3 && i <= maxSize; i += 32) {
        avg = 0;
        i1 = (i == 0) ? 1 : i;
        res << N << ' ' << i1 << ' ' << (N+1-((N+1)/i1)*i1) << ' ';

        if(N > 1000 && i > 64)
            test_count = 2;

        for(int j = 0; j < test_count; j++) {
            matrix2CLBuff(A, buffA);
            t2 = GaussCL(buffA, N, xcl, i1, context);
            /*xcl.read(&x2[0], N);

            float max = 0;
            for(int k = 0; k < N; k++) {
                max = (max > fabs(x1[k]-x2[k])) ? max : fabs(x1[k]-x2[k]);
            }*/

            avg += t2;

            qDebug() << i1 << j << t2 << -1;

            res << t2 << ' ';
        }

        res << endl;
    }

    res << endl;
    res.close();
}
