#include "square.h"

#include "util.h"

int Square(QVector< QVector<float> > &A, int n, float *x) {
    QVector< QVector<float> > U(n, QVector<float>(n, 0));
    for(int i = 0; i < n; i++) {
        float s = A[i][i];
        for(int k = 0; k < i; k++) {
            s -= U[k][i]*U[k][i];
        }
        U[i][i] = sqrt(s);

        for(int j = i+1; j < n; j++) {
            s = A[i][j];
            for(int k = 0; k < i; k++) {
                s -= U[k][i]*U[k][j];
            }
            U[i][j] = s / U[i][i];
        }
    }

    A = U;
}

int SquareCL(QCLBuffer buffA, int n, float *x, QCLContext *context) {
    if(!context) {
        context = new QCLContext();

        if(!context->create(QCLDevice::GPU)) {
            qFatal("Could not create OpenCL context");
        }
    }

    QCLProgram program;

    program = context->buildProgramFromSourceFile(QLatin1String("cl/square.cl"));

    QCLKernel square_fwd1 = program.createKernel("square_fwd1");
    QCLKernel square_fwd2 = program.createKernel("square_fwd2");

    square_fwd1.setGlobalWorkSize(1);
    square_fwd2.setGlobalWorkSize(n);

    QCLVector<float> xcl = context->createVector<float>(n, QCLMemoryObject::ReadWrite);

    for(int i = 0; i < n; i++) {
        square_fwd1(buffA, n+1, i);
        square_fwd2(buffA, n+1, i);
    }
}

int SquareCL( QVector< QVector<float> > &A, int n, float *x, QCLContext *context) {
    if(!context) {
        context = new QCLContext();

        if(!context->create(QCLDevice::GPU)) {
            qFatal("Could not create OpenCL context");
        }
    }

    QCLBuffer buffA = context->createBufferDevice(n*(n+1)*sizeof(float), QCLMemoryObject::ReadWrite);
    matrix2CLBuff(A, buffA);

    SquareCL(buffA, n, x, context);
    CLBuff2matrix(buffA, A);
    return 0;
}
