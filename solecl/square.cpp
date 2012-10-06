#include "square.h"

#include "util.h"

int Square(QVector< QVector<float> > &A, int n, QVector<float> &x) {
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

    QVector<float> y(x.size(), 0);

    for(int i = 0; i < n; i++) {
        y[i] = A[i][n];
        for(int k = 0; k < i; k++) {
            y[i] -= U[k][i]*y[k];
        }
        y[i] /= U[i][i];
    }

    for(int i = n-1; i >= 0; i--) {
        x[i] = y[i];
        for(int k = i+1; k < n; k++) {
            x[i] -= U[i][k]*x[k];
        }
        x[i] /= U[i][i];
    }
}

int SquareCL(QCLBuffer buffA, int n, QVector<float> &x, QCLContext *context) {
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

    QCLKernel square_y1 = program.createKernel("square_y1");
    QCLKernel square_y2 = program.createKernel("square_y2");

    square_fwd1.setGlobalWorkSize(1);
    square_fwd2.setGlobalWorkSize(n);

    square_y1.setGlobalWorkSize(1);

    QCLVector<float> xcl = context->createVector<float>(n, QCLMemoryObject::ReadWrite);

    for(int i = 0; i < n; i++) {
        square_fwd1(buffA, n+1, i);
        square_fwd2(buffA, n+1, i);
    }

    for(int i = 0; i < n; i++) {
        square_y1(buffA, xcl, n+1, i);

        if(n-i-1 > 0) {
            square_y2.setGlobalWorkSize(n-i-1);
            square_y2.setGlobalWorkOffset(i+1, 0, 0);
            square_y2(buffA, xcl, n+1, i);
        }
    }

    xcl.read(x.data(), n);

    qDebug() << x;
}

int SquareCL( QVector< QVector<float> > &A, int n, QVector<float> &x, QCLContext *context) {
    if(!context) {
        context = new QCLContext();

        if(!context->create(QCLDevice::GPU)) {
            qFatal("Could not create OpenCL context");
        }
    }

    QCLBuffer buffA = context->createBufferDevice(n*(n+1)*sizeof(float), QCLMemoryObject::ReadWrite);
    matrix2CLBuff(A, buffA);

    return SquareCL(buffA, n, x, context);
}
