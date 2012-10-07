#-------------------------------------------------
#
# Project created by QtCreator 2012-08-28T00:51:34
#
#-------------------------------------------------

QT       += core

QT       -= gui

TARGET = solecl
CONFIG   += console
CONFIG   -= app_bundle

TEMPLATE = app


SOURCES += main.cpp \
    gauss.cpp \
    jacobi.cpp \
    util.cpp \
    zeidel.cpp \
    square.cpp \
    rotation.cpp

HEADERS += \
    gauss.h \
    jacobi.h \
    util.h \
    zeidel.h \
    square.h \
    rotation.h

OTHER_FILES += \
    cl/gauss.cl \
    cl/jacobi.cl \
    cl/zeidel.cl \
    cl/matrix.cl \
    cl/square.cl
