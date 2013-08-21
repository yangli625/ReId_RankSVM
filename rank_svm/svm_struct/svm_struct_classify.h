#ifndef SVM_STRUCT_CLASSIFY
#define SVM_STRUCT_CLASSIFY

#include <stdio.h>

#include "svm_light/svm_common.h"

#include "svm_struct_api.h"
#include "svm_struct/svm_struct_common.h"


int svm_struct_classify_main (int argc, char* argv[], SAMPLE &test_sample, bool read_data);

#endif