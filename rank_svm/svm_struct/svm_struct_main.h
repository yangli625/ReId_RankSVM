#ifndef SVM_STRUCT_MAIN
#define SVM_STRUCT_MAIN


#include "svm_light/svm_common.h"
#include "svm_light/svm_learn.h"

# include "svm_struct/svm_struct_learn.h"
# include "svm_struct/svm_struct_common.h"
# include "svm_struct_api.h"

#include <stdio.h>
#include <string.h>
#include <assert.h>

int svm_struct_main (int argc, char* argv[], SAMPLE &train_sample, bool read_data);

void svm_model_initialization(SAMPLE *sample, STRUCTMODEL *model);

#endif