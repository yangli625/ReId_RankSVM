#ifndef SVM_STRUCT_MAIN
#define SVM_STRUCT_MAIN

#ifdef __cplusplus
extern "C" {
#endif
#include "../svm_light/svm_common.h"
#include "../svm_light/svm_learn.h"
#ifdef __cplusplus
}
#endif
# include "svm_struct_learn.h"
# include "svm_struct_common.h"
# include "../svm_struct_api.h"

#include <stdio.h>
#include <string.h>
#include <assert.h>

int svm_struct_main(int argc, char *argv[]);

int svm_struct_classify_main (int argc, char* argv[]);

void svm_model_initialization(SAMPLE *sample, MODEL *model){
  
  LEARN_PARM learn_parm;
  KERNEL_PARM kernel_parm;
  STRUCT_LEARN_PARM struct_parm;
  STRUCTMODEL sm;
  int totdoc = 0;
  
  struct_parm.C=-0.01;
  struct_parm.slack_norm=1;
  struct_parm.epsilon=DEFAULT_EPS;
  struct_parm.custom_argc=0;
  struct_parm.loss_function=DEFAULT_LOSS_FCT;
  struct_parm.loss_type=DEFAULT_RESCALING;
  struct_parm.newconstretrain=100;
  struct_parm.ccache_size=5;
  struct_parm.batch_size=100;

  //strcpy (modelfile, "svm_struct_model");
  strcpy (learn_parm.predfile, "trans_predictions");
  strcpy (learn_parm.alphafile, "");
  verbosity=0;/*verbosity for svm_light*/
  struct_verbosity=1; /*verbosity learning portion*/
  learn_parm.biased_hyperplane=1;
  learn_parm.remove_inconsistent=0;
  learn_parm.skip_final_opt_check=0;
  learn_parm.svm_maxqpsize=10;
  learn_parm.svm_newvarsinqp=0;
  learn_parm.svm_iter_to_shrink=-9999;
  learn_parm.maxiter=100000;
  learn_parm.kernel_cache_size=40;
  learn_parm.svm_c=99999999;  /* overridden by struct_parm->C */
  learn_parm.eps=0.001;       /* overridden by struct_parm->epsilon */
  learn_parm.transduction_posratio=-1.0;
  learn_parm.svm_costratio=1.0;
  learn_parm.svm_costratio_unlab=1.0;
  learn_parm.svm_unlabbound=1E-5;
  learn_parm.epsilon_crit=0.001;
  learn_parm.epsilon_a=1E-10;  /* changed from 1e-15 */
  learn_parm.compute_loo=0;
  learn_parm.rho=1.0;
  learn_parm.xa_depth=0;
  kernel_parm.kernel_type=0;
  kernel_parm.poly_degree=3;
  kernel_parm.rbf_gamma=1.0;
  kernel_parm.coef_lin=1;
  kernel_parm.coef_const=1;
  strcpy(kernel_parm.custom,"empty");
  
  init_struct_model(*sample, &sm, &struct_parm, &learn_parm, &kernel_parm); 
  
  
  model->supvec = (DOC **)my_malloc(sizeof(DOC *)*(totdoc+2));
  model->alpha = (double *)my_malloc(sizeof(double)*(totdoc+2));
  model->index = (long *)my_malloc(sizeof(long)*(totdoc+2));

  model->at_upper_bound=0;
  model->b=0;	       
  model->supvec[0]=0;  /* element 0 reserved and empty for now */
  model->alpha[0]=0;
  model->lin_weights=NULL;
  model->totwords=sm.sizePsi+1;
  model->totdoc=totdoc;
  model->kernel_parm=kernel_parm;
  model->sv_num=1;
  model->loo_error=-1;
  model->loo_recall=-1;
  model->loo_precision=-1;
  model->xa_error=-1;
  model->xa_recall=-1;
  model->xa_precision=-1;
  
  free_struct_model(structmodel);
}

#endif
