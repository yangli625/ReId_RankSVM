#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <assert.h>
//#include "SVM_model.h"
#include "svm_struct/svm_struct_main.h"
#include "svm_struct/svm_struct_classify.h"
#include "svm_struct_api_types.h"
#include "svm_struct_api.h"
#include "mat.h"
#include "matrix.h"


//#define SEMBLE
#define DOUBLE_CAM_DATA
//#define SINGLE_TEST
//#define CROSS_VALIDATION
#define TEST
#define CREATE_MODEL_FILE
#define PERSON_NUM 632
#define FEATURE_DIMENSION 2592 //768, 1824, 2592
#define TRAIN_NUM 100
#define TEST_NUM 100
#define WEAK_CLASSIFIER_NUM 5
//#define RANK 1

double** read_matlab_data(const char *file, const char *variable_name) {
  mxArray *variablePtr, *elementPtr;
  std::cout<<"Reading file"<<file<<"...\n\n";

  /* Open file to get directory*/
  MATFile *pmat = matOpen(file, "r");
  if (pmat == NULL) {
    std::cerr<<"Error opening file "<<file<<std::endl;
    return(NULL);
  }
  
  std::cout<<"Reading in variable "<<variable_name<<" ...\n";
  variablePtr = matGetVariable(pmat, variable_name);
  if (variablePtr == NULL) {
    std::cerr<<"Error can't find variable "<<variable_name<<std::endl;
  }

  std::cout<<"cell dimension: row "<<mxGetM(variablePtr)<<", col "<<mxGetN(variablePtr)<<std::endl;

  double **feature_data = new double *[mxGetM(variablePtr)*mxGetN(variablePtr)];
  
  for (int i=0; i<mxGetM(variablePtr)*mxGetN(variablePtr); i++){
      elementPtr = mxGetCell(variablePtr, i);
      if(elementPtr == NULL){
	std::cerr<<"Empty cell\n";
      }
      feature_data[i] = new double[FEATURE_DIMENSION];
      for (int j=0; j<FEATURE_DIMENSION; j++){
	feature_data[i][j] = *((double *)mxGetData(elementPtr)+j);
      }
  }
  mxDestroyArray(variablePtr);

  if (matClose(pmat) != 0) {
      std::cerr<<"Error closing file "<<file<<std::endl;
      return(NULL);
  }
  //std::cout<<"Successfully read in "<<sizeof(feature_data)<<" data.\n";
  return(feature_data);
}

void write_train_test_data(std::ofstream *file, double **feature,  int i, const int j){
  //i,j are person index from cam_a and cam_b seperatly, start from 0
  //here we can have different data schemes
  //cam_a to cam_b
  //cam_b to cam_a
  //cam_a to cam_a + cam_b *****should notice the indexing order, no self subtraction******
  //cam_b to cam_a + cam_b
  //cam_a to cam_b + cam_b to cam_a
  //consider cam_a to cam_b
  if(i==j)
    *file<<"2 qid:"<<(i+1)<<" ";
  else
    *file<<"1 qid:"<<(i+1)<<" ";
  for(int n=0; n<FEATURE_DIMENSION; n++){
    double abs_feature = fabs(feature[i][n]-feature[j+PERSON_NUM][n]);
    if(abs_feature!=0){
      *file<<(n+1)<<":"<<abs_feature<<" ";
    }
  }
  *file<<"# cam_a "<<i+1<<" cam_b  "<<j+1<<std::endl;
  
#ifdef DOUBLE_CAM_DATA
  //consider cam_b to cam_a
  if(i==j)
    return;
  
  *file<<"1 qid:"<<(i+1)<<" ";
  for(int n=0; n<FEATURE_DIMENSION; n++){
    double abs_feature = fabs(feature[i][n]-feature[j][n]);
    if(abs_feature!=0){
      *file<<(n+1)<<":"<<abs_feature<<" ";
    }
  }
  *file<<"# cam_b "<<i+1<<" cam_a  "<<j+1<<std::endl;
#endif
}

void generate_train_test_data(double **feature, int *train_ind, const int train_size,
  int *test_ind, const int test_size){  

  std::ofstream *train_file = new std::ofstream("train.dat");
  std::ofstream *test_file = new std::ofstream("test.dat");

  int train_count = 0;// record the position in train_ind
  int test_count = 0;
  for(int i=0; i<PERSON_NUM; i++){
    if(i==train_ind[train_count]){
      for(int j=0; j<train_size; j++){
	write_train_test_data(train_file, feature, i, train_ind[j]);
      }
      train_count++;
    }else if(i==test_ind[test_count]){
      for(int j=0; j<test_size; j++){
	write_train_test_data(test_file, feature, i,test_ind[j]);
      }
      test_count++;
    }
    if(i*100/PERSON_NUM%10==0){
      std::cout<<"...";
    }
  }
  train_file->close();
  test_file->close();
  delete train_file;
  delete test_file;
}

double* read_data_from_file(const char* file_name, const int line_num){
  std::ifstream *file = new std::ifstream(file_name);
  double *score = new double[line_num];
  int ind = 0;
  while(!file->eof()){
    *file>>score[ind++];
  }
  assert(ind!=line_num);
  file->close();
  delete file;
  return score;
}

double* read_test_result(const char* file_name, const int test_num){
  int observation_per_person;
#ifdef DOUBLE_CAM_DATA
  observation_per_person= 2*test_num-1;
#else
  observation_per_person = test_num;
#endif
  return read_data_from_file(file_name, observation_per_person*test_num);
}

double* analyse_test_result(const double *score, const int test_num, const int rank_boundary[], const int rank_num){  
  int observation_per_person;
#ifdef DOUBLE_CAM_DATA
  observation_per_person= 2*test_num-1;
#else
  observation_per_person = test_num;
#endif
  int ranking = 1;
  int *correct_classification = new int[rank_num];
  double *percentage = new double[rank_num];
  std::fill(correct_classification, correct_classification+rank_num, 0);
  //std::fill(percentage, percentage+rank_num, 0);
  
  double highest_score;
  for(int i=0; i<test_num; i++){
    ranking = 1;
#ifdef DOUBLE_CAM_DATA
    highest_score = score[i*observation_per_person+2*i];
#else
    highest_score = score[i*observation_per_person+i];
#endif
    for(int j=0; j<observation_per_person; j++){ //total number of scores per person
      if (highest_score<score[i*observation_per_person+j]){
	//std::cout<<"person "<<i<<" original score "<<highest_score<<" new score "<<score[i*observation_per_case+j]<<std::endl;
	ranking++;
      }
    }
    for(int n=0; n<rank_num; n++){     
      if(ranking<=rank_boundary[n])
	correct_classification[n]++;
    }
  }
  for (int n=0; n<rank_num; n++){
    percentage[n] = (double)correct_classification[n]*100/(double)test_num;
    std::cout<<"The Matching Rate is "<<percentage[n]<<"%, within rank "<<rank_boundary[n]<<std::endl;
  }
  delete [] correct_classification;
  return (percentage);
}

char* cross_validation(const int *train_ind, const int train_size, double **feature){
  const int C_num = 7;
  char *C_val[] = {"0.01","0.1","1","10","100", "1000", "10000"};//if change the trial number, pay attention to data type
  const int group_num = 5;
  const int cross_test_size = train_size/group_num;
  const int cross_train_size = train_size-cross_test_size;
  int *cross_test_ind = new int[cross_test_size];
  int *cross_train_ind = new int [cross_train_size];
  int rank[] = {1,10,20,50};
  int rank_num = sizeof(rank)/sizeof(int);
  double **C_score = new double*[rank_num];
  
  
  for (int m=0; m<rank_num; m++){
	  C_score[m] = new double[C_num];
	  for (int n=0; n<C_num; n++){
		  C_score[m][n]=0;
	  }
  }
    
  for(int i=0; i<group_num; i++){
    
    //assign test indexes and train indexes for cross validation
    int count=0; //mark position in cross_train_ind
    for(int j=0; j<train_size; j++){
      if(j>=i*cross_test_size && j<(i+1)*cross_test_size)
	cross_test_ind[j-count]=train_ind[j];
      else
	cross_train_ind[count++]=train_ind[j];
    }
    for(int n=0; n<cross_test_size; n++)
      std::cout<<cross_test_ind[n]<<" ";
    std::cout<<std::endl;
    for(int n=0; n<cross_train_size; n++)
      std::cout<<cross_train_ind[n]<<" ";
    std::cout<<std::endl;
    
    std::cout<<"Generating trainning & testing data";
    generate_train_test_data(feature, cross_train_ind, cross_train_size, cross_test_ind, cross_test_size);
    std::cout<<"done!\n";
    
	delete [] cross_test_ind;
	delete [] cross_train_ind;

    //start cross validation for each C value
    //bool read = 1;
    for(int n=0; n<C_num; n++){
      std::cout<<"C = "<<C_val[n]<<", trail "<<i<<"...\n";
      
      char *svm_argv[5] = {"svm_rank_learn", "-c", C_val[n], "train.dat", "model.dat"};
      svm_struct_main(5, svm_argv);
      
      char *svm_classify_argv[4] = {"svm_rank_classify", "test.dat", "model.dat", "predictions"};
      svm_struct_classify_main(4, svm_classify_argv);
      /*
      init_learn_model();
      if(read)
	read_train_file("train.dat");
      svm_learn("model.dat", C_val[n]);
      
      init_classify_model();
      read_model_file("model.dat");
      if(read)
	read_test_file("test.dat");
      svm_classify("predictions");
      free_model_file();
      
      read=0;
      */
      double *score = read_test_result("predictions", cross_test_size);
      double *classification_percentage = analyse_test_result(score, cross_test_size, rank, rank_num);
      delete [] score;
      for(int m=0; m<rank_num; m++){
		C_score[m][n]+=classification_percentage[m];
      }
      
	  delete [] classification_percentage;

    }
    //free_train_file();
    //free_test_file();

  }
  double max_C_score = 0;
  int best_C_ind;
  double total_C_score[C_num];
  std::fill(total_C_score, total_C_score+C_num, 0);
  
  for (int m=0; m<rank_num; m++){
    std::cout<<"At rank "<<rank[m]<<", the average classification rate are: "<<std::endl;
    for (int n=0; n<C_num; n++){
      std::cout<<"When C is "<<C_val[n]<<", "<<C_score[m][n]/5<<"%"<<std::endl;
      total_C_score[n] += C_score[m][n];
    }
  }
  for (int n=0; n<C_num; n++){
    if(total_C_score[n]>=max_C_score){
      max_C_score = total_C_score[n];
      best_C_ind = n;
    }
  }

  for(int i =0; i<rank_num; i++){
	  delete [] C_score[i];
  }
  delete [] C_score;

  return (C_val[best_C_ind]);
    
}

int *generate_rand_ind(const int total_num, const int required_num, bool* mask){
  //this function is used to generate required number of random indexes, from 0 to total_num
  int *ind = new int[required_num];
  srand(time(NULL));
  int count = 0;
  int temp;
  while(count < required_num){
    temp = rand()%total_num;
    if(!mask[temp]){
      mask[temp]=1;
      ind[count++]=temp;
    }
  }
  std::sort(ind, ind+required_num);
  return(ind);
}


void reIdnetification(double **feature){
  int test_id;
  std::cout<<"Please choice one person: ";
  std::cin>>test_id;
  if(test_id<1||test_id>PERSON_NUM){
    std::cerr<<"Invalid input! Please choose from 1 to "<<PERSON_NUM<<std::endl;
    return;
  }
  //generate test file name;
  char file_name[16];
  sprintf(file_name, "%d_test.dat", test_id);
  //generate data for testing person
  int candidate_size = PERSON_NUM; //here we test against all the available person
  std::ofstream * file = new std::ofstream (file_name);
  for(int i=0; i<candidate_size; i++){
    write_train_test_data(file, feature, test_id-1, i); //VERY IMPORTANT TO -1   !!!
  }
  std::cout<<"Done generating test data file for person "<<test_id<<std::endl;
  file->close();  std::cout<<"DONE reading testing result!"<<std::endl;
  delete file;
  
  //classification in the model
  char *svm_classify_argv[4] = {"svm_rank_classify", file_name, "model.dat", "predictions"};
  svm_struct_classify_main(4, svm_classify_argv);
  
  double *score = read_data_from_file("predictions",candidate_size);
  int ranking = 1;
  double highest_score = score[test_id-1];
  for(int i=0; i<candidate_size; i++){
    if(score[i]>highest_score){
      ranking++;
      std::cout<<"Person "<<i+1<<" score "<<score[i]<<std::endl;
    }
  }
  std::cout<<"Person "<<test_id<<" is identified in rank "<<ranking<<", score "<<score[test_id-1]<<std::endl;
  delete [] score;
  
}

double calculate_ranker_weighting_score(const double* D, const double *difference, const int size){
  double ranker_score=0;
  for(int n=0; n<size; n++){
    if(difference[n]>=0)
      ranker_score+=D[n];
  }
  return ranker_score;
}

double calculate_r(const double* D,  const double *difference, const int size){
  double r=0;
  double max_diff=0;
  for(int n=0; n<size; n++){
    r += D[n]*(-difference[n]);
    if(abs(difference[n])>max_diff)
      max_diff = abs(difference[n]);
  }
  return r/max_diff;
}

double* update_D(double *D, const int size, const double alpha, const double *difference){
  double *new_D = new double[size];
  double sum=0;
  for (int n=0; n<size; n++){
    new_D[n] = D[n]*exp(alpha*difference[n]);
    sum += new_D[n];
  }
  for (int n=0; n<size; n++){
    new_D[n] = new_D[n]/sum;
  }
  delete [] D;
  return new_D;
}

double *calculate_pairwise_difference(const int observation_per_person, const int test_num, const int classifier_num){
  double *pairwise_difference = new double[test_num*(observation_per_person-1)*classifier_num];
  char predictions_file[200];
  int count = 0;
  for(int n=0; n<classifier_num; n++){
    sprintf(predictions_file, "predictions_%d",n);
    double *score = read_test_result(predictions_file, test_num);
    int positive_observation_ind;
    for(int i=0; i<test_num; i++){
#ifdef DOUBLE_CAM_DATA
      positive_observation_ind = i*observation_per_person+2*i;
#else
      positive_observation_ind = i*observation_per_person+i;
#endif
      for(int j=0; j<observation_per_person; j++){
#ifdef DOUBLE_CAM_DATA
	if(2*i!=j)
#else
	if(i!=j)
#endif
	  //negtive - positive
	  pairwise_difference[count++]=score[i*observation_per_person+j]-score[positive_observation_ind];
      }
    }
    delete [] score;
  }
  assert(count==test_num*(observation_per_person-1)*classifier_num);
  return pairwise_difference;
}

void ensemble(double** feature){
  const int classifier_num = WEAK_CLASSIFIER_NUM;
  const int subset_size = TRAIN_NUM;
  int **train_ind = new int *[classifier_num];
  int *test_ind;
  bool *ind_mask = new bool[PERSON_NUM];
  std::fill(ind_mask, ind_mask+PERSON_NUM, 0);
  char subset_data_file[200];
  char model_file[200];
  char predictions_file[200];
  /* train weak classifiers */
  
  for (int i=0; i<classifier_num; i++){
    train_ind[i] = generate_rand_ind(PERSON_NUM, subset_size, ind_mask);
    
    std::cout<<"Group "<<i<<" training indexes: "<<std::endl;
    for(int n=0; n<subset_size; n++)
      std::cout<<train_ind[i][n]<<" ";
    std::cout<<std::endl;
    
    std::cout<<"Generating training data for group "<<i;
    sprintf(subset_data_file, "train_%d.dat", i);
    std::ofstream *train_file = new std::ofstream(subset_data_file);
    for(int m=0; m<subset_size; m++){
      for(int n=0; n<subset_size; n++){
	write_train_test_data(train_file, feature, train_ind[i][m], train_ind[i][n]);
      }
    }
    train_file->close();
    delete train_file;
    std::cout<<"... Done!"<<std::endl;
    
    sprintf(model_file, "model_%d.dat", i);
    char *svm_argv[5] = {"svm_rank_learn", "-c", "1", subset_data_file, model_file};
    svm_struct_main(5, svm_argv);
  }
  
  //generate test data file
  test_ind = generate_rand_ind(PERSON_NUM, TEST_NUM, ind_mask);
  std::cout<<"testing indexes: "<<std::endl;
  for(int n=0; n<TEST_NUM; n++)
    std::cout<<test_ind[n]<<" ";
  std::cout<<std::endl;
  
  std::cout<<"Generating testing data ";
  std::ofstream *test_file = new std::ofstream("test.dat");
  for(int i=0; i<TEST_NUM; i++){
    for(int j=0; j<TEST_NUM; j++){
      write_train_test_data(test_file, feature, test_ind[i], test_ind[j]);
    }
  }
  test_file->close();
  delete test_file;
  std::cout<<"... Done!"<<std::endl;
  
  delete [] ind_mask;
  delete [] train_ind;
  
  int observation_per_person;
#ifdef DOUBLE_CAM_DATA
  observation_per_person = 2*subset_size-1; 
#else
  observation_per_person = subset_size;
#endif
  int N = (observation_per_person-1)*subset_size*classifier_num;//total number of observations, NOTICE -1
  double *D = new double[N];//weighting parameter for each observcalculate_ranker_weighting_scoreation
  double **pairwise_difference = new double *[classifier_num];
  double ranker_weighting_score[classifier_num];
  std::fill(D, D+N, (double)1/N); // initialization
  const int T = classifier_num; // number of iterations
  int ranker_ind[T];
  double alpha[T];

  std::ofstream *diff = new std::ofstream("diff");
  
  for (int i=0; i<classifier_num; i++){
    sprintf(model_file, "model_%d.dat",i);
    ranker_weighting_score[i] = 0;
    for(int j =0; j<classifier_num; j++){
      sprintf(predictions_file, "predictions_%d", j);
      sprintf(subset_data_file, "train_%d.dat",j);
      char *svm_classify_argv[4] = {"svm_rank_classify", subset_data_file, model_file, predictions_file};
      svm_struct_classify_main(4, svm_classify_argv);

    }
    std::cout<<"calculate pairwise difference...";
    pairwise_difference[i] = calculate_pairwise_difference(observation_per_person, subset_size, classifier_num);
    std::cout<<"Done."<<std::endl;
    
    for(int n=0; n<N; n++)
      *diff<<pairwise_difference[i][n]<<'\n';
  }
  diff->close();
  delete diff;
  
  double alpha_sum=0;
  for (int t=0; t<T; t++){
    std::cout<<"boosting "<<t<<" iteration"<<std::endl;
    double min_rankers_score=1; //sum of D is 1
    for (int i=0; i<classifier_num; i++){
      ranker_weighting_score[i] = calculate_ranker_weighting_score(D, pairwise_difference[i], N);
      std::cout<<"weighting score for classifier "<<i<<" is "<<ranker_weighting_score[i]<<std::endl;
      if(ranker_weighting_score[i]<min_rankers_score){
	min_rankers_score = ranker_weighting_score[i];
	ranker_ind[t] = i;
      }
    }
    
    std::cout<<"chosen classifier is "<<ranker_ind[t]<<std::endl;
    
    //calculate r
    std::cout<<"calculate r..."<<std::endl;
    double r = calculate_r(D, pairwise_difference[ranker_ind[t]], N);
    std::cout<<" = "<<r<<std::endl;
    std::cout<<"calculate alpha...";
    alpha[t] = 0.5*log((1+r)/(1-r));
    alpha_sum += alpha[t];
    std::cout<<" = "<<alpha[t]<<std::endl;
    std::cout<<"update D..."<<std::endl;
    D = update_D(D, N, alpha[t], pairwise_difference[ranker_ind[t]]);
    
  }
  for (int i =0; i<classifier_num; i++){
	  delete [] pairwise_difference[i];
  }
  delete [] pairwise_difference;
  delete [] D;
  
  //normalize alpha
  for (int t=0; t<T; t++)
    alpha[t] /= alpha_sum;
  // form ensemble classifier
  MODEL *model[classifier_num]; 
  STRUCTMODEL model_ensemble;
  double *ensemble_weight = new double [FEATURE_DIMENSION];
  double ensemble_b=0;
  STRUCT_LEARN_PARM sparm;
  SAMPLE testsample;
  double *predictions_score = new double [observation_per_person*subset_size];
  
  model_ensemble.svm_model=(MODEL *)my_malloc(sizeof(MODEL));
  
  std::fill(ensemble_weight, ensemble_weight+FEATURE_DIMENSION, (double)0);
  for (int i=0; i<classifier_num; i++){
    sparm.custom_argc=0;
    parse_struct_parameters_classify(&sparm);
    
    sprintf(model_file, "model_%d.dat", i);
    std::cout<<"read svm model "<<model_file<<"...";
    model[i]=read_model(model_file);
    add_weight_vector_to_linear_model(model[i]);
    std::cout<<"done."<<std::endl;
    
    for (int j=0; j<FEATURE_DIMENSION; j++){
      ensemble_weight[j] += alpha[i]*model[i]->lin_weights[j];
    }
    ensemble_b += alpha[i]*model[i]->b;
    
    free_model(model[i],1);
  }
  
  sparm.loss_function=FRACSWAPPEDPAIRS;
  
  std::cout<<"New classification..."<<std::endl;
  testsample=read_struct_examples("test.dat",&sparm);
  std::cout<<"Done reading testing data"<<std::endl;
  
  svm_model_initialization(&testsample, &model_ensemble);
  model_ensemble.svm_model->lin_weights = ensemble_weight;
  //model_ensemble.w=ensemble_weight;
  model_ensemble.svm_model->b=ensemble_b;
  
  
  
  int count =0;
  for (int i=0; i<testsample.n; i++){
    for (int j=0; j<testsample.examples[i].x.totdoc; j++)
      predictions_score[count++]=classify_example(model_ensemble.svm_model, testsample.examples[i].x.doc[j]);
  }
  //assert(count==observation_per_person*subset_size);
  assert(count==(2*TEST_NUM-1)*TEST_NUM);
  int rank[] = {1,10,20,50};
  analyse_test_result(predictions_score,TEST_NUM, rank, sizeof(rank)/sizeof(int));
  
  free_struct_sample(testsample);
  free_struct_model(model_ensemble);
  delete [] ensemble_weight;
  delete [] predictions_score;
}

double* read_model_file(char* filename){
  MODEL *model;
  double* model_return = new double[FEATURE_DIMENSION+1];//another one for b for SVM model!!!
  model = read_model(filename);
  add_weight_vector_to_linear_model(model);
  for(int i=0; i<FEATURE_DIMENSION; i++){
    model_return[i] = model->lin_weights[i+1]; //linear weights start from 1.... wtf...
  }
  model_return[FEATURE_DIMENSION]=model->b;
  free_model(model,1);
  return model_return;
}

void write_model_to_matfile(double model[], double b){
  MATFile *pmat;
  mxArray *pa;
  mxArray *pd;
  int status;
  char* file = "model.mat";
  pmat = matOpen(file, "w");
  if (pmat == NULL) {
    printf("Error creating file %s\n", file);
    printf("(Do you have write permission in this directory?)\n");
    return;
  }
  pa = mxCreateDoubleMatrix(FEATURE_DIMENSION, 1, mxREAL);
  if (pa == NULL) {
      printf("%s : Out of memory on line %d\n", __FILE__, __LINE__); 
      printf("Unable to create mxArray.\n");
      return;
  }
  memcpy((void *)(mxGetPr(pa)), (void *)model, sizeof(double)*FEATURE_DIMENSION);
  status = matPutVariable(pmat, "svm_weights", pa);
  if (status != 0) {
      printf("%s :  Error using matPutVariable on line %d\n", __FILE__, __LINE__);
      return;
  }
  pd = mxCreateDoubleScalar(b);
  memcpy((void *)(mxGetPr(pd)), (void *)&b, sizeof(b));
  status = matPutVariable(pmat, "svm_b", pd);
  
  mxDestroyArray(pa);
  mxDestroyArray(pd);
  if (matClose(pmat) != 0) {
    printf("Error closing file %s\n",file);
    return;
  }
}

int main(int argc, char **argv) {

  double **feature = read_matlab_data(argv[1], argv[2]);
#ifdef SEMBLE
  ensemble(feature);
#else
  
#ifndef SINGLE_TEST
  
  int train_size = TRAIN_NUM;
  int test_size = TEST_NUM;
  
  int *train_ind;
  int *test_ind;
  bool *ind_mask = new bool[PERSON_NUM];
  std::fill(ind_mask, ind_mask+PERSON_NUM, 0);
  train_ind = generate_rand_ind(PERSON_NUM, train_size, ind_mask);
  test_ind = generate_rand_ind(PERSON_NUM, test_size, ind_mask);
  delete [] ind_mask;
  
  std::cout<<"training indexes: "<<std::endl;
  for(int n=0; n<train_size; n++)
    std::cout<<train_ind[n]<<" ";
  std::cout<<std::endl;
  std::cout<<"testing indexes: "<<std::endl;
  for(int n=0; n<test_size; n++)
    std::cout<<test_ind[n]<<" ";
  std::cout<<std::endl;
  
  std::cout<<"DONE generating training and testing indexes!"<<std::endl;
  

  
  
  /* ****************
   * Cross Validation
   * ****************/
#ifdef CROSS_VALIDATION  
  char *C = cross_validation(train_ind, train_size, feature);
  std::cout<<"The choice of C is "<<C<<std::endl;
#endif

#ifdef TEST
  std::cout<<"calculating trainning data: ";
  generate_train_test_data(feature, train_ind,train_size, test_ind, test_size);
  std::cout<<std::endl;
  
  char *svm_argv[5] = {"svm_rank_learn", "-c", "1", "train.dat", "model.dat"};
  svm_struct_main(5, svm_argv);
  
#ifdef CREATE_MODEL_FILE
  double* model = read_model_file("model.dat");
  write_model_to_matfile(model, model[FEATURE_DIMENSION]);
  //return 0;
#endif
  
  char *svm_classify_argv[4] = {"svm_rank_classify", "test.dat", "model.dat", "predictions"};
  svm_struct_classify_main(4, svm_classify_argv);

  
  double *score = read_test_result("predictions",test_size);
  std::cout<<"DONE reading testing result!"<<std::endl;
  int rank[] = {1,10,20,50};
  analyse_test_result(score,test_size, rank, sizeof(rank)/sizeof(int));
 
  delete [] score;
#endif
  delete [] train_ind;
  delete [] test_ind;

#else
  reIdnetification(feature);
#endif
#endif
  for(int i=0; i<PERSON_NUM*2; i++){
    delete [] feature[i];
  }
  delete [] feature;
  
  return 0;
}