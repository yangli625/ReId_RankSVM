#include<iostream>
#include<vector>
#include<fstream>

using namespace std;

#include "svm_struct/svm_struct_main.h"
#include "svm_struct/svm_struct_classify.h"
extern "C"{
#include "mat.h"
#include "matrix.h"
}
namespace ReId_svm_lib{
#define FEATURE_DIMENSION 2592 //2784
	double GaussianVal(double, double, double) ;
	class ReIdentification{
	public:
        //int feature_dimension;
		//int feature_subset_num; //number of sub-image per image
        //int feature_num;
		double* features; //all-in-one! features->shots->person
		double* mu; //mean of each feature dimension
		double* var;//variance of each feature dimension, sigma-SQURE!
        //vector< vector<double> >* features; //features stores all the feature vector. features[i] indicate each
                                            //person, and each person maintains a vector feature vectors.
                                            //In matlab, it saves as cell{person_num,1}, in each element, it is
                                            //cell{1, person_imgs}
		double *svm_model;
        int person_num;
		int* img_per_person;
		int* accumulate_img_per_person; //integral version of img_per_person
        int* train_img_per_person;
        int* test_img_per_person;
        int total_train_img;
        int total_test_img;
        int* train_ind;
        int* test_ind;
        int* test_gallery_ind;//For testing, we randomly choose 1 image from each person as gallery set, all other
                              //images consist probe set. IN testing, each image in probe set will be used to
                              //find a matching image in gallery set.
		int* train_shots_ind;//This is only for ETHZ, each person randomly choose train_img_per_person number of shots
		int train_size;
		int test_size;
		double* test_score;
		double* test_score_disc;
		char* train_filename;
		char* train_ind_filename;
		char* test_gallery_filename;
		char model_filename[16];
		char* test_filename;
		char* prediction_filename;
		char* svm_C;

		bool read_train_ind_from_file;
		bool read_test_gallery_from_file;
		//int* rerank;
		SAMPLE train_sample;
		SAMPLE test_sample;
		int dnum;

		//boosting para
		double * D;

		ReIdentification(){
			features = NULL;
			mu = NULL;
			var = NULL;
			svm_model=NULL;
			img_per_person = NULL;
			accumulate_img_per_person = NULL;
            train_img_per_person = NULL;
            test_img_per_person = NULL;
            total_train_img = 0;
            total_test_img = 0;
			train_ind = NULL;
			test_ind = NULL;
            test_gallery_ind = NULL;
			train_shots_ind = NULL;
			test_score = NULL;
			test_score_disc = NULL;
			train_ind_filename=NULL;
			test_gallery_filename = NULL;
			train_filename = "train.dat";
			strcpy(model_filename , "model.dat");
			test_filename = "test.dat";
			prediction_filename = "predictions";
            svm_C = "1"; //0.01/0.001 for CAVIAR, 1 for VIPeR
			//rerank = NULL;
			read_train_ind_from_file=false;
			read_test_gallery_from_file = false;
			D=NULL;
			dnum=0;
		}
		
//		__declspec(dllexport) void ReadMatlabData(const char* filename, const char* variable_name);
//		__declspec(dllexport) void ReadSubImageFeatures(const char* filename, const char* variable_name);
//		__declspec(dllexport) void WriteModelToMatfile();
//		__declspec(dllexport) void GenerateTrainTestData();//(const int train_size, const int test_size);
//		__declspec(dllexport) void GenerateSample();
//		__declspec(dllexport) void TrainModel(bool read_data);
//		__declspec(dllexport) void TestModel(bool read_data);
//		__declspec(dllexport) void ModelBoosting();
//		__declspec(dllexport) void EnSemble();
		
        void ReadMatlabData(const char* filename);
        void ReadSubImageFeatures(const char* filename, const char* variable_name);
        void WriteModelToMatfile();
        void WriteModelToBinfile();
		void ReadModelFromBinfile();
        void GenerateTrainTestData();//(const int train_size, const int test_size);
        void Initialization();
        void TrainModel(bool read_data);
        void TestModel(bool read_data);
        void ModelBoosting();
        void EnSemble();
		void CalculateFeatureStats();

	private:
		//std::vector<std::vector<int>>person_ind; //indexes in feature_label for each person
		void init();
		//double* ReadModelFile();
		//void PersonIndInLabel();
        void WritePairDiffData(std::ofstream *file, int m, int n, int ref_shot=0);
		double* ReadPredictions(const int test_size);
		//int TotalImageNum(int ind[], int size);
		double* ReadPrediction(const char* file_name);
        int* AnalysePrediction(const double *score, const int prediction_size, bool train_test);
        //double* PredictionScore(SAMPLE sample, double* w, bool train_test); //0 for train, 1 for test
        void GenerateTestScore(double* w); //just for produce test sample prediction score, to save memory...
		double* AnalyseTestScore(const double* score);
        SAMPLE ConstructSample(int* index, const int size, bool train_test=0);
        //void WriteSample(int m, int n, PATTERN* x, LABEL* y, const int ref_shot = 0, bool train_test=0);
        void WriteSample(int ind, double f1[], double f2[], PATTERN* x, LABEL* y, int rank);
		void set_test_gallery_ind();
		inline int get_curr_feature_pos(const int person_ind, const int img_ind) { return (accumulate_img_per_person[person_ind]+img_ind)*FEATURE_DIMENSION; }
		inline int get_train_feature_pos(const int i, const int img_ind) {return (accumulate_img_per_person[train_ind[i]]+img_ind)*FEATURE_DIMENSION;}
		inline int get_test_feature_pos(const int i, const int img_ind) {return (accumulate_img_per_person[test_ind[i]]+img_ind)*FEATURE_DIMENSION;}
	};
}
