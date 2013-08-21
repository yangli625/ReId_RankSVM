#include <string.h>
#include "ReId_svm_lib.h"

using namespace ReId_svm_lib;

int main(int argc, char* argv[]){
	ReIdentification x;

    char* feature_file = "ETHZ_features.mat";//argv[1];
	if(argc>2){
		x.read_train_ind_from_file = true;
		x.train_ind_filename = argv[2];
		std::cout<<"train indexes file name:"<<x.train_ind_filename<<std::endl;
	}
	if(argc>3){
		x.read_test_gallery_from_file = true;
		x.test_gallery_filename = argv[3];
		cout<<"test gallery file name: "<< x.test_gallery_filename<<endl;
	}
	
	/*
	x.D = new double[x.train_size*(x.train_size*x.img_per_person-1)];
	std::fill(x.D, x.D+x.train_size*(x.train_size*x.img_per_person-1), 1/(double)(x.train_size*(x.train_size-1)*x.img_per_person));
	x.boosting_svm_model = new double[x.feature_dimension+1];
	std::fill(x.boosting_svm_model, x.boosting_svm_model+x.feature_dimension+1, 0);
	*/
	x.ReadMatlabData(feature_file);
	x.train_size = 50;
	x.test_size = x.person_num - x.train_size;
	//std::cout<<"train size is "<<x.train_size<<std::endl;
	//x.ReadSubImageFeatures("VIPeR_features_simple_subimg.mat","features");


	x.Initialization();
	x.CalculateFeatureStats();
	x.TrainModel(0);
	//x.WriteModelToMatfile();
    x.WriteModelToBinfile();
	x.ReadModelFromBinfile();
    x.TestModel(0);
	
	//x.EnSemble();
	//x.TestModel(0);
	/*
	for(int i=0; i<10; i++){		
		x.ModelBoosting();
		x.TrainModel(false);
		//x.TestModel(false);
	}
	*/

	return 0;
}
