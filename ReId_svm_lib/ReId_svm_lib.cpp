#include<iostream>
#include<string>
#include<fstream>
#include<algorithm>
#include<functional>
#include<utility>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/stat.h>

#include "ReId_svm_lib.h"

#define PI acos((double)-1)
#define ETHZ
//#define PRID
//#define DISCRIMINATIVE
#define SHOT_NUM 3

namespace ReId_svm_lib{
	//every MATLAB data reading goes column first, that is first dimension first
	void ReIdentification::ReadMatlabData(const char* filename) {
		const char *variableName1 = "FeatureSet";
		const char *variableName2 = "ImagesNumPerPerson";
        mxArray *variablePtr, *elementPtr, *personImgsPtr;
		std::cout<<"Reading file"<<filename<<"...\n\n";

		/* Open file to get directory*/
        MATFile *pmat = matOpen(filename, "r");
		if (pmat == NULL) {
            std::cerr<<"Error opening file "<<filename<<std::endl;
			return;
		}

		std::cout<<"Reading in variable "<<variableName2<<" ...\n";
		variablePtr = matGetVariable(pmat, variableName2);
		if (variablePtr == NULL) {
			std::cerr<<"Error can't find variable "<<variableName2<<std::endl;
		}

		person_num = mxGetM(variablePtr)*mxGetN(variablePtr);
		img_per_person = new int[person_num];
		accumulate_img_per_person = new int[person_num];
		int total_img_num = 0;
		for(int i=0; i<person_num; i++){
			int personImgsNum = (int)*((double *)mxGetData(variablePtr)+i);
			img_per_person[i] = personImgsNum;
			accumulate_img_per_person[i] = total_img_num;
			total_img_num += personImgsNum;
		}
		mxDestroyArray(variablePtr);
		cout<<"total image num "<<total_img_num<<endl;
		std::cout<<"Reading in variable "<<variableName1<<" ...\n";
		variablePtr = matGetVariable(pmat, variableName1);
		if (variablePtr == NULL) {
			std::cerr<<"Error can't find variable "<<variableName1<<std::endl;
		}
		features = new double[total_img_num*FEATURE_DIMENSION];
		for(int i=0; i<total_img_num*FEATURE_DIMENSION; i++){
			features[i] = *((double *)mxGetData(variablePtr)+i);
		}
		mxDestroyArray(variablePtr);

        /*
		feature_num = mxGetM(variablePtr)*mxGetN(variablePtr);
		features = new double *[mxGetM(variablePtr)*mxGetN(variablePtr)];

		for (int i=0; i<mxGetM(variablePtr)*mxGetN(variablePtr); i++){
			elementPtr = mxGetCell(variablePtr, i);
			if(elementPtr == NULL){
				std::cerr<<"Empty cell\n";
			}
			features[i] = new double [feature_dimension];		
			for (int j=0; j<feature_dimension; j++){
				features[i][j] = *((double *)mxGetData(elementPtr)+j);
			}
			
		}
		mxDestroyArray(variablePtr);
        */
		/*
        person_num = mxGetM(variablePtr);
        features = new vector< vector<double> >[person_num];
        for(int i=0; i<person_num; i++){
            personImgsPtr =mxGetCell(variablePtr,i);
            if(personImgsPtr==NULL){
                cerr<<"No images for person "<<i<<endl;
            }
            int personImgsNum = mxGetM(personImgsPtr)*mxGetN(personImgsPtr);
            cout<<"Read in person "<<i<<" with "<<personImgsNum<<" images."<<endl;
            for(int j=0; j<personImgsNum; j++){
                elementPtr = mxGetCell(personImgsPtr,j);
                if(elementPtr == NULL){
                    cerr<<"person "<<i<<" image "<<j<<" cannot be read in"<<endl;
                }
                vector<double> img_feature;
                for(int k=0; k<FEATURE_DIMENTION; k++){
                    img_feature.push_back( *((double *)mxGetData(elementPtr)+k));
                }
                features[i].push_back(img_feature);
                //mxDestroyArray(elementPtr);
            }
            mxDestroyArray(personImgsPtr);
        }
		*/
		/*
		person_num = mxGetM(variablePtr);
		img_per_person = new int[person_num];
		accumulate_img_per_person = new int[person_num];
		int total_img_num = 0;
		for(int i=0; i<person_num; i++){
			personImgsPtr =mxGetCell(variablePtr,i);
			if(personImgsPtr==NULL){
				cerr<<"No images for person "<<i<<endl;
			}
			int personImgsNum = mxGetM(personImgsPtr)*mxGetN(personImgsPtr);
			//mxDestroyArray(personImgsPtr);
			total_img_num += personImgsNum;
			img_per_person[i] = personImgsNum;
			accumulate_img_per_person[i] = total_img_num - personImgsNum;
		}
		features = new double  [total_img_num*FEATURE_DIMENSION];
		for(int i=0; i<person_num; i++){
			personImgsPtr =mxGetCell(variablePtr,i);
			if(personImgsPtr==NULL){
				cerr<<"No images for person "<<i<<endl;
			}
			//cout<<"Read in person "<<i<<" with "<<img_per_person[i]<<" images."<<endl;			
			for(int j=0; j<img_per_person[i]; j++){
				elementPtr = mxGetCell(personImgsPtr,j);
				if(elementPtr == NULL){
					cerr<<"person "<<i<<" image "<<j<<" cannot be read in"<<endl;
				}
				for(int k=0; k<FEATURE_DIMENSION; k++){
					features[get_curr_feature_pos(i,j)+k] = *((double *)mxGetData(elementPtr)+k);
				}
			}
			mxDestroyArray(personImgsPtr);
		}
		*/
		if (matClose(pmat) != 0) {
			std::cerr<<"Error closing file "<<filename<<std::endl;
			return;
		}

	}
    /*
	void ReIdentification::ReadSubImageFeatures(const char* filename, const char* variable_name) {
		mxArray *variablePtr, *elementPtr;
		std::cout<<"Reading file"<<filename<<"...\n\n";

        // Open file to get directory
		MATFile *pmat = matOpen(filename, "r");
		if (pmat == NULL) {
			std::cerr<<"Error opening file "<<filename<<std::endl;
			return;
		}

		std::cout<<"Reading in variable "<<variable_name<<" ...\n";
		variablePtr = matGetVariable(pmat, variable_name);
		if (variablePtr == NULL) {
			std::cerr<<"Error can't find variable "<<variable_name<<std::endl;
		}

		std::cout<<"cell dimension: row "<<mxGetM(variablePtr)<<", col "<<mxGetN(variablePtr)<<std::endl;
		features_subset = new double **[feature_subset_num];
		feature_num = mxGetM(variablePtr)*mxGetN(variablePtr);
		for (int i=0; i<feature_subset_num; i++){
			features_subset[i] = new double *[mxGetM(variablePtr)*mxGetN(variablePtr)];
			for (int j=0; j<mxGetM(variablePtr)*mxGetN(variablePtr); j++){
                features_subset[i][j] = new double[FEATURE_DIMENTION];
			}
		}

		for (int i=0; i<mxGetM(variablePtr)*mxGetN(variablePtr); i++){
			elementPtr = mxGetCell(variablePtr, i);
			if(elementPtr == NULL){
				std::cerr<<"Empty cell\n";
			}
			//features_subset[i] = new double* [feature_dimension];		
            for (int j=0; j<FEATURE_DIMENTION; j++){
				//features_subset[i][j] = new double[feature_subset_num];
				for(int k=0; k<feature_subset_num; k++){
					features_subset[k][i][j] = *((double *)mxGetData(elementPtr)+j*feature_subset_num+k);
				}
			}

		}
		mxDestroyArray(variablePtr);

		if (matClose(pmat) != 0) {
			std::cerr<<"Error closing file "<<filename<<std::endl;
			return;
		}
		//PersonIndInLabel();
		//std::cout<<"Successfully read in "<<sizeof(feature_data)<<" data.\n";
	}
    */
	void ReIdentification::WriteModelToMatfile(){
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
        memcpy((void *)(mxGetPr(pa)), (void *)svm_model, sizeof(double)*FEATURE_DIMENSION);
		status = matPutVariable(pmat, "svm_weights", pa);
		if (status != 0) {
			printf("%s :  Error using matPutVariable on line %d\n", __FILE__, __LINE__);
			return;
		}
        double b = svm_model[FEATURE_DIMENSION];
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
    void ReIdentification::WriteModelToBinfile(){
        ofstream file("model.bin", ios::out|ios::binary|ios::trunc);
        int size = (FEATURE_DIMENSION+1)*sizeof(double);
        file.write((char*)svm_model, size);
        file.close();
        /* read bin file example
        ifstream infile("model.bin", ios::in|ios::binary);
        double* data = new double[FEATURE_DIMENTION+1];
        infile.seekg(0, infile.beg);
        infile.read((char*)data, size);
        infile.close();
        */
    }
	void ReIdentification::ReadModelFromBinfile(){
		ifstream infile("model.bin", ios::in|ios::binary);
		int size = (FEATURE_DIMENSION+1)*sizeof(double);
		svm_model = new double[FEATURE_DIMENSION+1];
		infile.seekg(0, infile.beg);
		infile.read((char*)svm_model, size);
		infile.close();
	}
    double* ReadModelFile(char* filename){
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
    /*
	double* ReadModelFromMatlab(const char* filename, const char* variable_name){
		mxArray *variablePtr;
		std::cout<<"Reading file"<<filename<<"...\n\n";

        // Open file to get directory
		MATFile *pmat = matOpen(filename, "r");
		if (pmat == NULL) {
			std::cerr<<"Error opening file "<<filename<<std::endl;
			return NULL;
		}

		std::cout<<"Reading in variable "<<variable_name<<" ...\n";
		variablePtr = matGetVariable(pmat, variable_name);
		if (variablePtr == NULL) {
			std::cerr<<"Error can't find variable "<<variable_name<<std::endl;
		}

		std::cout<<"variable dimension: row "<<mxGetM(variablePtr)<<", col "<<mxGetN(variablePtr)<<std::endl;

		int size = mxGetM(variablePtr)*mxGetN(variablePtr);
		double* var = new double [size+1]; // 1 for b....
		var = mxGetPr(variablePtr);
		var[size] = 0;
		//mxDestroyArray(variablePtr);
		
		if (matClose(pmat) != 0) {
			std::cerr<<"Error closing file "<<filename<<std::endl;
			return NULL;
		}
		return var;
	}
    */
	void WriteDataToMatfile(void* data, const int data_size, const char* file, const char* variable_name){
		MATFile *pmat;
		mxArray *pa;
		//mxArray *pd;
		int status;
		pmat = matOpen(file, "w");
		if (pmat == NULL) {
			printf("Error creating file %s\n", file);
			printf("(Do you have write permission in this directory?)\n");
			return;
		}
		pa = mxCreateDoubleMatrix(data_size, 1, mxREAL);
		if (pa == NULL) {
			printf("%s : Out of memory on line %d\n", __FILE__, __LINE__); 
			printf("Unable to create mxArray.\n");
			return;
		}
		memcpy((void *)(mxGetPr(pa)), (void *)data, sizeof(double)*data_size); //data[0] used to be double
		status = matPutVariable(pmat, variable_name, pa);
		if (status != 0) {
			printf("%s :  Error using matPutVariable on line %d\n", __FILE__, __LINE__);
			return;
		}

		mxDestroyArray(pa);
		//mxDestroyArray(pd);
		if (matClose(pmat) != 0) {
			printf("Error closing file %s\n",file);
			return;
		}
	}

    int* GenerateRandInd(const int total_num, const int required_num, bool* mask){
		//this function is used to generate required number of random indexes, from 0 to total_num
		int *ind = new int[required_num];
		srand(time(NULL));
		int count = 0;
		int temp;
		while(count < required_num){
			temp = rand()%total_num;
#ifdef ETHZ
			if (temp==137) continue; //麻痹不够6张...
#endif
			if(!mask[temp]){
				mask[temp]=1;
				ind[count++]=temp;// + rand()%img_per_person * total_num;
			}
		}
		std::sort(ind, ind+required_num);
		return(ind);
	}
/*
    void ReIdentification::WritePairDiffData(std::ofstream *file, int m, int n, int ref_shot){
		// m,n are person indexes, starting from 0
		// for training and testing purpose, we need to calculate pairwise feature difference
		// If one person have multiple images, we randomly choose one from person_ind
		//std::vector<int> m_ind = person_ind[m];
		//std::vector<int> n_ind = person_ind[n];

		srand(time(NULL));
		//int ind = 0;//rand()%(m_ind.size());
        vector<double> ref_feature = features[m].at(ref_shot);

		n=n%person_num;
		if((m%person_num)==n){         
            for(int i =0;i<features[n].size(); i++){
                if(ref_shot==i)
					continue;
				*file<<"1 qid:"<<(m+1)<<" ";
                vector<double> pair_feature = features[n].at(i);
				for(int j = 0; j<feature_dimension; j++){
                    //double abs_feature = fabs(features[m][j]-features[n+i*person_num][j]);
                    double abs_feature = fabs(ref_feature[j]-pair_feature[j]);
					if(abs_feature!=0){
						*file<<(j+1)<<":"<<abs_feature<<" ";
					}
				}
				*file<<"# person "<<m+1<<" person  "<<n+1<<std::endl;
			}
		}else{
            for( int i =0;i<features[n].size(); i++){
				*file<<"2 qid:"<<(m+1)<<" ";
                vector<double> pair_feature = features[n].at(i);
				for(int j = 0; j<feature_dimension; j++){
                    //double abs_feature = fabs(features[m][j]-features[n+i*person_num][j]);
                    double abs_feature = fabs(ref_feature[j]-pair_feature[j]);
					if(abs_feature!=0){
						*file<<(j+1)<<":"<<abs_feature<<" ";
					}
				}
				*file<<"# person "<<m+1<<" person  "<<n+1<<std::endl;
			}
		}
	}
	double* ReIdentification::ReadPredictions(const int img_num){
		std::ifstream *file = new std::ifstream(prediction_filename);
		double *score = new double[img_num*(img_num*img_per_person-1)];
		int ind = 0;
		while(!file->eof()){
			*file>>score[ind++];
		}
		assert(ind!=img_num*(img_num*img_per_person-1));
		file->close();
		delete file;
		return score;
	}
    double* ReIdentification::ReadPrediction(const char* file_name){
		std::ifstream *file = new std::ifstream(file_name);
		double *score = new double[test_size*(test_size*img_per_person-1)];
		int ind = 0;
		while(!file->eof()){
			*file>>score[ind++];
		}
		assert(ind!=test_size*img_per_person-1);
		file->close();
		delete file;
		return score;
	}
    */
	bool ScorePairCompare(std::pair<int, double> p1, std::pair<int, double> p2){
		return p1.second>p2.second;
	}

	void RankExamples(const double *score, const int top_rank_num, const int img_rank_num, const int size, int* img_per_person, int* inds, int* gallery_inds){
		int *found_rank = new int[top_rank_num];
		double *top_rank_inds_cand = new double[top_rank_num*img_rank_num];
		double *top_rank_inds_gallery = new double[top_rank_num * img_rank_num];
		double *top_rank_inds = new double[top_rank_num];
		double *top_rank_inds_prob = new double[top_rank_num]; //for multishot
		std::fill(top_rank_inds, top_rank_inds+top_rank_num, 0);
		std::fill(found_rank, found_rank+top_rank_num, 0);

		//loop through all the person, find some top rank examples
		int global_pos = 0;
		for (int i=0; i<size; i++){
			//std::vector<double> score_per_person(score+i*observation_per_person, score+(i+1)*observation_per_person);
            std::vector<std::pair<int, double> > score_per_person;
			for (int j = 0; j < img_per_person[i]; j++) {
				if (j == gallery_inds[i]) continue;
				score_per_person.clear();
				for(int k = 0; k < size; k++) {
					score_per_person.push_back(std::make_pair(k, *(score + global_pos + k)));
				}

				std::sort(score_per_person.begin(), score_per_person.end(), ScorePairCompare);
				std::vector<std::pair<int, double> >::iterator target_it = std::find(score_per_person.begin(), score_per_person.end(), std::make_pair(i, *(score+global_pos+i)));
				global_pos += size;
				int rank = (int)(target_it - score_per_person.begin());
				if(rank>top_rank_num || found_rank[rank])
					continue;
				found_rank[rank]=1;
				top_rank_inds[rank]= inds[target_it->first];
				top_rank_inds_prob[rank] = j;

				std::vector<std::pair<int, double> >::iterator example_it;
				for(example_it=score_per_person.begin(); example_it<score_per_person.begin()+img_rank_num; example_it++){
					int k = example_it-score_per_person.begin();
					top_rank_inds_cand[rank * img_rank_num + k] = inds[example_it->first];
					top_rank_inds_gallery[rank * img_rank_num + k] = gallery_inds[example_it->first];
				}
				break; //after find one shot in any person, there is no point to go to other shots of the same person
			}

			bool found_all = true;
			for(int j=0; j<top_rank_num; j++){
				if(found_rank[j]==0){
					found_all = false;
					break;
				}
			}
			if(found_all)
				break;
		}
		WriteDataToMatfile(top_rank_inds, top_rank_num, "top_rank_inds.mat", "top_inds");
		WriteDataToMatfile(top_rank_inds_prob, top_rank_num, "top_rank_inds_prob.mat", "top_inds_prob");
		WriteDataToMatfile(top_rank_inds_cand, top_rank_num*img_rank_num, "top_rank_inds_cand.mat", "top_inds_cand");
		WriteDataToMatfile(top_rank_inds_gallery, top_rank_num*img_rank_num, "top_rank_inds_gallery.mat", "top_inds_gallery");
		delete [] top_rank_inds;
		delete [] found_rank;
		delete [] top_rank_inds_gallery;
		delete [] top_rank_inds_cand;
		delete [] top_rank_inds_prob;
	}

	/*
	int ReIdentification::TotalImageNum(int ind[], int size){
		int count = 0;
		for(int i=0; i<size; i++){
			std::vector<int> temp = person_ind[ind[i]];
			count+=temp.size();
		}
		return count;
	}
	*/
	void WriteIndFile(const char* filename, const int* ind, const int size){
		std::ofstream *file= new std::ofstream(filename);
		for(int i=0; i<size; i++){
			*file<<ind[i]<<" ";
		}
		*file<<std::endl;
		file->close();
		delete file;
	}
	void ReadIndFile(const char* filename, int* ind, const int size){
		std::ifstream *file = new std::ifstream(filename);
		for(int i=0; i<size; i++){
			*file>>ind[i];
		}
		file->close();
		delete file;
	}
/*
    void ReIdentification::GenerateTrainTestData(){
		int *train_ind = new int[train_size];
		int *test_ind = new int[test_size];
		bool *ind_mask = new bool[person_num];
		std::fill(ind_mask, ind_mask+person_num, 0);
		//struct stat buf;
		if(read_ind_from_file){//file exists
			ReadIndFile("train_ind.txt", train_ind, train_size);
			ReadIndFile("test_ind.txt", test_ind, test_size);
		}else{
            train_ind = GenerateRandInd(person_num, train_size, ind_mask);
            test_ind = GenerateRandInd(person_num, test_size, ind_mask);
			WriteIndFile("train_ind.txt", train_ind, train_size);
			WriteIndFile("test_ind.txt", test_ind, test_size);
		}
		
        srand(time(NULL));
		std::ofstream *train_file = new std::ofstream(train_filename);
		std::ofstream *test_file = new std::ofstream(test_filename);
		int train_count = 0;
		int test_count = 0;
		for(int i=0; i<person_num; i++){
			if(i==train_ind[train_count]){
                int ref_shot = rand()%features[i].size();
				for(int j=0; j<train_size; j++){
                    WritePairDiffData(train_file, i, train_ind[j],ref_shot);
				}
				train_count++;
			}else if(i==test_ind[test_count]){
                int ref_shot = rand()%features[i].size();
				for(int j=0; j<test_size; j++){
                    WritePairDiffData(test_file, i, test_ind[j],ref_shot);
				}
				test_count++;
			}
		}

		delete [] train_ind;
		delete [] test_ind;
		delete [] ind_mask;
	}
*/
/*
    void ReIdentification::WriteSample(int m, int n, PATTERN* x, LABEL* y, const int ref_shot, bool train_test){
		WORD* words;
		int wpos;

        vector<double> ref_feature = features[m].at(ref_shot);

		n=n%person_num;
        if((m%person_num)==n){ //same person
            //for test, we only generate one image pair

            for(int i =0;i<features[n].size(); i++){ //loop through multi-shots
                if(ref_shot==i)//(m/person_num==i)
                    continue;
				x->totdoc++;
				y->totdoc++;

				y->_class[y->totdoc-1]=2;
                words = (WORD *)my_malloc(sizeof(WORD)*(FEATURE_DIMENTION));
                wpos = 0;
                vector<double> pair_feature = features[n].at(i); //here m==n
                for(int j = 0; j<FEATURE_DIMENTION; j++){
                    //double abs_feature = fabs(features[m][j]-features[n+i*person_num][j]);
                    double abs_feature = fabs(ref_feature[j] - pair_feature[j]);
					if(abs_feature!=0){
						words[wpos].wnum = j+1;
                        words[wpos].weight =(FVAL)abs_feature;
						wpos++;
					}
				}
				(words[wpos]).wnum=0;
				//char * comment = "person "+(m+1)+" person  "+(n+1);
				x->doc[x->totdoc-1] = create_example(dnum++,m+1,0,1,create_svector(words,"",1.0));
				free(words);
                //if in test, we only consider 1 positive pair, default choose first image.
                if(train_test) break;
#ifdef SINGLE_SHOT
                break;
#endif
			}
        }else{ //different person
            for(int i =0;i<features[n].size(); i++){
				x->totdoc++;
				y->totdoc++;

				y->_class[y->totdoc-1]=1;
                words = (WORD *)my_malloc(sizeof(WORD)*(FEATURE_DIMENTION));
				wpos = 0;
                vector<double> pair_feature = features[n].at(i);
                for(int j = 0; j<FEATURE_DIMENTION; j++){
                    //double abs_feature = fabs(features[m][j]-features[n+i*person_num][j]);
                    double abs_feature = fabs(ref_feature[j] - pair_feature[j]);
					if(abs_feature!=0){
						words[wpos].wnum = j+1;
                        words[wpos].weight =(FVAL)abs_feature;
						wpos++;
					}
				}
				(words[wpos]).wnum=0;
                x->doc[x->totdoc-1] = create_example(dnum++,m+1,0,1,create_svector(words,"",1.0));
                free(words);
#ifdef SINGLE_SHOT
                if(i==1) break;
#endif
			}
		}
		
	}
*/
    void ReIdentification::WriteSample(int m, double f1[], double f2[], PATTERN* x, LABEL* y, int rank){
        //rank - sample rank, positive->2, negative->1
        //ind - person index
        WORD* words;
        int wpos;

        x->totdoc++;
        y->totdoc++;
        y->_class[y->totdoc-1]=rank;
        words = (WORD *)my_malloc(sizeof(WORD)*(FEATURE_DIMENSION));
        wpos = 0;
        for(int i = 0; i<FEATURE_DIMENSION; i++){
            double abs_feature = fabs(f1[i] - f2[i]);
            if(abs_feature!=0){
                words[wpos].wnum = i+1;
                words[wpos].weight =(FVAL)abs_feature;
                wpos++;
            }
        }
        (words[wpos]).wnum=0;
        //char * comment = "person "+(m+1)+" person  "+(n+1);
        x->doc[x->totdoc-1] = create_example(dnum++,m+1,0,1,create_svector(words,"",1.0));
        free(words);
    }
    SAMPLE ReIdentification::ConstructSample(int* index, const int size, bool train_test){
		SAMPLE sample;
		EXAMPLE  *examples;	
		PATTERN  *x;
		LABEL    *y;

		sample.n = 0;
		examples = NULL;
		x=NULL;
		y=NULL;

		for (int i=0; i<size; i++){
            int tmp = 0;
			/*
            for(int j=i+1; j<size; j++){
                tmp += train_img_per_person[j];
            }
            tmp = tmp*train_img_per_person[i] + train_img_per_person[i]*(train_img_per_person[i]-1)/2;
			*/
			tmp = train_img_per_person[i]*(total_train_img-train_img_per_person[i])+train_img_per_person[i]*(train_img_per_person[i]-1)/2;

			sample.n++;
			examples=(EXAMPLE *)realloc(examples,sizeof(EXAMPLE)*sample.n);
			x=&examples[sample.n-1].x;
			y=&examples[sample.n-1].y;
            x->doc=(DOC**)my_malloc(sizeof(DOC*)*(tmp));
			x->totdoc=0;
			y->factor=NULL;
            y->_class=(double *)my_malloc(sizeof(double)*(tmp));
			y->loss=0;
			y->totdoc=0;

            //the index i,j represent person, m,n represent shot.
            if(train_test==0){//construct training sample
                for(int m=0; m<train_img_per_person[i]; m++){
                    //positive samples
                    for(int n=m+1; n<train_img_per_person[i]; n++){
#ifdef ETHZ
						WriteSample(index[i], &features[get_train_feature_pos(i, train_shots_ind[i*SHOT_NUM+m])], &features[get_train_feature_pos(i,train_shots_ind[i*SHOT_NUM+n])], x, y, 2);
#else
                        WriteSample(index[i], &features[get_train_feature_pos(i, m)], &features[get_train_feature_pos(i,n)], x, y, 2);
#endif
                    }
                    //negative samples
                    for(int j=0; j<size; j++){
						if(i==j) continue;
                        for(int n=0; n<train_img_per_person[j]; n++){
#ifdef ETHZ
							WriteSample(index[i], &features[get_train_feature_pos(i, train_shots_ind[i*SHOT_NUM+m])], &features[get_train_feature_pos(j,train_shots_ind[j*SHOT_NUM+n])], x, y, 1);
#else
                            WriteSample(index[i], &features[get_train_feature_pos(i, m)], &features[get_train_feature_pos(j,n)], x, y, 1);
#endif
                        }
                    }
                }
            }else{//construct testing sample
                for(int m=0; m<test_img_per_person[i]; m++){
					if(m==test_gallery_ind[i]) continue;
                    for(int j=0; j<size; j++){//test_gallery size = test_size
                         WriteSample(index[i], &features[get_test_feature_pos(i,m)], &features[get_test_feature_pos(j, test_gallery_ind[j])], x, y, (i==j)?2:1);
                    }
                }
            }
			//cout<<"I think sample num: "<<tmp<<", true sample num: "<<x->totdoc<<endl;
			assert(tmp==x->totdoc);
            /*
            srand(time(NULL));
            int ref_shot = rand()%features[index[i]].size();//for each person, choose one shot as reference to compare with others
			for(int j=0; j<size; j++){
                WriteSample(index[i], index[j], x, y, ref_shot, train_test);
			}
            */
		}
		sample.examples = examples;		
		return sample;
	}
	void ReIdentification::Initialization(){
		train_ind = new int[train_size];
		test_ind = new int[test_size];
		test_gallery_ind = new int[test_size];
        train_img_per_person = new int[train_size];
        test_img_per_person = new int[test_size];
		bool *ind_mask = new bool[person_num];
		std::fill(ind_mask, ind_mask+person_num, 0);
		//struct stat buf;
		if(read_train_ind_from_file){//file exists
			ReadIndFile(train_ind_filename, train_ind, train_size);
			//ReadIndFile("test_ind.txt", test_ind, test_size);
			for(int i=0; i<train_size; i++){
				ind_mask[train_ind[i]%person_num]=1;
			}	
		}else{
#ifdef PRID
			train_ind = GenerateRandInd(200, train_size, ind_mask);
#else
            train_ind = GenerateRandInd(person_num, train_size, ind_mask);
#endif
			WriteIndFile("train_ind.txt", train_ind, train_size);
			//test_ind = GenerateRandInd(person_num, img_per_person, test_size, ind_mask);		
			//WriteIndFile("test_ind.txt", test_ind, test_size);
		}
		
		int ind = 0;

        for (int i=0; i<person_num; i++){
			if(!ind_mask[i])
                test_ind[ind++]=i;//+ rand()%img_per_person * person_num;
            if(ind == test_size) break;
		}

		assert(ind==test_size);

        total_train_img = 0;
        total_test_img = 0;
#ifdef ETHZ
		train_shots_ind = new int[SHOT_NUM*train_size];
		int c = 0;
#endif
        for(int i=0; i<train_size; i++){
            train_img_per_person[i] = img_per_person[train_ind[i]];
#ifdef ETHZ
			train_img_per_person[i] = SHOT_NUM;//when change this, search all ETHZ
			if(read_train_ind_from_file){
				ReadIndFile("ETHZ_train_shots_ind.txt", train_shots_ind, SHOT_NUM*train_size);
			}else{
				srand(time(NULL));
				int j=0;
				int num = img_per_person[train_ind[i]];
				bool* tmp_mask = new bool[num];
				fill(tmp_mask, tmp_mask+num, false);
				while(j<train_img_per_person[i]){
					int tmp = rand()%num;
					if(tmp_mask[tmp]) continue;
					train_shots_ind[c++] = tmp;
					tmp_mask[tmp] = true;
					j++;
				}
				delete [] tmp_mask;
				WriteIndFile("ETHZ_train_shots_ind.txt", train_shots_ind, SHOT_NUM*train_size);
			}
			
#endif
            total_train_img += train_img_per_person[i];
        }
        for(int i=0; i<test_size; i++){
            test_img_per_person[i] = img_per_person[test_ind[i]];
            total_test_img += test_img_per_person[i];
        }
		if(read_test_gallery_from_file){
			ReadIndFile(test_gallery_filename, test_gallery_ind, test_size);
		}else{
			set_test_gallery_ind();
			WriteIndFile("test_gallery_ind.txt", test_gallery_ind, test_size);
		}
        //train_sample = ConstructSample(train_ind, train_size);
        //test_sample = ConstructSample(test_ind, test_size, 1);
		delete [] ind_mask;
	}
    /*
	double eval_error(const double *D, const double *score, const int img_per_person, const int person_num, bool error_flag){
		double error = 0;
		int count = 0;

		int observation_per_person = img_per_person*person_num-1;
		for(int i=0; i<person_num; i++){
			int positive_observation_ind = i*observation_per_person+i*img_per_person;
			double error_per_person = 0;
			double max_error = 0;
			for(int j=0; j<observation_per_person; j++){				
				if(img_per_person*i!=j){
					double temp_error = score[i*observation_per_person+j]-score[positive_observation_ind];
					if(temp_error>0){
						error_per_person += error_flag?D[count]*(-temp_error):D[count];
					}
					else{
						error_per_person += error_flag?D[count]*(-temp_error):0;
					}
					if(abs(temp_error)>max_error)
						max_error = abs(temp_error);
					count++;
				}				
			}
			error +=  error_flag?error_per_person/max_error:error_per_person;
		}
		assert(count==person_num*(observation_per_person-1));
		return error;
	}
	void update_D(double alpha, double *D, const double *score, const int img_per_person, const int person_num){
		int count = 0;
		double sum = 0;
		int observation_per_person = img_per_person*person_num-1;
		for(int i=0; i<person_num; i++){
			int positive_observation_ind = i*observation_per_person+i*img_per_person;
			double max_error = 0;
			for(int j=0; j<observation_per_person; j++){
				if(img_per_person*i!=j){
					if(max_error<abs(score[i*observation_per_person+j]-score[positive_observation_ind]))
						max_error = abs(score[i*observation_per_person+j]-score[positive_observation_ind]);
				}
			}
			for(int j=0; j<observation_per_person; j++){
				if(img_per_person*i!=j){			
					//if(score[i*observation_per_person+j]>score[positive_observation_ind]){
						D[count]= D[count]*exp(alpha*(score[i*observation_per_person+j]-score[positive_observation_ind])/max_error);
					//}
					sum +=D[count];
					count++;
				}
			}
		}
		for (int i=0; i<person_num*(observation_per_person-1); i++){
			D[i] /=sum;
		}
		assert(count==person_num*(observation_per_person-1));
	}
	void ReIdentification::EnSemble(){
		SAMPLE* subimg_samples = new SAMPLE[feature_subset_num];
		double **scores = new double *[feature_subset_num];
		double *model = ReadModelFile(model_filename, feature_dimension);
		//ensemble para
		int total_pair_num = train_size*(train_size-1)*img_per_person;
		double *D = new double[total_pair_num];
		std::fill(D, D+total_pair_num, (double)1/(double)total_pair_num);
		double *error = new double[feature_subset_num];
		std::fill(error, error+feature_subset_num, 0);
		double *error_R = new double[feature_subset_num];
		std::fill(error_R, error_R+feature_subset_num, 0);

		//double *score_a = new double[feature_subset_num*train_size*(train_size*img_per_person-1)];
		//subimg_samples = train_sample;
		
		for (int i=0; i<feature_subset_num; i++){
			subimg_samples[i] = ConstructSample(train_ind, train_size, features_subset[i]);
		}
		
		//return para
		int iter = 20;
		double* beta = new double[iter];
		double* inds = new double[iter];
		for (int n = 0; n<iter; n++){
			double min_error;
			int min_ind;
			double alpha;
			//int** rankcount = new int* [feature_subset_num];
			for(int i=0; i<feature_subset_num; i++){
				//subimg_samples = ConstructSample(train_ind, train_size, features_subset[i]);
				std::string modelname = "model";
				modelname += i+1+48;
				modelname += ".dat";
				model = ReadModelFile((char*)modelname.c_str(), feature_dimension);

                //for(int j=0; j<feature_subset_num; j++){
                //	scores[i] = PredictionScore(subimg_samples[j], model);
                //	score_a = PredictionScore(subimg_samples[j], model);
                //}

				scores[i] = PredictionScore(subimg_samples[i], model);
				error[i] = eval_error(D, scores[i], img_per_person, train_size,false);
				error_R[i] = eval_error(D, scores[i], img_per_person, train_size,true);
				AnalysePrediction(scores[i],train_size);
				if(i==0 || min_error<error_R[i] ){//|| (abs(min_error - error[i])< (double)100/(double)total_pair_num && rankcount[min_ind][0]<rankcount[i][0]) ){
					min_error = error_R[i];
					min_ind=i;
				}
				//free_struct_sample(subimg_samples);
			}
			double r = error_R[min_ind];
			alpha = 0.5*log((1+r)/(1-r));
			update_D(alpha, D, scores[min_ind], img_per_person, train_size);

			beta[n] = alpha;
			inds[n] = (double)min_ind;
		}
		WriteDataToMatfile((void*)beta, iter, "ensemble_weights.mat", "beta");
		WriteDataToMatfile((void*)inds, iter, "ensemble_ind.mat", "min_inds");

		
		//delete [] subimg_samples;
		delete [] scores;
		delete [] model;
		delete [] D;
		delete [] error;
		
		//define the weightings for first (number of subset) and normalized it
		double *alpha_w = new double[feature_subset_num];
		double alpha_sum = 0;
		for (int i=0; i<feature_subset_num; i++){
			alpha_w[i] = beta[i];
			alpha_sum+=alpha_w[i];
		}
		for (int i=0; i<feature_subset_num; i++){
			alpha_w[i] /= alpha_sum;
		}

		//initialize weighted_features
		features_weighted = new double*[person_num*img_per_person];
		for(int i=0; i<person_num*img_per_person; i++){
			features_weighted[i] = new double[feature_dimension];
			for(int j=0; j<feature_dimension; j++){
				features_weighted[i][j]=0;
			}
		}
		for(int k =0; k<feature_subset_num; k++){
			for(int i=0; i<person_num*img_per_person; i++){
				for(int j=0; j<feature_dimension; j++){
					features_weighted[i][j] += alpha_w[(int)inds[k]]*features_subset[(int)inds[k]][i][j];
				}
			}
		}
		test_sample = ConstructSample(test_ind, test_size, features_weighted);
		delete [] beta;
		delete [] inds;

	}
    */
	//return model with dimension of feature_dimension, the last element is b value
	void ReIdentification::CalculateFeatureStats(){
		mu = new double[FEATURE_DIMENSION];
		var = new double[FEATURE_DIMENSION];

		for(int k = 0; k<FEATURE_DIMENSION; k++){
			//calculate mean
			double mean = 0;//accumulate dimensional feature
			for(int i=0; i<train_size; i++){
				double tmp = 0;
				for(int j=0; j<train_img_per_person[i]; j++){
#ifdef ETHZ
					tmp+=features[get_train_feature_pos(i,train_shots_ind[i*SHOT_NUM+j])+k];
#else
					tmp+=features[get_train_feature_pos(i,j)+k];
#endif
				}
				mean += tmp/train_img_per_person[i];
			}
			mu[k] = mean/train_size;
			//calculate variance
			double variance = 0;
			for(int i=0; i<train_size; i++){
				double tmp = 0;
				for(int j = 0; j<train_img_per_person[i]; j++){
#ifdef ETHZ
					tmp+=features[get_train_feature_pos(i,train_shots_ind[i*SHOT_NUM+j])+k];
					//tmp += (features[get_train_feature_pos(i,train_shots_ind[i*6+j])+k]-mu[k])*(features[get_train_feature_pos(i,train_shots_ind[i*6+j])+k]-mu[k]);
#else
					tmp += features[get_train_feature_pos(i,j)+k];
					//tmp += (features[get_train_feature_pos(i,j)+k]-mu[k])*(features[get_train_feature_pos(i,j)+k]-mu[k]);
#endif
				}
				mean = tmp/train_img_per_person[i];
				variance += (mean-mu[k])*(mean-mu[k]);
			}
			var[k] = variance/(train_size-1);
		}
		
	}
	void ReIdentification::TrainModel(bool read_data){
		train_sample = ConstructSample(train_ind, train_size);
		char *svm_argv[5] = {"svm_rank_learn", "-c", svm_C, train_filename, model_filename};
		svm_struct_main(5, svm_argv, train_sample, read_data);
		
        svm_model = ReadModelFile(model_filename);
	}
	void ReIdentification::set_test_gallery_ind(){
		srand(time(NULL));
		for(int i = 0; i<test_size; i++){
			test_gallery_ind[i] = rand()%test_img_per_person[i];
		}
	}
	void ReIdentification::TestModel(bool read_data){
        //char *svm_classify_argv[4] = {"svm_rank_classify", test_filename, model_filename, prediction_filename};
		//svm_struct_classify_main(4, svm_classify_argv, test_sample, read_data);
		GenerateTestScore(svm_model);
		cout<<"DONE reading testing result!"<<endl;
		cout<<"RDC results:"<<endl;
		double* rank_percent = AnalyseTestScore(test_score);
		WriteDataToMatfile(rank_percent, 100, "VIPeR_rank.mat", "VIPeR_rank");
#ifdef DISCRIMINATIVE
		cout<<"discriminative results:"<<endl;
		rank_percent = AnalyseTestScore(test_score_disc);
		WriteDataToMatfile(rank_percent, 100, "VIPeR_DF_rank.mat", "VIPeR_DF_rank");
#endif

	}
/*
    double* ReIdentification::PredictionScore(SAMPLE sample, double* w, bool train_test){
        int global_pos = 0;
        int observation_cur_person = 0;

        int total_pair_num;
        //calculate total number of examples
        if(!train_test){
            total_pair_num = train_size*(total_train_img -1);
        }else{
            total_pair_num =(test_size-1)*total_test_img + test_size;
        }
#ifdef SINGLE_SHOT
        if(!train_test){
            total_pair_num = train_size*(2*train_size - 1);
        }else{
            total_pair_num = test_size*(2*test_size - 1);
        }
#endif
        //for constant img_per_person, total_img = sample.n*img_per_person
        double *score = new double[total_pair_num];
        //weighting in the inner product start from index 1, don't know why.../
        double *lin_w = new double[FEATURE_DIMENTION+1];
        lin_w[0] = w[FEATURE_DIMENTION];
        for(int i=1; i<FEATURE_DIMENTION+1; i++){
			lin_w[i] = w[i-1];
		}
        //calculate scores/
        for(int i=0; i<sample.n; i++){//sample.n is the size of training/testing set (train_size/test_size?)

            if(!train_test){
                observation_cur_person = total_train_img-1;
            }else{
                observation_cur_person = total_test_img-test_img_per_person[i]+1;
            }
#ifdef SINGLE_SHOT
            if(!train_test){
                observation_cur_person = 2*train_size - 1;
            }else{
                observation_cur_person = 2*test_size - 1;
            }
#endif
            for(int j=0; j<observation_cur_person; j++){
				//inner product
                score[global_pos+j]=0;
				SVECTOR *f;
				for(f=sample.examples[i].x.doc[j]->fvec;f;f=f->next)  {
                    score[global_pos+j]+=f->factor*sprod_ns(lin_w,f);
				}
                score[global_pos+j] -= lin_w[0];
				
			}
            global_pos += observation_cur_person;
		}

		delete [] lin_w;
		return score;
	}
    */

	double GaussianVal(double x, double mu, double var)  {
		return exp(-(x-mu)*(x-mu)/(2*var));
	}
    void ReIdentification::GenerateTestScore(double *w){
        EXAMPLE  examples;
        //PATTERN  x;
        //LABEL    y;

        int global_pos = 0, relative_pos = 0;
        int observation_cur_person = 0;

        int total_pair_num;
        //calculate total number of examples
        total_pair_num = (total_test_img-test_size)*test_size;

        //for constant img_per_person, total_img = sample.n*img_per_person
        test_score = new double[total_pair_num];
		fill(test_score, test_score+total_pair_num, 0);
#ifdef DISCRIMINATIVE
		test_score_disc = new double[total_pair_num];
		fill(test_score_disc, test_score_disc+total_pair_num, 0);
#endif
        /*weighting in the inner product start from index 1, don't know why...*/
        double *lin_w = new double[FEATURE_DIMENSION+1];
        lin_w[0] = w[FEATURE_DIMENSION];
        for(int i=1; i<FEATURE_DIMENSION+1; i++){
            lin_w[i] = w[i-1];
        }
        /*calculate scores*/
        for(int i=0; i<test_size; i++){//sample.n is the size of training/testing set (train_size/test_size?)
			relative_pos = 0;
            //observation_cur_person = total_test_img-test_img_per_person[i]+1;
            observation_cur_person = (test_img_per_person[i]-1)*test_size;
            //examples=(EXAMPLE *)my_malloc(sizeof(EXAMPLE));
            //x=&examples.x;
            //y=&examples.y;
            examples.x.doc= (DOC**)my_malloc(sizeof(DOC*)*(observation_cur_person));;
            examples.x.totdoc=0;
            examples.y.factor=NULL;
            examples.y._class=(double *)my_malloc(sizeof(double)*(observation_cur_person));
            examples.y.loss=0;
            examples.y.totdoc=0;

            for(int m=0; m<test_img_per_person[i]; m++){
				if(m==test_gallery_ind[i]) continue;

				double* lin_w_disc = new double[FEATURE_DIMENSION+1];
				lin_w_disc[0] = lin_w[0];
				for(int k = 0; k<FEATURE_DIMENSION; k++){
#ifdef DISCRIMINATIVE
					double p = GaussianVal(features[get_test_feature_pos(i,m)+k], mu[k], var[k]);
					lin_w_disc [k+1] = lin_w[k+1]*((p<0.5)?(1.3*(1-p)+1):1);
#else
					lin_w_disc[k+1] = lin_w[k+1];
#endif
				}
				
                for(int j=0; j<test_size; j++){//test_gallery size = test_size
					WriteSample(test_ind[i], &features[get_test_feature_pos(i,m)], &features[get_test_feature_pos(j, test_gallery_ind[j])], &examples.x, &examples.y, (i==j)?2:1);
					//calculate score
					SVECTOR *f;
					for(f=examples.x.doc[relative_pos+j]->fvec;f;f=f->next)  {
						test_score[global_pos+j]+=f->factor*sprod_ns(lin_w,f);
					}
					test_score[global_pos+j] -= lin_w[0];
#ifdef DISCRIMINATIVE
					for(f=examples.x.doc[relative_pos+j]->fvec;f;f=f->next)  {
						test_score_disc[global_pos+j]+=f->factor*sprod_ns(lin_w_disc,f);
					}
					test_score_disc[global_pos+j] -= lin_w_disc[0];
#endif
                }
				relative_pos += test_size;
				global_pos += test_size;
				delete [] lin_w_disc;

            }

            //free memory!
            for(int j=examples.x.totdoc-1; j>=0; j--){
                free(examples.x.doc[j]->fvec->userdefined);
                free(examples.x.doc[j]->fvec->words);
                free(examples.x.doc[j]->fvec);
                free(examples.x.doc[j]);
            }

            free(examples.x.doc);
            free(examples.y._class);

        }
        assert(global_pos==total_pair_num);
        delete [] lin_w;
    }
	double* ReIdentification::AnalyseTestScore(const double* score){
		int rank_count [100];
		double* rank_percent = new double[100];
		fill(rank_count, rank_count+100, 0);
		int global_pos = 0;
		int total_test_num = total_test_img-test_size;
		for(int i=0; i<test_size; i++){
			for(int j=0; j<test_img_per_person[i]; j++){
				if(j==test_gallery_ind[i]) continue;
				vector<double> tmp_score(score+global_pos, score+global_pos+test_size); //test gallery is fixed size, same as test_size
				sort(tmp_score.begin(), tmp_score.end(), greater<double>());
				int r = (int)(find(tmp_score.begin(), tmp_score.end(), score[global_pos+i]) - tmp_score.begin());
				for(int k = 0; k<100; k++){
					if(r<=k) rank_count[k]++;
				}
				global_pos += test_size;
			} //end for each person imgs		
		}
		assert(global_pos == total_test_num*test_size);
		RankExamples(score, 8, 25, test_size, test_img_per_person, test_ind, test_gallery_ind);
		cout<<"total test number: "<<total_test_num<<endl;
		for(int k = 0; k<100; k++){
			rank_percent[k] = (double)rank_count[k]*100/(double) total_test_num;
			cout<<"Ranking "<<k+1<<": "<<rank_percent[k]<<endl;
		}
		return rank_percent;
	}
    int* ReIdentification::AnalysePrediction(const double *score, const int prediction_size, bool train_test){
        //int observation_per_person = img_per_person*prediction_size-1;
		//int rank[5] ={1,5,10,20,50};
        //int* rank_count = new int[5];
		//int rank_count[5]={0,0,0,0,0};
		int count_num = 100;
		int* rank_count = new int[count_num];
        std::fill(rank_count, rank_count+count_num, 0);
		double rank_percentage[100];	
		int r;

        int global_pos=0, relative_pos = 0; //relative_pos gives the position of positive examples
        int positive_example_num = 0;

		for(int i=0; i<prediction_size; i++){

            int observation_cur_person;
            if(!train_test){
                positive_example_num = train_img_per_person[i]-1;
                observation_cur_person = total_train_img -1; //reference shot paired with every other shots
                if(i!=0) relative_pos += train_img_per_person[i-1];
            }else{
                positive_example_num = 1; //for testing we only use 1 pair of positive example!!!
                observation_cur_person = total_test_img - test_img_per_person[i]+1; //reference shot only pair with itself for 1 shot and all others
                if(i!=0) relative_pos += test_img_per_person[i-1];
            }

#ifdef SINGLE_SHOT
            if(!train_test){
                observation_cur_person = 2*train_size - 1;
            }else{
                observation_cur_person = 2*test_size - 1;
            }
#endif

            //std::vector<double> score_for_person(score+i*observation_per_person, score+(i+1)*observation_per_person);
            std::vector<double> score_for_person(score+global_pos, score+global_pos+observation_cur_person);


            std::sort(score_for_person.begin(), score_for_person.end());
            std::reverse(score_for_person.begin(), score_for_person.end());

			//std::cout<<"Ranking of testing person is ";
			std::vector<double>::iterator it;
            for(int j=0; j<positive_example_num; j++){ //total number of scores per person
                //it = std::find(score_for_person.begin(), score_for_person.end(),*(score+i*observation_per_person+img_per_person*i+j));
                it = std::find(score_for_person.begin(), score_for_person.end(),*(score+global_pos+relative_pos+j));
				//std::cout<<(it-score_for_person.begin())<<" ";
				
				r=(int)(it-score_for_person.begin());
				for(int k=0; k<count_num; k++){
					if(r<(k+1)) rank_count[k]++;
				}
			}			
            global_pos += observation_cur_person;
			//std::cout<<std::endl;
		}
		for(int k=0; k<count_num; k++){
			rank_percentage[k] = (double)rank_count[k]*100/(double)prediction_size;
			std::cout<<"Rank "<<k+1<<": "<<rank_percentage[k]<<"%\n";
		}
		std::cout<<std::endl;
        //WriteDataToMatfile(rank_percentage, count_num, "VIPeR_us.mat", "PriorSVM");
        //RankExamples(score, 8, 25, test_size, img_per_person, test_ind, person_num);
		return rank_count;
	}
    /*
	void ReIdentification::ModelBoosting(){
		//char *svm_classify_argv[4] = {"svm_rank_classify", train_filename, model_filename, prediction_filename};
		//svm_struct_classify_main(4, svm_classify_argv, train_sample, false);
		int size = train_size;
		//double *score = ReadPredictions(size);
        svm_model = ReadModelFile(model_filename);
        double *score = PredictionScore(train_sample, svm_model, 0);
		double *score_diff = new double[size*(size*img_per_person-1)];
		double *score_positive_best = new double[size]; //only useful when multiple images per persons

		double *rerank = new double[size*(size*img_per_person-1)];
		std::fill(rerank, rerank+size*(size*img_per_person-1),0);

		int observation_per_person = img_per_person*size-1;
		int rank[5] ={1,5,10,20,50};
		int rank_count[5]={0,0,0,0,0};
		int r;
		double R = 0;
		for(int i=0; i<size; i++){
			std::vector<double> score_for_person(score+i*observation_per_person, score+(i+1)*observation_per_person);

			std::sort(score_for_person.begin(), score_for_person.end());
			//std::reverse(score_for_person.begin(), score_for_person.end());

			std::cout<<"Ranking of testing person is ";
			std::vector<double>::iterator it;
			for(int j=0; j<img_per_person-1; j++){ //total number of scores per person
				it = std::find(score_for_person.begin(), score_for_person.end(),*(score+i*observation_per_person+img_per_person*i+j));
				std::cout<<(it-score_for_person.begin())<<" ";
				if(j==0)
					score_positive_best[i]=*(it);
				else if(*(it)>score_positive_best[i])
					score_positive_best[i] = *(it);
				r=(int)(it-score_for_person.begin());
				for(int k=0; k<5; k++){
					if(r<rank[k]) rank_count[k]++;
				}

				//mark the rank needs to be updated
				for(int k=0;k<size*img_per_person-1;k++){
					if(score[i*observation_per_person+k]<*(it) &&( k<img_per_person*i || k>=img_per_person*(i+1)-1) && rerank[i*observation_per_person+k]==0)
						rerank[i*observation_per_person+k]++;
				}
			}
			std::cout<<std::endl;

			for(int j=0; j<observation_per_person; j++){
				if(j>=img_per_person*i && j<img_per_person*(i+1)-1){ //all belongs to positive pair
					D[i*observation_per_person+j]=0;
					score_diff[i*observation_per_person+j]=0;
					continue;
				}
				score_diff[i*observation_per_person+j] = score[i*observation_per_person+j] - score_positive_best[i];
				if(score_diff[i*observation_per_person+j] <0)
					R+=D[i*observation_per_person+j];
			}
		}
		double alpha = 0.5*log((1-R)/R);
		double sum = 0;
		for(int j=0; j<size*(size*img_per_person-1); j++){
			D[j] *=exp(alpha*(-score_diff[j]));
			sum+=D[j] ;
		}
		for(int j=0; j<size*(size*img_per_person-1); j++){
			D[j] /= sum;
		}
		//rewrite model
		
        for(int i=0; i<FEATURE_DIMENTION+1; i++){
			boosting_svm_model[i]+=alpha*svm_model[i];
		}
		
		//update rank
		for(int i=0; i<size; i++){
			for(int j=0; j<observation_per_person; j++){
				train_sample.examples[i].y._class[j] += rerank[i*observation_per_person+j];
			}
		}

		for(int k=0; k<5; k++){
			std::cout<<"Rank "<<rank[k]<<": "<<rank_count[k]*100/size<<"%\n";
		}

		delete [] score_diff;
		delete [] score_positive_best;
		delete [] rerank;
		delete [] score;

		std::cout<<"********test for new weighting*******"<<std::endl;
        double* new_score = PredictionScore(test_sample, boosting_svm_model, 1);
        AnalysePrediction(new_score,test_size,1);
		delete [] new_score;
	}
    */
	
}
