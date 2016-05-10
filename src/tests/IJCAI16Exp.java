package tests;

import it.unimi.dsi.fastutil.objects.Object2ObjectMap;
import it.unimi.dsi.fastutil.objects.Object2ObjectOpenHashMap;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.Arrays;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.LibLINEAR;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.Utils;
import weka.core.converters.ArffSaver;
import weka.experiment.PairedStats;
import weka.experiment.PairedStatsCorrected;
import weka.filters.Filter;
import weka.filters.SimpleBatchFilter;
import weka.filters.unsupervised.attribute.DistantSupervisionFilter;
import weka.filters.unsupervised.attribute.EmoticonDistantSupervision;
import weka.filters.unsupervised.attribute.RemoveByName;
import weka.filters.unsupervised.attribute.RemoveType;
import weka.filters.unsupervised.attribute.SampleFromTopics;
import weka.filters.unsupervised.attribute.ASA;
import weka.filters.unsupervised.attribute.WordCentroidMultiLabel;
import weka.filters.unsupervised.attribute.WordVSMMulti;
import weka.filters.unsupervised.instance.RemoveWithValues;

public class IJCAI16Exp {



	public IJCAI16Exp(){

	}


	public Instances createTrainData(String inputFile, DistantSupervisionFilter distantFilt) throws Exception{
		BufferedReader reader = new BufferedReader(
				new FileReader(inputFile));
		Instances train = new Instances(reader);
		reader.close();


		distantFilt.setInputFormat(train);
		Instances words=Filter.useFilter(train, distantFilt);

		words.setClassIndex(words.numAttributes()-1);

		return words;

	}

	public Instances mapTargetData(String input, DistantSupervisionFilter distantFilt) throws Exception{
		BufferedReader readerTest = new BufferedReader(
				new FileReader(input));

		Instances corpus = new Instances(readerTest);

		Instances targetData=distantFilt.mapTargetInstance(corpus);
		//	targetData.setClassIndex(targetData.numAttributes()-1);


		targetData.setClassIndex(targetData.numAttributes()-1);

		return targetData;

	}




	// Use a classifier trained from word-label for classifying tweets 


	public void evaluateDataSet(Instances trainData,Instances targetData) throws Exception{

		LibLINEAR ll=new LibLINEAR();
		ll.setOptions(Utils.splitOptions("-S 7 -C 1.0 -E 0.01 -B 1.0 -P"));

		RemoveType rm=new RemoveType();

		rm.setOptions(Utils.splitOptions("-T String"));

		FilteredClassifier fc = new FilteredClassifier();
		fc.setFilter(rm);
		fc.setClassifier(ll);
		fc.buildClassifier(trainData);

		// 
		Evaluation targetEval = new weka.classifiers.Evaluation(trainData);


		targetEval.evaluateModel(fc, targetData);


		System.out.println("kappa,"+targetEval.kappa());
		System.out.println("AvgF1,"+(targetEval.fMeasure(0)+targetEval.fMeasure(1))/2);
		System.out.println("AUC,"+targetEval.weightedAreaUnderROC());

	}




	public void processDistFilt(Instances sourceData,WordVSMMulti wcFilt) throws Exception{


		wcFilt.setInputFormat(sourceData);


		Instances wordCentroid=Filter.useFilter(sourceData, wcFilt);

		// Remove Unlabeled Words
		RemoveWithValues removeUnlabeled=new RemoveWithValues();
		removeUnlabeled.setOptions(Utils.splitOptions("-S 0.0 -C last -L first-last -V -M"));
		removeUnlabeled.setInputFormat(wordCentroid);
		Instances trainWords=Filter.useFilter(wordCentroid, removeUnlabeled);


		trainWords.setClassIndex(trainWords.numAttributes()-1);




		System.out.println(wcFilt.getUsefulInfo());
		System.out.println("Dist Model Attributes,"+ trainWords.numAttributes());
		System.out.println("Dist ModelInstances,"+ trainWords.numInstances());

		int classDist[]=trainWords.attributeStats(trainWords.numAttributes()-1).nominalCounts;
		
		System.out.println("Word label dist,"+classDist[0]+"-"+classDist[1]);

		String path1="testTweets/6HumanPosNeg.arff";
		Instances targetData1=this.mapTargetData(path1,wcFilt);
		System.out.println("Results on "+path1);		
		this.evaluateDataSet(trainWords, targetData1);

		System.out.println("Reverse results on "+path1);	
		this.evaluateDataSet(targetData1,trainWords );


		String path2="testTweets/SandersPosNeg.arff";
		Instances targetData2=this.mapTargetData(path2,wcFilt);
		System.out.println("Results on "+path2);		
		this.evaluateDataSet(trainWords, targetData2);

		System.out.println("Reverse results on "+path2);	
		this.evaluateDataSet(targetData2,trainWords);
		
		String path3="testTweets/SemEvalPosNeg.arff";
		Instances targetData3=this.mapTargetData(path3,wcFilt);
		System.out.println("Results on "+path3);		
		this.evaluateDataSet(trainWords, targetData3);

		System.out.println("Reverse results on "+path3);	
		this.evaluateDataSet(targetData3,trainWords );
		
		



		//		this.evaluateDataSet(trainWords, "testTweets/SandersPosNeg.arff", wcFilt);
		//		this.evaluateDataSet(trainWords, "testTweets/SemEvalPosNeg.arff", wcFilt);


		//		
		//		
		//		// Classify Unlabaeled Words and retrain
		//		RemoveWithValues removeLabeled=new RemoveWithValues();
		//		removeLabeled.setOptions(Utils.splitOptions("-S 0.0 -C last -L first-last"));
		//		removeLabeled.setInputFormat(wordCentroid);
		//		Instances unLabWords=Filter.useFilter(wordCentroid, removeLabeled);
		//		System.out.println("Unlabeled Instances,"+ unLabWords.numInstances());
		//		
		//		unLabWords.setClassIndex(unLabWords.numAttributes()-1);
		//		
		//	
		//		// Self-Training Step
		//		LibLINEAR ll=new LibLINEAR();
		//		ll.setOptions(Utils.splitOptions("-S 7 -C 1.0 -E 0.01 -B 1.0 -P"));
		//
		//		RemoveType rm=new RemoveType();
		//
		//		rm.setOptions(Utils.splitOptions("-T String"));
		//
		//		FilteredClassifier fc = new FilteredClassifier();
		//		fc.setFilter(rm);
		//		fc.setClassifier(ll);
		//		fc.buildClassifier(trainWords);
		//		
		//		
		//		double negProbs[]=new double[unLabWords.numInstances()];
		//		for(int i=0;i<unLabWords.numInstances();i++){
		//			negProbs[i]=fc.distributionForInstance(unLabWords.instance(i))[0];
		//		}
		//		
		//		System.out.println("\n Self-Trained Models");
		//	
		//		
		//		double alphas[]={0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95};
		//		
		//		
		//		for(double alpha:alphas){
		//			// Training words with Self Training
		//			Instances trainWordsSelfTr=new Instances(trainWords);
		//			
		//			
		//			
		//			for(int i=0;i<unLabWords.numInstances();i++){
		//				
		//				if(negProbs[i]>alpha){
		//					Instance negInst=unLabWords.instance(i);
		//					negInst.setValue(unLabWords.numAttributes()-1, 0);
		//					trainWordsSelfTr.add(negInst);
		//				}
		//				
		//				if(1-negProbs[i]>alpha){
		//					Instance posInst=unLabWords.instance(i);
		//					posInst.setValue(unLabWords.numAttributes()-1, 1);
		//					trainWordsSelfTr.add(posInst);
		//					
		//				}
		//				
		//				
		//				
		//			}
		//			
		//			System.out.println("\n alpha value,"+alpha);
		//	
		//			System.out.println("Dist ModelInstances,"+trainWordsSelfTr.numInstances());
		//			this.evaluateDataSet(trainWordsSelfTr, "testTweets/6HumanPosNeg.arff", wcFilt);
		//			this.evaluateDataSet(trainWordsSelfTr, "testTweets/SandersPosNeg.arff", wcFilt);
		//			this.evaluateDataSet(trainWordsSelfTr, "testTweets/SemEvalPosNeg.arff", wcFilt);
		//			
		//		}






	}



	// Take an input collection of tweets partionate it 10 versions, create different datasets and use them for evaluation


	// 	edimEx.arff	experiment/   1

	// 	edimEx.arff	experiment/   1 10 10000
	static public void main(String args[]) throws Exception{

		// Input String in args[0]
		String inputFile=args[0];


		IJCAI16Exp wlf=new IJCAI16Exp();

		//		DistantSupervisionFilter	emoFilt=new EmoticonDistantSupervision();
		//		emoFilt.setOptions(Utils.splitOptions("-M 1 -N 1 -W -C -I 3 -P WORD- -Q CLUST- -L -J lexicons/AFINN-posneg.txt -H resources/50mpaths2.txt -T resources/stopwords.txt -O"));


		WordVSMMulti distFiltNonEx=new WordVSMMulti();		
		// Non mutually exclusive false			
		distFiltNonEx.setOptions(Utils.splitOptions("-M 0 -N 0 -W -C -I 3 -P WORD- -Q CLUST- -L -K -J lexicons/AFINN-posneg-emot.txt  -H resources/50mpaths2.txt -R -T resources/stopwords.txt -O"));




		BufferedReader reader = new BufferedReader(
				new FileReader(inputFile));
		Instances sourceData = new Instances(reader);
		reader.close();

		//		emoFilt.setInputFormat(sourceData);
		//		Instances trainEmotData=Filter.useFilter(sourceData, emoFilt);
		//		trainEmotData.setClassIndex(trainEmotData.numAttributes()-1);
		//
		//
		//
		//		System.out.println(emoFilt.getUsefulInfo());
		//		System.out.println("Emot Corpus Attributes,"+ trainEmotData.numAttributes());
		//		System.out.println("Emot Corpus Instances,"+ trainEmotData.numInstances());
		//
		//		System.out.println("Emoticon Model results");
		//		wlf.evaluateDataSet(trainEmotData, "testTweets/6HumanPosNeg.arff", emoFilt);
		//		wlf.evaluateDataSet(trainEmotData, "testTweets/SandersPosNeg.arff", emoFilt);
		//		wlf.evaluateDataSet(trainEmotData, "testTweets/SemEvalPosNeg.arff", emoFilt);


		//	System.out.println("\n\n");


		System.out.println("Tweet Centroid");
		wlf.processDistFilt(sourceData, distFiltNonEx);




	}





}
