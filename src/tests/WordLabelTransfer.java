package tests;

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
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ArffSaver;
import weka.filters.Filter;
import weka.filters.SimpleBatchFilter;
import weka.filters.unsupervised.attribute.RemoveByName;
import weka.filters.unsupervised.attribute.RemoveType;
import weka.filters.unsupervised.attribute.WordCentroidMultiLabel;
import weka.filters.unsupervised.attribute.WordVSMMulti;
import weka.filters.unsupervised.instance.RemoveWithValues;

public class WordLabelTransfer {
	private String targetFolder;
	private WordVSMMulti wCentFilter;

	public WordLabelTransfer(String targetFolder){
		this.targetFolder=targetFolder;
	}


	public Instances createCentroids(String inputFile) throws Exception{
		BufferedReader reader = new BufferedReader(
				new FileReader(inputFile));
		Instances train = new Instances(reader);
		reader.close();
		this.wCentFilter=new WordVSMMulti();
		this.wCentFilter.setOptions(Utils.splitOptions("-M 1 -N 1 -W -C -I 3 -P WORD- -Q CLUST- -L -K -J lexicons/metaLexEmo.csv -H resources/50mpaths2.txt -R -T resources/stopwords.txt -O"));


		this.wCentFilter.setInputFormat(train);
		Instances words=Filter.useFilter(train, this.wCentFilter);


		RemoveWithValues f=new RemoveWithValues();
		f.setOptions(Utils.splitOptions("-S 0.0 -C last -L first-last -V -M"));
		f.setInputFormat(words);
		Instances wordsTrain=Filter.useFilter(words, f);

//		RemoveWithValues g=new RemoveWithValues();
//		g.setOptions(Utils.splitOptions("-S 0.0 -C last -L 2 -H"));
//		g.setInputFormat(wordsTrain);
//		Instances word2classTrain=Filter.useFilter(wordsTrain, g);


		wordsTrain.setClassIndex(wordsTrain.numAttributes()-1);

		return wordsTrain;

	}

	public Instances mapTargetData(String input) throws Exception{
		BufferedReader readerTest = new BufferedReader(
				new FileReader(input));

		Instances corpus = new Instances(readerTest);

		Instances targetData=this.wCentFilter.mapTargetInstance(corpus);
		targetData.setClassIndex(targetData.numAttributes()-1);

//		RemoveWithValues g=new RemoveWithValues();
//		g.setOptions(Utils.splitOptions("-S 0.0 -C last -L 2 -H"));
//		g.setInputFormat(targetData);
//		targetData=Filter.useFilter(targetData, g);

		targetData.setClassIndex(targetData.numAttributes()-1);
		
		return targetData;

	}
	
	public Classifier getClassifier() throws Exception{
		LibLINEAR ll=new LibLINEAR();
		ll.setOptions(Utils.splitOptions("-S 7 -C 1.0 -E 0.01 -B 1.0 -P"));
		
		
		RemoveType rm=new RemoveType();

		rm.setOptions(Utils.splitOptions("-T String"));
		
		FilteredClassifier fc = new FilteredClassifier();
		fc.setFilter(rm);
		fc.setClassifier(ll);
		return fc;
		
	}
	



	// Use a classifier trained from word-label for classifying tweets 


	// corpus.arff 
	static public void main(String args[]) throws Exception{


		// Input String in args[0]
		String inputFile=args[0];
		String targetPath=args[1];

		WordLabelTransfer wlf=new WordLabelTransfer(targetPath);


		Instances trainData=wlf.createCentroids(inputFile);


		ArffSaver saver = new ArffSaver();
		saver.setInstances(trainData);
		saver.setFile(new File(targetPath+"wordsTransferTrain.arff"));
		saver.writeBatch();


		Instances targetData=wlf.mapTargetData("testTweets/SandersPosNeg.arff");

		saver = new ArffSaver();
		saver.setInstances(targetData);
		saver.setFile(new File(targetPath+"SandersFormat.arff"));
		saver.writeBatch();

		
		Classifier fc=wlf.getClassifier();
		fc.buildClassifier(trainData);

		// 
		weka.classifiers.Evaluation targetEval = new weka.classifiers.Evaluation(trainData);
		System.out.println("Model Transfer Results");
		targetEval.evaluateModel(fc, targetData);
		System.out.println(targetEval.toSummaryString());	
		System.out.println(targetEval.toMatrixString("Confusion Matrix"));
		System.out.println(targetEval.toClassDetailsString());
		
	
		
		fc.buildClassifier(targetData);
		weka.classifiers.Evaluation targetEval2 = new weka.classifiers.Evaluation(targetData);
		System.out.println("Cross-Validation on Target Data Results");
		targetEval2.crossValidateModel(fc, targetData, 10, new Random(1));
		System.out.println(targetEval2.toSummaryString());
		System.out.println(targetEval2.toMatrixString("Confusion Matrix"));
		System.out.println(targetEval2.toClassDetailsString());
		

		//		mapTargetData("testTweets/6HumanPosNeg.arff",filter,g,targetPath+"6HumanPosNegFormat.arff");
		//		mapTargetData("testTweets/SemEvalPosNeg.arff",filter,g,targetPath+"SemEvalPosNegFormat.arff");







	}





}
