package tests;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.Random;

import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ArffSaver;
import weka.filters.Filter;
import weka.filters.SimpleBatchFilter;
import weka.filters.unsupervised.attribute.RemoveByName;
import weka.filters.unsupervised.attribute.WordPMI;
import weka.filters.unsupervised.instance.RemoveWithValues;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.LibLINEAR;
import weka.classifiers.meta.FilteredClassifier;

// Evaluates PMI using different Endiburgh datasets

public class PMIEval {

	static public void main(String args[]) throws Exception {
		
		System.out.println(args[0]);
		System.out.println("PMI");
		
		BufferedReader reader = new BufferedReader(
				new FileReader(args[0]));
		Instances train = new Instances(reader);
		reader.close();
		long startTime=System.nanoTime();
		SimpleBatchFilter wordPMI=new WordPMI();
		wordPMI.setOptions(Utils.splitOptions("-M 10 -I 3 -L -S -P WORD-"));
		wordPMI.setInputFormat(train);
		Instances wordVSM=Filter.useFilter(train, wordPMI);
		long timewordVSMCompactFast=System.nanoTime()-startTime;	
		System.out.println("Time Minutes,"+timewordVSMCompactFast/(Math.pow(10, 9)*60));

		// Remove unlabelled data for training
		wordVSM.setClassIndex(wordVSM.numAttributes()-1);
		
		RemoveWithValues f=new RemoveWithValues();
		f.setOptions(Utils.splitOptions("-S 0.0 -C last -L first-last -V -M"));
		f.setInputFormat(wordVSM);
		Instances wordVSMTrain=Filter.useFilter(wordVSM, f);

		
		
		// Trains a LibLinear classifier using the FilteredClassifier
		RemoveByName rbn=new RemoveByName();
		rbn.setOptions(Utils.splitOptions("-E WORD_NAME"));
		

		LibLINEAR ll=new LibLINEAR();
		
		ll.setOptions(Utils.splitOptions("-S 7 -C 1.0 -E 0.01 -B 1.0 -P"));

				
		FilteredClassifier fc = new FilteredClassifier();
		fc.setFilter(rbn);
		fc.setClassifier(ll);


		Evaluation eval = new Evaluation(wordVSMTrain);
		eval.crossValidateModel(fc, wordVSMTrain, 10, new Random(1));


		System.out.println("Train Instances,"+eval.numInstances());
		System.out.println("Accuracy,"+eval.pctCorrect());
		System.out.println("Weighted AUC,"+eval.weightedAreaUnderROC());

//		System.out.println(eval.toClassDetailsString());
//
//
//		System.out.println(eval.toMatrixString());
		 
		 
	}

}
