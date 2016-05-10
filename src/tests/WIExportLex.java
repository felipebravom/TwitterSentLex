package tests;

import it.unimi.dsi.fastutil.objects.Object2ObjectMap;
import it.unimi.dsi.fastutil.objects.Object2ObjectOpenHashMap;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.Arrays;
import java.util.Random;

import lexexpand.core.LexiconEvaluator;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.LibLINEAR;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.Attribute;
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

public class WIExportLex {



	public WIExportLex(){

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


	public void evaluateDataSet(Instances trainData,Instances targetData,LexiconEvaluator lex) throws Exception{

		LibLINEAR ll=new LibLINEAR();
		ll.setOptions(Utils.splitOptions("-S 7 -C 1.0 -E 0.01 -B 1.0 -P"));

		RemoveType rm=new RemoveType();

		rm.setOptions(Utils.splitOptions("-T String"));

		FilteredClassifier fc = new FilteredClassifier();
		fc.setFilter(rm);
		fc.setClassifier(ll);
		fc.buildClassifier(trainData);

		// 
		for(Instance inst:targetData){
			double[] dist=fc.distributionForInstance(inst);
			
			
		//	Attribute contentAtt=targetData.attribute("content");
			
			String wordName=inst.stringValue(targetData.numAttributes()-2);
			if(lex.getDict().containsKey(wordName))
				System.out.println(wordName+"\t"+dist[0]+"\t"+dist[1]+"\t"+lex.retrieveValue(wordName));
						
		
		}
		
		
	}




	public void processDistFilt(Instances sourceData,WordVSMMulti wcFilt) throws Exception{

		LexiconEvaluator lex= new LexiconEvaluator("lexicons/AFINN-posneg.txt");
		lex.processDict();

		wcFilt.setInputFormat(sourceData);


		Instances trainWords=Filter.useFilter(sourceData, wcFilt);

	
		trainWords.setClassIndex(trainWords.numAttributes()-1);


		String path1="testTweets/6HumanPosNeg.arff";
		Instances targetData1=this.mapTargetData(path1,wcFilt);


		System.out.println("\n\n Lexicon "+path1);	
		this.evaluateDataSet(targetData1,trainWords,lex );


		String path2="testTweets/SandersPosNeg.arff";
		Instances targetData2=this.mapTargetData(path2,wcFilt);

		System.out.println("\n \n Lexicon "+path2);	
		this.evaluateDataSet(targetData2,trainWords, lex);

		

			
		String path3="testTweets/SemEvalPosNeg.arff";
		Instances targetData3=this.mapTargetData(path3,wcFilt);		

		System.out.println("\n \n Lexicon "+path3);	
		this.evaluateDataSet(targetData3,trainWords,lex);
		

	}



	// Take an input collection of tweets partionate it 10 versions, create different datasets and use them for evaluation


	// 	edimEx.arff	experiment/   1

	// 	edimEx.arff	experiment/   1 10 10000
	static public void main(String args[]) throws Exception{

		// Input String in args[0]
		String inputFile=args[0];


		WIExportLex wlf=new WIExportLex();

		//		DistantSupervisionFilter	emoFilt=new EmoticonDistantSupervision();
		//		emoFilt.setOptions(Utils.splitOptions("-M 1 -N 1 -W -C -I 3 -P WORD- -Q CLUST- -L -J lexicons/AFINN-posneg.txt -H resources/50mpaths2.txt -T resources/stopwords.txt -O"));


		WordVSMMulti distFiltNonEx=new WordVSMMulti();		
		// Non mutually exclusive false			
		distFiltNonEx.setOptions(Utils.splitOptions("-M 10 -N 10 -W -C -I 3 -P WORD- -Q CLUST- -L -K -J lexicons/AFINN-posneg.txt  -H resources/50mpaths2.txt -R -T resources/stopwords.txt -O"));




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


		System.out.println("Expanded Lexicon Transfer Learning");
		wlf.processDistFilt(sourceData, distFiltNonEx);




	}





}
