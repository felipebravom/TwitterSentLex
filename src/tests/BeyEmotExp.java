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
import weka.filters.unsupervised.attribute.LexiconDistSuper;
import weka.filters.unsupervised.attribute.RemoveByName;
import weka.filters.unsupervised.attribute.RemoveType;
import weka.filters.unsupervised.attribute.SampleFromTopics;
import weka.filters.unsupervised.attribute.ASA;
import weka.filters.unsupervised.attribute.WordCentroidMultiLabel;
import weka.filters.unsupervised.attribute.WordVSMMulti;
import weka.filters.unsupervised.instance.RemoveWithValues;

public class BeyEmotExp {
	private String targetFolder;


	public BeyEmotExp(String targetFolder){
		this.targetFolder=targetFolder;
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


	public void evaluateDataSet(Instances trainData,String path, DistantSupervisionFilter distantFilt) throws Exception{
		Instances targetData=this.mapTargetData(path,distantFilt);


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

		System.out.println("Results on "+path);
		System.out.println("kappa,"+targetEval.kappa());
		System.out.println("AvgF1,"+(targetEval.fMeasure(0)+targetEval.fMeasure(1))/2);
		System.out.println("AUC,"+targetEval.weightedAreaUnderROC());


		//		System.out.println(targetEval.toSummaryString());	
		//		System.out.println(targetEval.toMatrixString("Confusion Matrix"));
		//		System.out.println(targetEval.toClassDetailsString());
		//		




		//
		//		fc.buildClassifier(targetData);
		//		weka.classifiers.Evaluation targetEval2 = new weka.classifiers.Evaluation(targetData);
		//		System.out.println("Cross-Validation on Target Data Results"+path);
		//		targetEval2.crossValidateModel(fc, targetData, 10, new Random(1));
		//		System.out.println(targetEval2.toSummaryString());
		//		System.out.println(targetEval2.toMatrixString("Confusion Matrix"));
		//		System.out.println(targetEval2.toClassDetailsString());

	}


	public Object2ObjectMap<String,double[]> evaluateModels(Instances trainData,String path, DistantSupervisionFilter distantFilt,int folds) throws Exception{


		System.out.println("Training "+distantFilt.getClass().getName()+" "+path);

		Instances targetData=this.mapTargetData(path,distantFilt);


		LibLINEAR ll=new LibLINEAR();
		ll.setOptions(Utils.splitOptions("-S 7 -C 1.0 -E 0.01 -B 1.0 -P"));

		RemoveType rm=new RemoveType();

		rm.setOptions(Utils.splitOptions("-T String"));

		FilteredClassifier fc = new FilteredClassifier();
		fc.setFilter(rm);
		fc.setClassifier(ll);
		fc.buildClassifier(trainData);

		// 
		weka.classifiers.Evaluation targetEval = new weka.classifiers.Evaluation(trainData);
		System.out.println("Model Transfer Results on:"+path);



		targetEval.evaluateModel(fc, targetData);
		System.out.println(targetEval.toSummaryString());	
		System.out.println(targetEval.toMatrixString("Confusion Matrix"));
		System.out.println(targetEval.toClassDetailsString());

		Random rand = new Random(1);
		Instances randData = new Instances(targetData);
		randData.randomize(rand);
		if (randData.classAttribute().isNominal())
			randData.stratify(folds);


		Object2ObjectMap<String,double[]> foldValues=new Object2ObjectOpenHashMap<String,double[]>();
		double[] kappas=new double[folds];
		double[] f1s=new double[folds];
		double[] aucs=new double[folds];

		for (int n = 0; n < folds; n++) {
			Evaluation eval = new Evaluation(randData);
			Instances test = randData.testCV(folds, n);
			// the above code is used by the StratifiedRemoveFolds filter, the
			// code below by the Explorer/Experimenter:
			// Instances train = randData.trainCV(folds, n, rand);
			eval.evaluateModel(fc, test);
			kappas[n]=eval.kappa();
			f1s[n]=(eval.fMeasure(0)+eval.fMeasure(1))/2;
			aucs[n]=eval.weightedAreaUnderROC();

		}

		foldValues.put("kappas", kappas);
		foldValues.put("f1s", f1s);
		foldValues.put("aucs", aucs);

		System.out.println("------------------------------");

		return foldValues;

		//   System.out.println(eval.toSummaryString("=== " + folds + "-fold Cross-validation ===", false));



	}



	static public void compareRes(Object2ObjectMap<String,double[]> res1,Object2ObjectMap<String,double[]> res2){
		for(String key:res1.keySet()){

			System.out.println(key);

			double[] res1Scores=res1.get(key);
			double[] res2Scores=res2.get(key);

			double res1Mean=weka.core.Utils.mean(res1Scores);
			double res1Sd=Math.sqrt(weka.core.Utils.variance(res1Scores));

			System.out.println(	weka.core.Utils.doubleToString(res1Mean, 3)+" +-"+weka.core.Utils.doubleToString(res1Sd, 3));


			double res2Mean=weka.core.Utils.mean(res2Scores);
			double res2Sd=Math.sqrt(weka.core.Utils.variance(res2Scores));


			PairedStats ps=new PairedStats(0.05);
			ps.add(res2Scores, res1Scores);
			ps.calculateDerived();
			//System.out.println(ps.toString());

			System.out.println(	weka.core.Utils.doubleToString(res2Mean, 3)+" +-"+weka.core.Utils.doubleToString(res2Sd, 3)+" "+ps.differencesSignificance);	



		}

	}


	public void processDistFilt(Instances sourceData,ASA distFiltEx) throws Exception{



		distFiltEx.setInputFormat(sourceData);
//		distFiltEx.setTweetsPerCentroid(10);			
//		System.out.println("DistFilt Tweets per Centroid "+distFiltEx.getTweetsPerCentroid());

		Instances trainDistData=Filter.useFilter(sourceData, distFiltEx);
//		trainDistData.setClassIndex(trainDistData.numAttributes()-1);
//
//
//
//		System.out.println(distFiltEx.getUsefulInfo());
//		System.out.println("Dist Model Attributes,"+ trainDistData.numAttributes());
//		System.out.println("Dist ModelInstances,"+ trainDistData.numInstances());
//		this.evaluateDataSet(trainDistData, "testTweets/6HumanPosNeg.arff", distFiltEx);
//		this.evaluateDataSet(trainDistData, "testTweets/SandersPosNeg.arff", distFiltEx);
//		this.evaluateDataSet(trainDistData, "testTweets/SemEvalPosNeg.arff", distFiltEx);
		
		
		System.out.println("");
		

		int[] tweetsPerCentroids={20};
		double[] fracsGenerate={0.2};
		
		
		for (int centSize:tweetsPerCentroids){
			distFiltEx.setTweetsPerCentroid(centSize);	
			
			for (double fracGen:fracsGenerate){
				System.out.println("DistFilt Tweets per Centroid "+distFiltEx.getTweetsPerCentroid()+" Frac Generate "+fracGen);


				trainDistData= distFiltEx.generateData( fracGen*sourceData.numInstances());
				trainDistData.setClassIndex(trainDistData.numAttributes()-1);



				System.out.println(distFiltEx.getUsefulInfo());
				System.out.println("Dist Model Attributes,"+ trainDistData.numAttributes());
				System.out.println("Dist ModelInstances,"+ trainDistData.numInstances());
				this.evaluateDataSet(trainDistData, "testTweets/6HumanPosNeg.arff", distFiltEx);
				this.evaluateDataSet(trainDistData, "testTweets/SandersPosNeg.arff", distFiltEx);
				this.evaluateDataSet(trainDistData, "testTweets/SemEvalPosNeg.arff", distFiltEx);


				System.out.println("");
				
			}
			

		}
		
			
		
		
		
		
		
	}



	// Take an input collection of tweets partionate it 10 versions, create different datasets and use them for evaluation


	// 	edimEx.arff	experiment/   1

	// 	edimEx.arff	experiment/   1 10 10000
	static public void main(String args[]) throws Exception{

		// Input String in args[0]
		String inputFile=args[0];
		String targetPath=args[1];

		BeyEmotExp wlf=new BeyEmotExp(targetPath);

//		DistantSupervisionFilter	emoFilt=new EmoticonDistantSupervision();
//		emoFilt.setOptions(Utils.splitOptions("-M 10 -N 10 -W -C -I 3 -P WORD- -Q CLUST- -L -J lexicons/AFINN-posneg.txt -H resources/50mpaths2.txt -T resources/stopwords.txt -O"));


//		SampleFromWords distFiltEx=new SampleFromWords();		
		double centNum=Double.parseDouble(args[2]);
		// mutually exclusive false			
//		distFiltEx.setOptions(Utils.splitOptions("-M 10 -N 10 -W -C -I 3 -P WORD- -Q CLUST- -L -E -J lexicons/AFINN-posneg.txt -H resources/50mpaths2.txt -T resources/stopwords.txt -O -A 10 -B "+centNum));

//		ASA distFiltNonEx=new ASA();		
//		// Non mutually exclusive false			
//		distFiltNonEx.setOptions(Utils.splitOptions("-M 10 -N 10 -W -C -I 3 -P WORD- -Q CLUST- -L -J lexicons/AFINN-posneg.txt -H resources/50mpaths2.txt -T resources/stopwords.txt -O -A 10 -B "+centNum));


		
	
		
		


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


		System.out.println("\n\n");


		// train data with distant model

//
//		System.out.println("Exclusive Sets");
//		wlf.processDistFilt(sourceData, distFiltNonEx);

//		Uncomment this for NonExclusive Results
//		
//		System.out.println("\n\n NonExclusive Sets");
//		wlf.processDistFilt(sourceData, distFiltNonEx);


		
		LexiconDistSuper lexFilt=new LexiconDistSuper();		
//		// Non mutually exclusive false			
		lexFilt.setOptions(Utils.splitOptions("-M 10 -N 10 -W -C -I 3 -P WORD- -Q CLUST- -L -J lexicons/AFINN-posneg.txt -H resources/50mpaths2.txt -T resources/stopwords.txt -O"));

		lexFilt.setBalance(false);
		
		System.out.println("Lexicon UnBalanced");
		lexFilt.setInputFormat(sourceData);
		Instances trainLexData=Filter.useFilter(sourceData, lexFilt);
		trainLexData.setClassIndex(trainLexData.numAttributes()-1);



		System.out.println(lexFilt.getUsefulInfo());
		System.out.println("Lex Corpus Attributes,"+ trainLexData.numAttributes());
		System.out.println("Lex Corpus Instances,"+ trainLexData.numInstances());

		System.out.println("Lexicon Model results");
		wlf.evaluateDataSet(trainLexData, "testTweets/6HumanPosNeg.arff", lexFilt);
		wlf.evaluateDataSet(trainLexData, "testTweets/SandersPosNeg.arff", lexFilt);
		wlf.evaluateDataSet(trainLexData, "testTweets/SemEvalPosNeg.arff", lexFilt);
		
		
		lexFilt=new LexiconDistSuper();		
//		// Non mutually exclusive false			
		lexFilt.setOptions(Utils.splitOptions("-M 10 -N 10 -W -C -I 3 -P WORD- -Q CLUST- -L -J lexicons/AFINN-posneg.txt -H resources/50mpaths2.txt -T resources/stopwords.txt -O"));

		lexFilt.setBalance(true);
		
		System.out.println("\n\n Lexicon Balanced");
		lexFilt.setInputFormat(sourceData);
		trainLexData=Filter.useFilter(sourceData, lexFilt);
		trainLexData.setClassIndex(trainLexData.numAttributes()-1);



		System.out.println(lexFilt.getUsefulInfo());
		System.out.println("Lex Corpus Attributes,"+ trainLexData.numAttributes());
		System.out.println("Lex Corpus Instances,"+ trainLexData.numInstances());

		System.out.println("Lexicon Model results");
		wlf.evaluateDataSet(trainLexData, "testTweets/6HumanPosNeg.arff", lexFilt);
		wlf.evaluateDataSet(trainLexData, "testTweets/SandersPosNeg.arff", lexFilt);
		wlf.evaluateDataSet(trainLexData, "testTweets/SemEvalPosNeg.arff", lexFilt);
		
		



	}





}
