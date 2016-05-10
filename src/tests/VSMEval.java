package tests;

import it.unimi.dsi.fastutil.objects.ObjectOpenHashSet;
import it.unimi.dsi.fastutil.objects.ObjectSet;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.PrintWriter;
import java.util.List;
import java.util.Map;
import java.util.Random;

import lexexpand.core.ExpandLexEvaluator;
import lexexpand.core.LexiconEvaluator;
import cmu.arktweetnlp.Twokenize;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.filters.Filter;
import weka.filters.MultiFilter;
import weka.filters.unsupervised.attribute.LexExpandTestFilter;
import weka.filters.unsupervised.attribute.RemoveByName;
import weka.filters.unsupervised.attribute.Reorder;
import weka.filters.unsupervised.attribute.TextToVSM;
import weka.filters.unsupervised.attribute.WordPMI;
import weka.filters.unsupervised.attribute.WordVSMCompactFast;
import weka.filters.unsupervised.instance.RemoveWithValues;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.LibLINEAR;
import weka.classifiers.meta.FilteredClassifier;

// Evaluates VSM and PMI word using different Endiburgh datasets
// Trains a logistic regression over the labelled words

public class VSMEval {
	
	private String path; // input Path
	private String lexPath; // Path of destination Lex
	private TextToVSM textToVsm;  // the TextToVSM Filter to be used 
	private int option; // defines the type of Filter to be used

	public VSMEval(String path, int option, String lexPath){
		this.path=path;	
		this.option=option;			
		this.lexPath=lexPath;
	}
	
	// evaluated 
	public void evalWordLevel() throws Exception{
		
		BufferedReader reader = new BufferedReader(
				new FileReader(this.path));

		System.out.println(" ,"+this.path);
		


		Instances train = new Instances(reader);
		reader.close();
		
		long startTime=System.nanoTime();
		
		if(this.option==1){
			this.textToVsm=new WordVSMCompactFast();
			System.out.println(" ,WordVSMCompactFact");
		}
			
		else{
			this.textToVsm=new WordPMI();
			System.out.println(", WordPMI");
		}
			
		
		this.textToVsm.setOptions(Utils.splitOptions("-M 10 -I 3 -L -D -P WORD-"));
		this.textToVsm.setInputFormat(train);
		Instances wordVSM=Filter.useFilter(train, this.textToVsm);
		long timewordVSMCompactFast=System.nanoTime()-startTime;	

		System.out.println("Filter Time Minutes,"+timewordVSMCompactFast/(Math.pow(10, 9)*60));

		wordVSM.setClassIndex(wordVSM.numAttributes()-1);
		
		System.out.println("Number of attributes,"+(wordVSM.numAttributes()-1));

		// Creates a 3-class dataset of labelled words
		RemoveWithValues f=new RemoveWithValues();
		f.setOptions(Utils.splitOptions("-S 0.0 -C last -L first-last -V -M"));
		f.setInputFormat(wordVSM);
		Instances word3VSMTrain=Filter.useFilter(wordVSM, f);
		System.out.println("avgNumValues,"+avgsValues(word3VSMTrain));
		
		// The unlabelled words to be used for testing
		RemoveWithValues h=new RemoveWithValues();
		h.setOptions(Utils.splitOptions("-S 0.0 -C last -L first-last"));
		h.setInputFormat(wordVSM);
		Instances word3VSMTest=Filter.useFilter(wordVSM, h);
		
		

		// This filter removes neutral words to evaluate a polarity classifier.
		RemoveWithValues g=new RemoveWithValues();
		g.setOptions(Utils.splitOptions("-S 0.0 -C last -L 2 -H"));
		g.setInputFormat(word3VSMTrain);
		Instances word2VSMTrain=Filter.useFilter(word3VSMTrain, g);
		
		
		word2VSMTrain.setClassIndex(word2VSMTrain.numAttributes()-1);



		//weka.filters.unsupervised.instance.RemoveWithValues 


		// Trains a LibLinear classifier using the FilteredClassifier
		RemoveByName rbn=new RemoveByName();
		rbn.setOptions(Utils.splitOptions("-E WORD_NAME|NUM_DOCS"));
		LibLINEAR ll=new LibLINEAR();
		ll.setOptions(Utils.splitOptions("-S 7 -C 1.0 -E 0.01 -B 1.0 -P"));
		FilteredClassifier fc = new FilteredClassifier();
		fc.setFilter(rbn);
		fc.setClassifier(ll);


		Evaluation eval2Class = new Evaluation(word2VSMTrain);
		eval2Class.crossValidateModel(fc, word2VSMTrain, 10, new Random(1));

		System.out.println("Train Instances 2class,"+eval2Class.numInstances());
		System.out.println("Accuracy 2class,"+eval2Class.pctCorrect());
		System.out.println("Weighted AUC 2class,"+eval2Class.weightedAreaUnderROC());
		System.out.println("Kappa 2class,"+eval2Class.kappa());
	
		
		
		Evaluation eval3Class = new Evaluation(word3VSMTrain);
		eval3Class.crossValidateModel(fc, word3VSMTrain, 10, new Random(1));
		
	

		System.out.println("Train Instances 3class,"+eval3Class.numInstances());
		System.out.println("Accuracy 3class,"+eval3Class.pctCorrect());
		System.out.println("Weighted AUC 3class,"+eval3Class.weightedAreaUnderROC());
		System.out.println("Kappa 3class,"+eval3Class.kappa());

		


		// Train 3-class word-level classifier and use it to classify unlabelled words
		fc.buildClassifier(word3VSMTrain);
		PrintWriter pw=new PrintWriter(this.lexPath);
		pw.println("word\tnumDocs\tlabel\tnegative\tneutral\tpositive");
		
		for(Instance target:word3VSMTest){
			String wordName=target.stringValue(word3VSMTrain.numAttributes()-2);
			double numDocs=target.value(word3VSMTrain.numAttributes()-3);
		
			double[] pred=fc.distributionForInstance(target);
			String label=wordLabel(pred);
			
			pw.println(wordName+"\t"+numDocs+"\t"+label+"\t"+pred[0]+"\t"+pred[1]+"\t"+pred[2]);
			
		}
		pw.close();

		
		
	}
	
	
	// This function returns the average number of active attributes
	public static double avgsValues(Instances insts){
		double count=0;
		for(Instance inst:insts){
			count += inst.numValues();			
		}
		return count/insts.numInstances();

	}
	
	// gets the word label given the distribution for instance
	public static String wordLabel(double[] values){
		int max=0;
		double tmpVal=values[0];
		for(int i=1;i<values.length;i++){
			if(values[i]>tmpVal){
				max=i;
				tmpVal=values[i];
								
			}
		}
		if(max==0)
			return "negative";
		else if(max==1)
			return "neutral";
		else
			return "positive";
	}

	
	public void evalMessage(String dataPath,String exLexPath) throws Exception{
		
		BufferedReader reader = new BufferedReader(
				new FileReader(dataPath));

			Instances corpus = new Instances(reader);
			
			LexExpandTestFilter lexFilt=new LexExpandTestFilter();
			
			lexFilt.setSeedPath("lexicons/metaLexEmo.csv");
			lexFilt.setExPath(exLexPath);
			
			int bestPosThres=0;
			int bestNegThres=0;
			double bestKappa=-100;
			Evaluation bestEval=null;
						
			
			for(int i=10;i<=50;i=i+10){
				for(int j=10;j<=50;j=j+10){					
					
					lexFilt.setPosThres(i);
					lexFilt.setNegThres(j);
					
					Reorder reod=new Reorder();
					reod.setOptions(Utils.splitOptions("-R 3-last,2"));
					
					Filter[] filters={lexFilt,reod};
					
					MultiFilter mf=new MultiFilter();
					mf.setFilters(filters);
				
					
					mf.setInputFormat(corpus);
					Instances train=Filter.useFilter(corpus, mf);
					train.setClassIndex(train.numAttributes()-1);
					
					LibLINEAR ll=new LibLINEAR();
					ll.setOptions(Utils.splitOptions("-S 7 -C 1.0 -E 0.01 -B 1.0 -P"));
				

					Evaluation eval = new Evaluation(train);
					eval.crossValidateModel(ll, train, 10, new Random(1));
					
					if(eval.kappa()>bestKappa){
						bestKappa=eval.kappa();
						bestPosThres=i;
						bestNegThres=j;
						bestEval=eval;						
					}
				
				}			
				
			}

			System.out.println(dataPath+":Best Pos Thres,"+bestPosThres);
			System.out.println(dataPath+":Best Neg Thres,"+bestNegThres);
			
			System.out.println(dataPath+":Train Instances message,"+bestEval.numInstances());
			System.out.println(dataPath+":Train Instances message,"+bestEval.numInstances());
			System.out.println(dataPath+":Accuracy message,"+bestEval.pctCorrect());
			System.out.println(dataPath+":Weighted AUC message,"+bestEval.weightedAreaUnderROC());
			System.out.println(dataPath+":kappa message,"+bestEval.kappa());
			
			
	
			reader.close();
		
		
	}
	

	// compares the words between a lexicon and collection of tweets
	public void compWordsLexCol(String dataPath,String exLexPath) throws Exception{
		BufferedReader reader = new BufferedReader(
				new FileReader(dataPath));

			Instances corpus = new Instances(reader);
			
			
			ExpandLexEvaluator metaLex = new ExpandLexEvaluator(
					exLexPath);
			metaLex.processDict();
			

			// This HashSet contains the vocabulary of the corpus
			ObjectSet<String> terms= new ObjectOpenHashSet<String>();
			
			double posCount=0;
			double negCount=0;
			
			for(Instance inst:corpus){
							
				String content = inst.stringValue(0);
				content=content.toLowerCase();
				// tokenises the content 
				List<String> tokens=Twokenize.tokenizeRawTweetText(content); 
				terms.addAll(tokens);
				
				Map<String,Double> vals=metaLex.evaluatePolarity(tokens);
				posCount += vals.get("posCount");
				negCount += vals.get("negCount");
		
			}
			
		
			
			System.out.println(dataPath+":Collection Voc size,"+terms.size());
					
		

			System.out.println(dataPath+":Avg PosWord per Tweet,"+posCount/corpus.size());
			System.out.println(dataPath+":Avg NegWord per Tweet,"+negCount/corpus.size());
			
			
			
			int lexSize=metaLex.getDict().keySet().size();
			System.out.println(dataPath+":Lex Size,"+lexSize);
			
			int countPos=0;
			int countNeu=0;
			int countNeg=0;
			
			int match=0;
			int matchPos=0;
			int matchNeu=0;
			int matchNeg=0;
			
			for(String word:metaLex.getDict().keySet()){
				String wordLabel=metaLex.getDict().get(word).get("label");
				if(wordLabel.equals("positive"))
					countPos++;
				else if(wordLabel.equals("neutral"))
					countNeu++;
				else
					countNeg++;
				
				if(terms.contains(word)){
					match++;
					
					if(wordLabel.equals("positive"))
						matchPos++;
					else if(wordLabel.equals("neutral"))
						matchNeu++;
					else
						matchNeg++;
					
					
					
				}
					
			}
			
			System.out.println(dataPath+":Lex PosWords,"+countPos);
			System.out.println(dataPath+":Lex NeuWords,"+countNeu);
			System.out.println(dataPath+":Lex NegWords,"+countNeg);
			
			
			System.out.println(dataPath+":Matching words,"+match);
			System.out.println(dataPath+":Matching Poswords,"+matchPos);
			System.out.println(dataPath+":Matching Neuwords,"+matchNeu);
			System.out.println(dataPath+":Matching Negwords,"+matchNeg);
			
			
		
	}
	
	


	// Evaluates and creates a lexicon
	// args[0] collection of raw tweets
	// args[1] option for Filter type
	// args[2] File to Store generated Lexicon
	static public void main(String args[]) throws Exception {
		
		
		System.out.println("---------------------------------");
		VSMEval vsmEv=new VSMEval(args[0],Integer.parseInt(args[1]),args[2]);
		vsmEv.evalWordLevel();
		vsmEv.compWordsLexCol("testTweets/SemEvalPosNeg.arff", args[2]);
		vsmEv.evalMessage("testTweets/SemEvalPosNeg.arff",args[2]);
		
		vsmEv.compWordsLexCol("testTweets/SandersPosNeg.arff", args[2]);
		vsmEv.evalMessage("testTweets/SandersPosNeg.arff",args[2]);
		

		vsmEv.compWordsLexCol("testTweets/6HumanPosNeg.arff", args[2]);
		vsmEv.evalMessage("testTweets/6HumanPosNeg.arff",args[2]);
		
		
//		vsmEv.compWordsLexCol("testTweets/sent140.arff", args[2]);
//		vsmEv.evalMessage("testTweets/sent140.arff",args[2]);		
		
		System.out.println("---------------------------------");


	}

}
