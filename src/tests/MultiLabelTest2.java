package tests;

import it.unimi.dsi.fastutil.objects.Object2ObjectMap;
import it.unimi.dsi.fastutil.objects.Object2ObjectOpenHashMap;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.Random;

import meka.classifiers.multilabel.BCC;
import meka.classifiers.multilabel.BCCSoft;
import meka.classifiers.multilabel.BR;
import meka.classifiers.multilabel.CC;
import meka.classifiers.multilabel.CCSoft;
import meka.classifiers.multilabel.Evaluation;
import meka.classifiers.multilabel.MultilabelClassifier;
import meka.classifiers.multilabel.MyCC;
import meka.classifiers.multilabel.meta.BaggingML;
import meka.core.MLEvalUtils;
import meka.core.Result;
import weka.classifiers.functions.LibLINEAR;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.Attribute;
import weka.core.AttributeStats;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ArffSaver;
import weka.experiment.PairedStats;
import weka.experiment.PairedStatsCorrected;
import weka.filters.Filter;
import weka.filters.MultiFilter;
import weka.filters.SimpleBatchFilter;
import weka.filters.unsupervised.attribute.EmoLexExpandTestFilter;
import weka.filters.unsupervised.attribute.LexExpandTestFilter;
import weka.filters.unsupervised.attribute.RemoveByName;
import weka.filters.unsupervised.attribute.RemoveType;
import weka.filters.unsupervised.attribute.Reorder;
import weka.filters.unsupervised.attribute.StringToWordVector;
import weka.filters.unsupervised.attribute.WordCentroidMultiLabel;
import weka.filters.unsupervised.instance.RemoveWithValues;

public class MultiLabelTest2 {
	/* Path of the Dataset to be used */
	private String collectionPath;
	/* Name of the File to write results */	
	private PrintWriter out;

	
	private String outputFolder;
	private Object2ObjectMap<String,Result[]> resultStorer;
	
	
	public MultiLabelTest2(String collectionPath,String outputFolder){
		this.collectionPath=collectionPath;
		this.resultStorer=new Object2ObjectOpenHashMap<String,Result[]>();
		this.outputFolder=outputFolder;
	}
	
	
	public void setOut(PrintWriter out){
		this.out=out;
	}
	
	
	public Instances createDataSet(int m,String word2VecDic) throws Exception{
		BufferedReader reader = new BufferedReader(
				new FileReader(this.collectionPath));
		Instances train = new Instances(reader);
		reader.close();
		SimpleBatchFilter filter=new WordCentroidMultiLabel();
		
		filter.setOptions(Utils.splitOptions("-M "+m+" -N "+m+" -W -C -I 3 -P WORD- -Q CLUST- -L -K -J lexicons/NRC-emotion-lexicon-wordlevel-v0.92.txt -H resources/50mpaths2.txt -R -T resources/stopwords.txt -A -B -E -F POS- -G "+word2VecDic));
		
		filter.setInputFormat(train);
		return Filter.useFilter(train, filter);		
	}
	
		
	public void intrinsic(Instances train,Instances test,MultilabelClassifier mClass,String name,String remExp) throws Exception{	
					
		LibLINEAR ll=new LibLINEAR();
		ll.setOptions(Utils.splitOptions("-S 7 -C 1.0 -E 0.01 -B 1.0 -P"));
		
		
		RemoveByName rm=new RemoveByName();

		rm.setOptions(Utils.splitOptions(remExp));
		
		FilteredClassifier fc = new FilteredClassifier();
		fc.setFilter(rm);
		fc.setClassifier(ll);
				
		// BR classifier
		mClass.setClassifier(fc);
		mClass.buildClassifier(train);
		Result brFold[] = Evaluation.cvModel(mClass,train,10,"PCut1", "3");
				
		Result brRes = MLEvalUtils.averageResults(brFold);
		this.resultStorer.put(name, brFold);		
		out.println("===="+name+" CV Results =====");
		out.println(brRes.toString());		
		createLexicon(this.outputFolder+"/"+name+"-Lex.csv",test, mClass);
	
	}
	
	
	public void compareModels(String model1,String model2,String criteria){
		Result r1[]=this.resultStorer.get(model1);
		Result r2[]=this.resultStorer.get(model2);
		
		
		int length=r1.length;
		double[] model1Values=new double[length];
		double[] model2Values=new double[length];
		
		for(int i=0;i<length;i++){
			model1Values[i]=r1[i].output.get(criteria);
			model2Values[i]=r2[i].output.get(criteria);
		}	
		
			
		PairedStats ps=new PairedStatsCorrected(0.05,0.111);
		ps.add(model1Values, model2Values);
		ps.calculateDerived();
		
		this.out.println(criteria+"\t: "+model1+">"+model2+": Prob(differences) "+ps.differencesProbability+" Sigflag="+ps.differencesSignificance+"\n\n");
		//this.out.println(ps.toString());
	
	}
	
	
	
	// conducts an extrinsic evaluation
	public void extrinsic(String trainFile,String expLexPath) throws Exception{
		BufferedReader reader = new BufferedReader(
				new FileReader(trainFile));

			Instances corpus = new Instances(reader);
			
			EmoLexExpandTestFilter emoLexFilt=new EmoLexExpandTestFilter();
			
			emoLexFilt.setSeedPath("lexicons/NRC-emotion-lexicon-wordlevel-v0.92.txt");
			emoLexFilt.setExPath(expLexPath);
			

			StringToWordVector sToWord=new StringToWordVector();
			sToWord.setOptions(Utils.splitOptions("-R 1 -P WORD- -W 1000 -prune-rate -1.0 -N 0 -L -stemmer weka.core.stemmers.NullStemmer -M 1 -tokenizer weka.core.tokenizers.TwitterNLPTokenizer"));
			
			
			
			Reorder reod=new Reorder();
			reod.setOptions(Utils.splitOptions("-R 2-last,first"));
			
			Filter[] filters={emoLexFilt,sToWord,reod};
			
			MultiFilter mf=new MultiFilter();
			mf.setFilters(filters);
		
			
			mf.setInputFormat(corpus);
			Instances train=Filter.useFilter(corpus, mf);
			train.setClassIndex(train.numAttributes()-1);
			
			LibLINEAR ll=new LibLINEAR();
			ll.setOptions(Utils.splitOptions("-S 7 -C 1.0 -E 0.01 -B 1.0 -P"));
					
		
			
			weka.classifiers.Evaluation eval = new weka.classifiers.Evaluation(train);
			eval.crossValidateModel(ll, train, 10, new Random(1));
			
			
					
			this.out.println("===== Extrinsic SEED+EX+UNI Features using Lex:"+expLexPath+"=======");
			this.out.println(eval.toSummaryString());
			
			this.out.println(eval.toMatrixString("Confusion Matrix"));
			
			this.out.println(eval.toClassDetailsString());
			
		
			RemoveByName rm=new RemoveByName();
			rm.setOptions(Utils.splitOptions("-E EX-.*"));
			rm.setInputFormat(train);
			Instances train2=Filter.useFilter(train, rm);
			train2.setClassIndex(train2.numAttributes()-1);
			
			eval = new weka.classifiers.Evaluation(train2);
			eval.crossValidateModel(ll, train2, 10, new Random(1));
			
			this.out.println("===== Extrinsic SEED+UNI Features using Lex:"+expLexPath+"=======");
			this.out.println(eval.toSummaryString());
			
			this.out.println(eval.toMatrixString("Confusion Matrix"));
			
			this.out.println(eval.toClassDetailsString());
			
			
			
			rm=new RemoveByName();
			rm.setOptions(Utils.splitOptions("-E EX-.*|WORD-.*"));
			rm.setInputFormat(train);
			train2=Filter.useFilter(train, rm);
			train2.setClassIndex(train2.numAttributes()-1);
			
			eval = new weka.classifiers.Evaluation(train2);
			eval.crossValidateModel(ll, train2, 10, new Random(1));
			
			this.out.println("===== Extrinsic SEED Features using Lex:"+expLexPath+"=======");
			this.out.println(eval.toSummaryString());
			
			this.out.println(eval.toMatrixString("Confusion Matrix"));
			
			this.out.println(eval.toClassDetailsString());
			
			
			
			
		
	}
	
	public void extrinsicBlock(String trainFile,String name) throws Exception{
		this.extrinsic(trainFile, this.outputFolder+"/"+name+"-BR-Lex.csv");
		this.extrinsic(trainFile, this.outputFolder+"/"+name+"-CC-Lex.csv");
		this.extrinsic(trainFile, this.outputFolder+"/"+name+"-BCC-Lex.csv");
		
	}
	
	
	
	
	public static void createLexicon(String fileName,Instances wordsTest, MultilabelClassifier mc) throws Exception{
		PrintWriter pw=new PrintWriter(fileName);
		pw.println("word\tanger\tanticipation\tdisgust\tfear\tjoy\tnegative\tpositive\tsadness\tsurprise\ttrust");
		
	
		for(Instance target:wordsTest){
			String wordName=target.stringValue(wordsTest.attribute("WORD_NAME"));		
			double[] pred=mc.distributionForInstance(target);			
			String outLine=wordName+"\t";
			for(int i=0;i<pred.length;i++){
				outLine += pred[i];
				if(i<pred.length-1)
					outLine += "\t";
			}
					
			
			pw.println(outLine);
			
		}
		pw.close();
	}
	
	public void intrinsicBlock(String name,String exp,Instances wordsTrain,Instances wordsTest,String baseline) throws Exception{
		this.intrinsic(wordsTrain,wordsTest, new BR(), name+"-BR",exp);
		

		if(!baseline.isEmpty()){
			this.compareModels(baseline+"-BR", name+"-BR", "F1 micro avg");
			this.compareModels(baseline+"-BR", name+"-BR", "F1 macro avg, by lbl");
		}
		
		
		this.intrinsic(wordsTrain,wordsTest, new CCSoft(), name+"-CC",exp);
		this.compareModels(name+"-CC", name+"-BR", "F1 micro avg");
		this.compareModels(name+"-CC", name+"-BR", "F1 macro avg, by lbl");
		
		if(!baseline.isEmpty()){
			this.compareModels(baseline+"-CC", name+"-CC", "F1 micro avg");
			this.compareModels(baseline+"-CC", name+"-CC", "F1 macro avg, by lbl");
		}
			
		
		this.intrinsic(wordsTrain,wordsTest, new BCCSoft(), name+"-BCC",exp);
		this.compareModels(name+"-BCC", name+"-BR", "F1 micro avg");
		this.compareModels(name+"-BCC", name+"-BR", "F1 macro avg, by lbl");
		
		if(!baseline.isEmpty()){
			this.compareModels(baseline+"-BCC", name+"-BCC", "F1 micro avg");
			this.compareModels(baseline+"-BCC", name+"-BCC", "F1 macro avg, by lbl");
		}
		
	}
	
	
	

		
	
	
	
	// experiment.txt edimEx.arff testTweets/emoTweets.arff 10 resources/edim_lab_word2Vec.csv
	
	static public void main(String args[]) throws Exception{
		
//		BufferedReader reader = new BufferedReader(
//				new FileReader(args[0]+"/wordsTrain.arff"));
//		Instances wordsTrain = new Instances(reader);
//		reader.close();
		
//		for (int i=0;i<10;i++){
//			System.out.println(wordsTrain.attribute(i).name()+" "+Arrays.toString(wordsTrain.attributeStats(i).nominalCounts));
//		}
//		
//		System.out.println("NumAttributes "+wordsTrain.numAttributes());
//
//		System.out.println("NumInstances "+wordsTrain.numInstances());
		
//		wordsTrain.setClassIndex(10);
//		
//
//		reader = new BufferedReader(
//				new FileReader(args[0]+"/wordsTest.arff"));
//		Instances wordsTest = new Instances(reader);
//		reader.close();
//		
//		wordsTest.setClassIndex(10);
		
	
		PrintWriter pw=new PrintWriter(args[0]+"/reportExtrinsicUni.txt");
		
		MultiLabelTest2 mlt=new MultiLabelTest2(args[1],args[0]);
		
		mlt.setOut(pw);
// 
//				
//		
//		mlt.intrinsicBlock("uni", "-E WORD_NAME|Word2Vec-.*|Est-.*|CLUST-.*|POS-.*", wordsTrain, wordsTest,"");

//
//		mlt.intrinsicBlock("uni-bwn", "-E WORD_NAME|Word2Vec-.*|Est-.*|POS-.*", wordsTrain, wordsTest,"uni");
//		mlt.intrinsicBlock("uni-bwn-pos", "-E WORD_NAME|Word2Vec-.*|Est-.*", wordsTrain, wordsTest,"uni");
//		mlt.intrinsicBlock("uni-bwn-pos-dp", "-E WORD_NAME|Word2Vec-.*", wordsTrain, wordsTest,"uni");
//		mlt.intrinsicBlock("uni-bwn-pos-dp-w2v", "-E WORD_NAME", wordsTrain, wordsTest,"uni");
//		
//		
//		mlt.intrinsicBlock("w2v", "-E WORD_NAME|WORD-.*|Est-.*|CLUST-.*|POS-.*", wordsTrain, wordsTest,"uni");
//		mlt.intrinsicBlock("w2v-bwn", "-E WORD_NAME|WORD-.*|Est-.*|POS-.*", wordsTrain, wordsTest,"uni");
//		mlt.intrinsicBlock("w2v-bwn-pos", "-E WORD_NAME|WORD-.*|Est-.*", wordsTrain, wordsTest,"uni");
//		mlt.intrinsicBlock("w2v-bwn-pos-dp", "-E WORD_NAME|WORD-.*", wordsTrain, wordsTest,"uni");
		
		
		
		mlt.extrinsicBlock(args[2], "w2v");
//		mlt.extrinsicBlock(args[2], "uni-bwn");
//		mlt.extrinsicBlock(args[2], "uni-bwn-pos");
//		mlt.extrinsicBlock(args[2], "uni-bwn-pos-dp");
//		mlt.extrinsicBlock(args[2], "uni-bwn-pos-dp-w2v");
//		
//		mlt.extrinsicBlock(args[2], "w2v");
//		mlt.extrinsicBlock(args[2], "w2v-bwn");
//		mlt.extrinsicBlock(args[2], "w2v-bwn-pos");
//		mlt.extrinsicBlock(args[2], "w2v-bwn-pos-dp");
//		
		
				
		
		

		
		
			

		
		
				
						
		
		pw.close();
		

	}
	
	




}
