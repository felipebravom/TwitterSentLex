package tests;

import java.beans.IntrospectionException;
import java.beans.PropertyDescriptor;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;

import javax.swing.DefaultListModel;

import meka.classifiers.multilabel.BCC;
import meka.classifiers.multilabel.BR;
import meka.classifiers.multilabel.CC;
import meka.classifiers.multilabel.Evaluation;
import meka.classifiers.multilabel.MultilabelClassifier;
import meka.core.MLEvalUtils;
import meka.core.Result;
import meka.experiment.MekaClassifierSplitEvaluator;
import meka.experiment.MekaCrossValidationSplitResultProducer;
import meka.experiment.MekaExperiment;
import weka.classifiers.Classifier;
import weka.core.Instances;
import weka.core.Range;
import weka.experiment.InstancesResultListener;
import weka.experiment.PairedCorrectedTTester;
import weka.experiment.PairedStats;
import weka.experiment.PairedStatsCorrected;
import weka.experiment.PairedTTester;
import weka.experiment.PropertyNode;
import weka.experiment.ResultMatrixPlainText;
import weka.experiment.SplitEvaluator;

public class PairedTestExample {
	
	static public void main (String args[]) throws Exception{
		
		
		exploreFolds();
		

		
	}
	
	  /**
	   * A significance indicator:
	   * 0 if the differences are not significant
	   * > 0 if r1 significantly greater than r2
	   * < 0 if r1 significantly less than r2
	   * metric=	 F1 micro avg","F1 macro avg, by lbl"
	   * 
	   *  
	   *  */	
	static public int compare(Result r1[],Result[] r2,String metric){
		int length=r1.length;
		double[] first=new double[length];
		double[] second=new double[length];
		
		for(int i=0;i<length;i++){
			first[i]=r1[1].output.get(metric);
			second[i]=r2[1].output.get(metric);
		}		
		
		PairedStats ps=new PairedStatsCorrected(0.05,0.111);
		ps.add(first, second);
		ps.calculateDerived();
		
		return(ps.differencesSignificance);		
		
		
	}
	
	static public void exploreFolds() throws Exception{
		
		BufferedReader reader = new BufferedReader(
				new FileReader("/Users/admin/meka-1.7.6/data/music.arff"));
		Instances train = new Instances(reader);
		train.setClassIndex(6);
		
				
		// BR classifier
		MultilabelClassifier br=new BR();
		br.buildClassifier(train);
		Result brFold[] = Evaluation.cvModel(br,train,10,"PCut1", "3");
		
		// BR classifier
		MultilabelClassifier cc=new CC();
		cc.buildClassifier(train);
		Result ccFold[] = Evaluation.cvModel(cc,train,10,"PCut1", "3");
	
		
		
		int sigF1micro=compare(brFold,ccFold,"F1 micro avg");
		int sigF1macro=compare(brFold,ccFold,"F1 macro avg, by lbl");
		
		System.out.println("Sig F1 micro:"+sigF1micro);
		System.out.println("Sig F1 macro:"+sigF1macro);
		
		
		
//		for(Result brr:brFold){
//			for(String metric:brr.vals.keySet()){
//				System.out.println(metric+":"+brr.vals.get(metric));
//			}
//			for(String metric:brr.output.keySet()){
//				System.out.println(metric+":"+brr.output.get(metric));
//			}
//			System.out.println();
//			
//		}
		
	//	Result brRes = MLEvalUtils.averageResults(brFold);

		
		
	//	System.out.println(brRes.toString());
	}
	
	


}
