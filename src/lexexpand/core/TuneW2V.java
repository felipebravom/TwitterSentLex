package lexexpand.core;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.PrintWriter;
import java.util.TreeMap;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import meka.classifiers.multilabel.BR;
import meka.classifiers.multilabel.Evaluation;
import meka.core.MLEvalUtils;
import meka.core.Result;
import weka.classifiers.functions.LibLINEAR;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.Instances;
import weka.core.Utils;
import weka.filters.unsupervised.attribute.RemoveByName;


public class TuneW2V {
	
	public static String evaluateDataSet(File f) throws Exception{
		
		BufferedReader reader = new BufferedReader(new FileReader(f));
		Instances train = new Instances(reader);
		reader.close();
		train.setClassIndex(10);
		
		LibLINEAR ll=new LibLINEAR();
		ll.setOptions(Utils.splitOptions("-S 7 -C 1.0 -E 0.01 -B 1.0 -P"));
		
		
		RemoveByName rm=new RemoveByName();

		rm.setOptions(Utils.splitOptions("-E word_name"));
		
		FilteredClassifier fc = new FilteredClassifier();
		fc.setFilter(rm);
		fc.setClassifier(ll);
				
		BR mClass=new BR();
		// BR classifier
		mClass.setClassifier(fc);
		mClass.buildClassifier(train);
		Result brFold[] = Evaluation.cvModel(mClass,train,2,"PCut1", "3");
				
		Result brRes = MLEvalUtils.averageResults(brFold);
		
		String f1micro=brRes.info.get("F1 micro avg").substring(0,5);
		String f1macro=brRes.info.get("F1 macro avg, by lbl").substring(0,5);
		
		//double f1=brRes.output.get("F1 micro avg");
		
		return f1micro+"\t"+f1macro;
		
	}
	
	
	
	static public void main(String args[]) throws Exception{
		String directoryName = "/Users/admin/workspace/TwitterSentLex/scripts/tuning_emb";
		TreeMap<Integer,TreeMap<Integer,File>> results=new TreeMap<Integer,TreeMap<Integer,File>>();
		
		
		
		Pattern p = Pattern.compile("\\d+");
		
		File targetFolder=new File(directoryName);
		if(targetFolder.isDirectory()){
			for(File file:targetFolder.listFiles()){
				
				Matcher m = p.matcher(file.getName());
				m.find();
				int n=Integer.parseInt(m.group());
				m.find();
				int w=Integer.parseInt(m.group());
				
				if(results.containsKey(n)){
					results.get(n).put(w, file);
				}
				else{
					TreeMap<Integer,File> fileMapper=new TreeMap<Integer,File>();
					fileMapper.put(w, file);
					results.put(n, fileMapper);
				}
				  
//				while (m.find()) {
//				  System.out.println(m.group());
//				  
//				}
				
				
			//	System.out.println(file.getName()+" "+n+" "+w);
				//evaluateDataSet(file);
				
			}
			
			PrintWriter pw=new PrintWriter("tuneResults.csv");
			pw.println("n\tw\tF1micro\tF1Macro");
			
			for(int n:results.keySet()){
				for(int w:results.get(n).keySet()){
					File f=results.get(n).get(w);
					String f1=evaluateDataSet(f);
					pw.println(n+"\t"+w+"\t"+f1);
				}				
			}
			pw.close();
			
			
			
		}
		
	}


}
