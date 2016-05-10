package lexexpand.core;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.PrintWriter;

import weka.core.Instance;
import weka.core.Instances;

public class ArffCorpusToTxt {
	// converts 
	static public void main(String args[]) throws Exception{
		String inputFile=args[0];
		BufferedReader reader = new BufferedReader(
				new FileReader(inputFile));
		Instances train = new Instances(reader);
		
		PrintWriter pw=new PrintWriter(args[1]);
		
		for(Instance inst:train){
			String content = inst.stringValue(train.attribute("text"));
			pw.println(content.toLowerCase());
		}
		pw.close();
		
	}
	

}
