package lexexpand.core;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Iterator;
import java.util.List;

import cmu.arktweetnlp.Twokenize;

import com.cybozu.labs.langdetect.Detector;
import com.cybozu.labs.langdetect.DetectorFactory;
import com.cybozu.labs.langdetect.LangDetectException;


// Creates and ArffFile from the EdinBurgh corpus discarding all tweets not written in English. 

public class EdinburghToArff {

	static public void main(String args[]) throws LangDetectException, IOException{

		DetectorFactory.loadProfile("profiles/");

		//String path="/Users/admin/Datasets/twitterBifet/edim100.txt";
		String inputPath=args[0];
		BufferedReader bf=new BufferedReader(new FileReader(inputPath));
		
		String outputPath=args[1];
		
		
		PrintWriter pw=new PrintWriter(new FileWriter(outputPath));


		pw.println("@relation 'Edinburgh Corpus English'\n\n"
				+ "@attribute date string\n"
				+ "@attribute user string\n"
				+ "@attribute content string\n\n"
				+"@data");

		String line;
		while((line=bf.readLine())!=null){

			try{
				String parts[]=line.split("\t");
				String date=parts[0];
				String user=parts[1];
				String content=parts[2];
			
				Detector detector= DetectorFactory.create();
				detector.append(content);
				String lang=detector.detect();
				if(lang.equals("en")){

					String cleanContent=content.replaceAll("\'", "");;

//					List<String> cleanWords=MyUtils.cleanTokenize(content);
//					
//					Iterator<String> iter = cleanWords.iterator();
//					while(iter.hasNext()){
//						cleanContent += iter.next();
//						if(iter.hasNext())
//							cleanContent += " ";					
//					
//					}
//					
//					// non unicode characters are converted into quotes by the tokenizer
//					cleanContent=cleanContent.replaceAll("\'", "");
					
					

					pw.println("'"+date+"','"+user+"','"+cleanContent+"'");
				}




			}
			catch(Exception e){
				continue;
			}

		}

		bf.close();
		pw.close();
		











	}

}
