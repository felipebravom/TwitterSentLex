package tests;

import it.unimi.dsi.fastutil.objects.AbstractObjectSet;
import it.unimi.dsi.fastutil.objects.Object2IntMap;
import it.unimi.dsi.fastutil.objects.Object2ObjectMap;
import it.unimi.dsi.fastutil.objects.Object2ObjectOpenHashMap;
import it.unimi.dsi.fastutil.objects.ObjectList;
import it.unimi.dsi.fastutil.objects.ObjectOpenHashSet;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;

import lexexpand.core.LexiconEvaluator;
import lexexpand.core.MyUtils;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

public class PMISOLex {

	/*  Computes the PMI-SO score for each word in a corpus of labelled tweets					
	 * 
	 * 	double posProb=wordPosCount[i]/posCount;
	 * 	double negProb=wordNegCount[i]/negCount;
	 * 	semanticOrientation[i]=logOfBase(2,posProb)-logOfBase(2,negProb);
	 * 						
	 * */

	String word; // the word
	int posCount; // number of tweets in each partition
	int negCount; // number of documents in the current partition		

	public PMISOLex(String word){
		this.word=word;
		this.posCount=1;
		this.negCount=1;


	}

	void addPos(){
		this.posCount++;
	}


	void addNeg(){
		this.negCount++;
	}


	static public double logOfBase(int base, double num) {
		return Math.log(num) / Math.log(base);
	}


	static public void main(String args[]) throws IOException{

		String inputFile="testTweets/SemEvalPosNeg.arff";

		BufferedReader reader = new BufferedReader(
				new FileReader(inputFile));
		Instances inp = new Instances(reader);
		reader.close();

		LexiconEvaluator lex= new LexiconEvaluator("lexicons/AFINN-posneg.txt");
		lex.processDict();

		double posCount=1.0;
		double negCount=1.0;

		Object2ObjectMap<String, PMISOLex> wordCounts=new Object2ObjectOpenHashMap<String, PMISOLex>();

		Attribute contentAtt=inp.attribute("content");
		Attribute attClassInp=inp.attribute("Class");

		for(Instance inst:inp){
			String content=inst.stringValue(contentAtt);
			content=content.toLowerCase();
			AbstractObjectSet<String> terms=new  ObjectOpenHashSet<String>(); 
			terms.addAll(MyUtils.cleanTokenize(content));


			String classValue=attClassInp.value((int)inst.value(attClassInp));


			boolean isPos=(classValue.equals("positive"))?true:false;
			if(isPos){
				posCount++;
			}

			else{
				negCount++;
			}



			for(String word:terms){
				if(lex.getDict().containsKey(word)){
					if(wordCounts.containsKey(word)){
						PMISOLex wc=wordCounts.get(word);
						if(isPos)
							wc.addPos();
						else
							wc.addNeg();
					}
					else{
						PMISOLex wc=new PMISOLex(word);
						if(isPos)
							wc.addPos();
						else
							wc.addNeg();

						wordCounts.put(word,wc);
					}
				}
			}
		}	

		String[] sortedWords=wordCounts.keySet().toArray(new String[0]);

		Arrays.sort(sortedWords);

		for(String word:sortedWords){
			PMISOLex wordCount=wordCounts.get(word);

			double posProb=wordCount.posCount/posCount;
			double negProb=wordCount.negCount/negCount;
			double semanticOrientation=logOfBase(2,posProb)-logOfBase(2,negProb);


			System.out.println(word+"\t"+semanticOrientation+"\t"+lex.retrieveValue(word));
		}




	}


}
