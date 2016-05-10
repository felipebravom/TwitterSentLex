package tests;

import it.unimi.dsi.fastutil.objects.AbstractObjectSet;
import it.unimi.dsi.fastutil.objects.Object2ObjectMap;
import it.unimi.dsi.fastutil.objects.Object2ObjectOpenHashMap;
import it.unimi.dsi.fastutil.objects.ObjectOpenHashSet;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import lexexpand.core.LexiconEvaluator;
import cmu.arktweetnlp.Twokenize;
import weka.core.Instance;
import weka.core.Instances;

public class WordTweetStats {
	String name;
	String label;
	int posCount;
	int negCount;

	public WordTweetStats(String name,String label){
		this.name=name;
		this.label=label;
		this.posCount=0;
		this.negCount=0;
	}

	public void addPos(){
		this.posCount++;
	}

	public void addNeg(){
		this.negCount++;
	}


	public String toString(){
		return name+"\t"+label+"\t"+posCount+"\t"+negCount;
	}




	public static List<String> tokenize(String content) {

		content=content.toLowerCase();

		content = content.replaceAll("([a-z])\\1+", "$1$1");

		List<String> tokens = new ArrayList<String>();

		for (String word : Twokenize.tokenizeRawTweetText(content)) {
			String cleanWord = word;


			// Replace URLs to a generic URL
			if (word.matches("http.*|ww\\..*")) {
				cleanWord = "http://www.url.com";
			}

			// Replaces user mentions to a generic user
			else if (word.matches("@.*")) {
				cleanWord = "@user";
			}


			tokens.add(cleanWord);
		}
		return tokens;
	}


	public static void main(String args[]) throws Exception{
		String inputFile="testTweets/SemEvalPosNeg.arff";

		String lexPath="lexicons/AFINN-posneg.txt";

		LexiconEvaluator lex=new LexiconEvaluator(lexPath);
		lex.processDict();

		Object2ObjectMap<String,WordTweetStats> wordStats=new Object2ObjectOpenHashMap<String,WordTweetStats>();



		System.out.println("testing");

		BufferedReader reader = new BufferedReader(
				new FileReader(inputFile));
		Instances dataset = new Instances(reader);
		reader.close();
		
		PrintWriter pwTweet=new PrintWriter("tweetData.csv");
		pwTweet.println("tweet\tlabel\tposWords\tnegWords\ttotWords");

		for(Instance inst:dataset){
			System.out.println(inst.toString());
			String content=inst.stringValue(0);
			double sentClass=inst.value(1); // 0.0 means positive, 1.0 means negative

			List<String> words=tokenize(content);


			Map<String,Integer> sentVals=lex.evaluatePolarityLexicon(words);
			int posCount=sentVals.get("posCount");
			int negCount=sentVals.get("negCount");

			pwTweet.println(content+"\t"+(sentClass==0.0?"positive":"negative")+"\t"+posCount+"\t"+negCount+"\t"+words.size());
			

			// Identifies the distinct terms
			AbstractObjectSet<String> terms=new  ObjectOpenHashSet<String>(); 
			terms.addAll(words);

			for(String word:terms){
				String wordPol=lex.retrieveValue(word);
				if(!wordPol.equals("not_found")){


					if(!wordStats.containsKey(word)){
						WordTweetStats ws=new WordTweetStats(word,wordPol);
						if(sentClass==0.0)
							ws.addPos();
						else
							ws.addNeg();	
						
						wordStats.put(word, ws);
						
					}
					else{
						WordTweetStats ws=wordStats.get(word);
						if(sentClass==0.0)
							ws.addPos();
						else
							ws.addNeg();	

					}	

				}

			}





		}
		
		pwTweet.close();

		PrintWriter pwWord=new PrintWriter("wordData.csv");
		pwWord.println("name\tlabel\tposCount\tnegCount");
		
		for(WordTweetStats wt:wordStats.values()){
			pwWord.println(wt.toString());			
		}
		pwWord.close();


	}

}
