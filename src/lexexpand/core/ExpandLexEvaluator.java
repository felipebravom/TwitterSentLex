package lexexpand.core;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import cmu.arktweetnlp.Tagger;
import cmu.arktweetnlp.Tagger.TaggedToken;


/**
 * 
 * @author fbravo Evaluates a generic lexicon where the first column is the word and 
 * 	the other are features of the lexicon
 *  
 */
public class ExpandLexEvaluator {

	protected String path;
	protected Map<String, Map<String,String>> dict; // entry and a metadata map

	public ExpandLexEvaluator(String file) {
		this.dict = new HashMap<String, Map<String,String>>();
		this.path = file;

	}

	public void processDict() throws IOException {
		// first, we open the file
		BufferedReader bf = new BufferedReader(new FileReader(this.path));

		String firstLine=bf.readLine();
		String fieldNames[] = firstLine.split("\t");

		String line;
		while ((line = bf.readLine()) != null) {
			Map<String,String> entry=new HashMap<String,String>();

			String pair[] = line.split("\t");
			String word=pair[0];
			for(int i=1;i<pair.length;i++)
				entry.put(fieldNames[i],pair[i]);


			this.dict.put(word, entry);

		}
		bf.close();

	}

	// returns the score associated to a word
	public Map<String,String> retrieveValue(String word) {
		if (!this.dict.containsKey(word)) {
			return null;
		} else {
			return this.dict.get(word);
		}

	}

	public Map<String, Map<String,String>> getDict() {
		return this.dict;
	}


	// returns a positive and negative score for a tagged list of Strings
	public Map<String, Double> evaluatePolarity(List<String> tagTokens) {

		Map<String, Double> sentCount = new HashMap<String, Double>();

		double negScore = 0.0;
		double posScore = 0.0;

		double negCount = 0.0;
		double posCount = 0.0;

		for (String w : tagTokens) {
			if(this.dict.containsKey(w)){
				Map<String,String> pol = this.retrieveValue(w);
				if (pol.get("label").equals("positive")) {
					posScore += Double.parseDouble(pol.get("positive")) ;
					posCount++;
				} else if (pol.get("label").equals("negative")) {
					negScore += Double.parseDouble(pol.get("negative")) ;
					negCount++;
				}

			}

		}

		sentCount.put("posScore", posScore);
		sentCount.put("negScore", negScore);

		sentCount.put("posCount", posCount);
		sentCount.put("negCount", negCount);


		return sentCount;
	}
	
	
	
	// returns a positive and negative scores for a tagged list of Strings without considering labels
		public Map<String, Double> getPosNegScores(List<String> tagTokens) {

			Map<String, Double> sentCount = new HashMap<String, Double>();

			double negScore = 0.0;
			double posScore = 0.0;


			for (String w : tagTokens) {
				if(this.dict.containsKey(w)){
					Map<String,String> pol = this.retrieveValue(w);
					posScore += Double.parseDouble(pol.get("positive")) ;
					negScore += Double.parseDouble(pol.get("negative")) ;
					
					}

				}

			

			sentCount.put("posScore", posScore);
			sentCount.put("negScore", negScore);

			return sentCount;
		}
	
	
	// returns a positive and negative score for a tagged list of Strings
	public Map<String, Double> evaluateEmotion(List<String> tokens) {
		
		
		Map<String, Double> emoCount = new HashMap<String, Double>();

		double anger = 0.0;
		double anticipation = 0.0;
		double disgust = 0.0;
		double fear = 0.0;
		double joy = 0.0;
		double sadness = 0.0;
		double surprise = 0.0;
		double trust = 0.0;
		double negative = 0.0;
		double positive = 0.0;

		for (String word : tokens) {
			// I retrieve the EmotionMap if the word match the lexicon
			if (this.getDict().containsKey(word)) {
				Map<String, String> emotions = this.getDict().get(word);
				anger += Double.parseDouble(emotions.get("anger"));
				anticipation += Double.parseDouble(emotions.get("anticipation"));
				disgust += Double.parseDouble(emotions.get("disgust"));
				fear += Double.parseDouble(emotions.get("fear"));
				joy += Double.parseDouble(emotions.get("joy"));
				sadness += Double.parseDouble(emotions.get("sadness"));
				surprise += Double.parseDouble(emotions.get("surprise"));
				trust += Double.parseDouble(emotions.get("trust"));
				negative += Double.parseDouble(emotions.get("negative"));
				positive += Double.parseDouble(emotions.get("positive"));

			}
		}

		emoCount.put("anger", anger);
		emoCount.put("anticipation", anticipation);
		emoCount.put("disgust", disgust);
		emoCount.put("fear", fear);
		emoCount.put("joy", joy);
		emoCount.put("sadness", sadness);
		emoCount.put("surprise", surprise);
		emoCount.put("trust", trust);
		emoCount.put("negative", negative);
		emoCount.put("positive", positive);

		
		
		return emoCount;
	}



	// returns a positive and negative score for a list of tokens using a threshold for the number of documents
	public Map<String, Double> evaluatePolThres(List<String> tokens, int thresPos, int thresNeg) {

		Map<String, Double> sentCount = new HashMap<String, Double>();

		double negScore = 0.0;
		double posScore = 0.0;

		double negCount = 0.0;
		double posCount = 0.0;

		for (String w : tokens) {
			if(this.dict.containsKey(w)){
				Map<String,String> pol = this.retrieveValue(w);

				double numDocs=Double.parseDouble(pol.get("numDocs"));

				

				if (pol.get("label").equals("positive")) {
					// Just include the words that appeared more than the positive threshold
					if(numDocs>=thresPos){
						posScore += Double.parseDouble(pol.get("positive")) ;
						posCount++;						
					}

				} else if (pol.get("label").equals("negative")) {
					if(numDocs>=thresNeg){
						negScore += Double.parseDouble(pol.get("negative")) ;
						negCount++;						
					}

				}





			}

		}

		sentCount.put("posScore", posScore);
		sentCount.put("negScore", negScore);

		sentCount.put("posCount", posCount);
		sentCount.put("negCount", negCount);


		return sentCount;
	}






	static public void main(String args[]) throws IOException {

		ExpandLexEvaluator consLex=new ExpandLexEvaluator("/Users/admin/workspace/IJCAI15/lexicons/STSLex.csv");
		consLex.processDict();
		
		HashMap<String,String> wordsCount=new HashMap<String,String>();

		
		for(String word:consLex.getDict().keySet()){
			String wordP=word.substring(2);
			String vals=consLex.getDict().get(word).get("label")+" "+consLex.getDict().get(word).get("POS");
			
			if(wordsCount.containsKey(wordP)){
				System.out.println(wordP+" "+wordsCount.get(wordP));		
				System.out.println(wordP+" "+vals);	
			}
			else{
				wordsCount.put(wordP,vals);
			}
			
			
		}


	}
}
