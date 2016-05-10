package lexexpand.core;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.HashMap;



public class EmotionEvaluator {
	protected String path;
	protected Map<String, Map<String, Integer>> dict; // each word is mapped to
														// different emotions
														// and their
														// corresponding values

	public EmotionEvaluator(String path) {
		this.path = path;
		this.dict = new HashMap<String, Map<String, Integer>>();

	}

	public Map<String, Map<String, Integer>> getDict() {
		return this.dict;
	}

	public Map<String, Integer> getWord(String word) {
		if (this.dict.containsKey(word))
			return dict.get(word);
		else
			return null;
	}

	public void processDict() throws IOException {

		BufferedReader bf = new BufferedReader(new FileReader(this.path));
		String line;
		while ((line = bf.readLine()) != null) {
			String content[] = line.split("\t");
			String word = content[0];
			String emotion = content[1];
			int value = Integer.parseInt(content[2]);

			// I check whether the word has been inserted into the dict
			if (this.dict.containsKey(word)) {
				Map<String, Integer> emotionMap = this.dict.get(content[0]);
				emotionMap.put(emotion, value);
			} else {
				Map<String, Integer> emotionMap = new HashMap<String, Integer>();
				emotionMap.put(emotion, value);
				this.dict.put(word, emotionMap);
			}

		}

		bf.close();

	}
	
	public void addNeutralDimension(){
		for(String word:this.dict.keySet()){
			int accVal=0;
			Map<String,Integer> emoValues=this.dict.get(word);
			for(String emo:emoValues.keySet()){
				accVal += emoValues.get(emo);
			}
			if(accVal==0)
				emoValues.put("neutral", 1);
			else
				emoValues.put("neutral", 0);
				
		}
	}

	// Calculate emotion-oriented features using NRC
	public Map<String, Integer> evaluateEmotion(List<String> words) {

		Map<String, Integer> emoCount = new HashMap<String, Integer>();

		int anger = 0;
		int anticipation = 0;
		int disgust = 0;
		int fear = 0;
		int joy = 0;
		int sadness = 0;
		int surprise = 0;
		int trust = 0;
		int negative = 0;
		int positive = 0;

		for (String word : words) {
			// I retrieve the EmotionMap if the word match the lexicon
			if (this.getDict().containsKey(word)) {
				Map<String, Integer> emotions = this.getDict().get(word);
				anger += emotions.get("anger");
				anticipation += emotions.get("anticipation");
				disgust += emotions.get("disgust");
				fear += emotions.get("fear");
				joy += emotions.get("joy");
				sadness += emotions.get("sadness");
				surprise += emotions.get("surprise");
				trust += emotions.get("trust");
				negative += emotions.get("negative");
				positive += emotions.get("positive");

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

	static public void main(String args[]) throws IOException {
		EmotionEvaluator eval = new EmotionEvaluator(
				"lexicons/NRC-emotion-lexicon-wordlevel-v0.92.txt");
		eval.processDict();
		eval.addNeutralDimension();
		Map<String,Integer> emoValue=eval.getDict().get("ham");
		for(String emo:emoValue.keySet()){
			System.out.println(emo+" "+emoValue.get(emo));
		}
		
//		List<String> words=MyUtils.cleanTokenize("mother");
//
//		Map<String, Integer> pal = eval.evaluateEmotion(words);
//
//		if (pal != null) {
//			for (String emo : pal.keySet()) {
//				System.out.println(emo + " " + pal.get(emo));
//			}
//		}

	}

}
