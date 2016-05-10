package weka.core.tokenizers;


import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import weka.core.RevisionUtils;
import cmu.arktweetnlp.Twokenize;

public class TweetCleanTokenizer extends Tokenizer {

	/**
	 * 
	 */
	private static final long serialVersionUID = 4352757127093531518L;


	protected transient Iterator<String> tokenIterator;


	@Override
	public String getRevision() {
		return RevisionUtils.extract("$Revision: 10203 $");
	}

	@Override
	public String globalInfo() {
		return "A tokenizer based on the TwitterNLP library.";				
	}





	@Override
	public boolean hasMoreElements() {
		return this.tokenIterator.hasNext();	
	}

	@Override
	public String nextElement() {
		return this.tokenIterator.next();	
	}

	@Override
	public void tokenize(String s) {

		String content = s.replaceAll("([a-z])\\1+", "$1$1");


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




		this.tokenIterator=tokens.iterator();	


	}



	/**
	 * Runs the tokenizer with the given options and strings to tokenize. The
	 * tokens are printed to stdout.
	 * 
	 * @param args the commandline options and strings to tokenize
	 */
	public static void main(String[] args) {
		runTokenizer(new TweetCleanTokenizer(), args);
	}

}
