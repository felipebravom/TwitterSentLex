package weka.core.tokenizers;


import java.util.Iterator;
import java.util.List;

import weka.core.RevisionUtils;
import cmu.arktweetnlp.Twokenize;

public class TwitterNLPTokenizer extends Tokenizer {

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

		List<String> words=Twokenize.tokenizeRawTweetText(s);
		this.tokenIterator=words.iterator();	


	}



	/**
	 * Runs the tokenizer with the given options and strings to tokenize. The
	 * tokens are printed to stdout.
	 * 
	 * @param args the commandline options and strings to tokenize
	 */
	public static void main(String[] args) {
		runTokenizer(new TwitterNLPTokenizer(), args);
	}

}
