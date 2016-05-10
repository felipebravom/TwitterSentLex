package weka.filters.unsupervised.attribute;


import it.unimi.dsi.fastutil.objects.AbstractObjectSet;
import it.unimi.dsi.fastutil.objects.Object2DoubleMap;
import it.unimi.dsi.fastutil.objects.Object2DoubleOpenHashMap;
import it.unimi.dsi.fastutil.objects.Object2ObjectMap;
import it.unimi.dsi.fastutil.objects.Object2ObjectOpenHashMap;
import it.unimi.dsi.fastutil.objects.ObjectArrayList;
import it.unimi.dsi.fastutil.objects.ObjectOpenHashSet;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Enumeration;
import java.util.List;
import java.util.ListIterator;
import java.util.Vector;

import lexexpand.core.LexiconEvaluator;
import cmu.arktweetnlp.Twokenize;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.SparseInstance;
import weka.core.Utils;
import weka.core.Capabilities.Capability;
import weka.filters.SimpleBatchFilter;

public class WordDocRecCentRand extends SimpleBatchFilter {

	/**
	 * 	 Given a corpus of documents creates Vector Space model for each word using bag-of-words and cluster-based attributes
	 * 	 words are labelled to sentiment labels using a seed lexicon	
	 * 
	 */



	private static final long serialVersionUID = 7553647795494402690L;


	/** the vocabulary and the WordRep */
	protected Object2ObjectMap<String, WordRep> wordInfo; 


	/** Number of iterations */
	protected int iterations=100;

	/** Number of attribute dimensions*/
	protected int k=10;


	/** adapting parameter */
	protected double alpha=0.1;

	/** the minimum number of documents for a word to be included. */
	protected int minInstDocs=0; 

	/** Minimum number of distinct words to consider a tweet */
	protected int minNumWords=0;


	/** the index of the string attribute to be processed */
	protected int textIndex=1; 


	/** True if all tokens should be downcased. */
	protected boolean toLowerCase=true;



	/** True is the words will be labelled with the seed lexicon */
	protected boolean labelWords=true;


	/** True if the word name is included as an attribute */
	protected boolean reportWord=true;




	/** True is stopwords are discarded */
	protected boolean removeStopWords=false;

	/** The stopwords file */
	protected String stopWordsPath="resources/stopwords.txt";


	/** True if url, users, and repeated letters are cleaned */
	protected boolean cleanTokens=false;


	/** The path of the seed lexicon . */
	protected String lexPath="lexicons/seed.csv";


	/** LexiconEvaluator for sentiment prefixes */
	protected LexiconEvaluator lex;




	// This class contains all the information of the word to compute the centroid
	class WordRep{
		String word; // the word
		int numDoc; // the number of documents where the word appears
		double[] wordVector; // the vector space model of the word



		public WordRep(String word, int k){
			this.word=word;
			this.numDoc=0;
			this.wordVector=new double[k];


			for(int i=0;i<k;i++){
				this.wordVector[i]= Math.random();
			}

		}

		public void incNumDoc(){
			this.numDoc++;
		}




		// update the wordVector using a contextVector
		public void updateVector(double[] contVec,double alpha){
			for(int i=0;i<wordVector.length;i++){
				wordVector[i]=(1-alpha)*wordVector[i]+alpha*contVec[i];
			}

		}



	}






	@Override
	public String globalInfo() {
		return "A batch filter that creates a vector representation of words with polarity labels "
				+ "provided by a seed lexicon.  ";
	}



	@Override
	public Capabilities getCapabilities() {

		Capabilities result = new Capabilities(this);
		result.disableAll();



		// attributes
		result.enableAllAttributes();
		result.enable(Capability.MISSING_VALUES);

		// class
		result.enableAllClasses();
		result.enable(Capability.MISSING_CLASS_VALUES);
		result.enable(Capability.NO_CLASS);

		result.setMinimumNumberInstances(0);

		return result;
	}




	@Override
	public Enumeration<Option> listOptions() {
		Vector<Option> result = new Vector<Option>();



		result.addElement(new Option("\t Minimum count of an instance.\n"
				+ "\t(default: " + this.minInstDocs + ")", "N", 1, "-N"));



		result.addElement(new Option("\t Index of string attribute.\n"
				+ "\t(default: " + this.textIndex + ")", "I", 1, "-I"));		



		result.addElement(new Option("\t Lowercase content.\n"
				+ "\t(default: " + this.toLowerCase + ")", "L", 0, "-L"));

		result.addElement(new Option("\t Label words with the seed lexicon.\n"
				+ "\t(default: " + this.labelWords + ")", "K", 0, "-K"));

		result.addElement(new Option("\t The path of the seed lexicon.\n"
				+ "\t(default: " + this.lexPath + ")", "J", 1, "-J"));



		result.addElement(new Option("\t Include the word name as a string attribute.\n"
				+ "\t(default: " + this.reportWord + ")", "R", 0, "-R"));

		result.addElement(new Option("\t Discard stopwords.\n"
				+ "\t(default: " + this.removeStopWords + ")", "S", 0, "-S"));

		result.addElement(new Option("\t The path of the stopwords file.\n"
				+ "\t(default: " + this.stopWordsPath + ")", "T", 1, "-T"));

		result.addElement(new Option("\t  Clean tokens (replace 3 or more repetitions of a letter to 2 repetitions of it e.g, gooood to good, standarise URLs and @users).\n"
				+ "\t(default: " + this.cleanTokens + ")", "O", 0, "-O"));


		result.addAll(Collections.list(super.listOptions()));

		return result.elements();
	}


	/**
	 * returns the options of the current setup
	 * 
	 * @return the current options
	 */
	@Override
	public String[] getOptions() {

		Vector<String> result = new Vector<String>();



		result.add("-N");
		result.add("" + this.getMinInstDocs());



		result.add("-I");
		result.add("" + this.getTextIndex());


		if(this.toLowerCase)
			result.add("-L");


		if(this.labelWords)
			result.add("-K");


		if(this.isReportWord())
			result.add("-R");

		if(this.isRemoveStopWords())
			result.add("-S");

		result.add("-T");
		result.add("" + this.getStopWordsPath());

		if(this.isCleanTokens())
			result.add("-O");

		Collections.addAll(result, super.getOptions());

		return result.toArray(new String[result.size()]);
	}


	/**
	 * Parses the options for this object.
	 * <p/>
	 * 
	 * <!-- options-start --> <!-- options-end -->
	 * 
	 * @param options
	 *            the options to use
	 * @throws Exception
	 *             if setting of options fails
	 */
	@Override
	public void setOptions(String[] options) throws Exception {




		String textMinInstDocsOption = Utils.getOption('N', options);
		if (textMinInstDocsOption.length() > 0) {
			String[] textMinInstDocsOptionSpec = Utils.splitOptions(textMinInstDocsOption);
			if (textMinInstDocsOptionSpec.length == 0) {
				throw new IllegalArgumentException(
						"Invalid index");
			}
			int minDocIns = Integer.parseInt(textMinInstDocsOptionSpec[0]);
			this.setMinInstDocs(minDocIns);

		}






		String textIndexOption = Utils.getOption('I', options);
		if (textIndexOption.length() > 0) {
			String[] textIndexSpec = Utils.splitOptions(textIndexOption);
			if (textIndexSpec.length == 0) {
				throw new IllegalArgumentException(
						"Invalid index");
			}
			int index = Integer.parseInt(textIndexSpec[0]);
			this.setTextIndex(index);

		}




		this.toLowerCase=Utils.getFlag('L', options);



		this.labelWords=Utils.getFlag('K', options);


		String lexPathOption = Utils.getOption('J', options);
		if (lexPathOption.length() > 0) {
			String[] lexPathOptionSpec = Utils.splitOptions(lexPathOption);
			if (lexPathOptionSpec.length == 0) {
				throw new IllegalArgumentException(
						"Invalid prefix");
			}
			String lexPathVal = lexPathOptionSpec[0];
			this.setLexPath(lexPathVal);

		}



		this.reportWord=Utils.getFlag('R', options);

		this.removeStopWords=Utils.getFlag('S', options);


		String stopWordsPathOption = Utils.getOption('T', options);
		if (stopWordsPath.length() > 0) {
			String[] stopWordsPathSpec = Utils.splitOptions(stopWordsPathOption);
			if (stopWordsPathSpec.length == 0) {
				throw new IllegalArgumentException(
						"Invalid prefix");
			}
			String stopWordsPathVal = stopWordsPathSpec[0];
			this.setStopWordsPath(stopWordsPathVal);

		}

		this.cleanTokens=Utils.getFlag('O', options);



		super.setOptions(options);

		Utils.checkForRemainingOptions(options);


	}



	/* To allow determineOutputFormat to access to entire dataset
	 * (non-Javadoc)
	 * @see weka.filters.SimpleBatchFilter#allowAccessToFullInputFormat()
	 */
	public boolean allowAccessToFullInputFormat() {
		return true;
	}


	// tokenises and cleans the content 
	public List<String> tokenize(String content) {

		if(this.toLowerCase)
			content=content.toLowerCase();


		if(!this.cleanTokens&&!this.removeStopWords)
			return Twokenize.tokenizeRawTweetText(content);


		AbstractObjectSet<String> stopWords=null;

		if(this.removeStopWords){
			stopWords=new  ObjectOpenHashSet<String>(); 
			// we add the stop-words from the file
			try {
				BufferedReader bf=new BufferedReader(new FileReader(this.stopWordsPath));
				String line;
				while((line=bf.readLine())!=null){
					stopWords.add(line);
				}
				bf.close();				

			} catch (IOException e) {
				this.removeStopWords=false;

			}


		}


		// if a letters appears two or more times it is replaced by only two
		// occurrences of it
		if(this.cleanTokens)
			content = content.replaceAll("([a-z])\\1+", "$1$1");

		List<String> tokens = new ArrayList<String>();

		for (String word : Twokenize.tokenizeRawTweetText(content)) {
			String cleanWord = word;


			if(this.cleanTokens){
				// Replace URLs to a generic URL
				if (word.matches("http.*|ww\\..*")) {
					cleanWord = "http://www.url.com";
				}

				// Replaces user mentions to a generic user
				else if (word.matches("@.*")) {
					cleanWord = "@user";
				}

				// check stopWord
				if(this.removeStopWords){
					if(stopWords.contains(cleanWord))
						continue;

				}

			}

			tokens.add(cleanWord);
		}
		return tokens;
	}







	/* Calculates the vocabulary and the word vectors from an Instances object
	 * The vocabulary is only extracted the first time the filter is run.
	 * 
	 */	 
	public void computeWordVecsAndVoc(Instances inputFormat) {


		if (!this.isFirstBatchDone()){



			// the list of words using the string value as key
			this.wordInfo = new Object2ObjectOpenHashMap<String, WordRep>();



			// reference to the content of the message, users index start from zero
			Attribute attrCont = inputFormat.attribute(this.textIndex-1);

			for(int z=0;z<this.iterations;z++){
				for (ListIterator<Instance> it = inputFormat.listIterator(); it
						.hasNext();) {
					Instance inst = it.next();
					String content = inst.stringValue(attrCont);




					// tokenises the content 
					List<String> tokens=this.tokenize(content); 

					// Identifies the distinct terms
					AbstractObjectSet<String> terms=new  ObjectOpenHashSet<String>(); 
					terms.addAll(tokens);

					if(terms.size()>=this.minNumWords){

						// the context centroid vector
						double[] contCent=new double[this.k];
						ObjectArrayList<WordRep> tweetWords=new ObjectArrayList<WordRep>();

						// if the word is new we add it to the vocabulary
						for (String word : terms) {
							WordRep wordRep=null;

							if (this.wordInfo.containsKey(word)) {
								wordRep=this.wordInfo.get(word);


							} else{
								wordRep=new WordRep(word,this.k);
								this.wordInfo.put(word, wordRep);						
							}					

							wordRep.incNumDoc();

							for(int i=0;i<this.k;i++){
								contCent[i]+=wordRep.wordVector[i]/terms.size(); 
							}

							tweetWords.add(wordRep);				

						}



						for(WordRep wordRep:tweetWords){
							wordRep.updateVector(contCent, this.alpha);

						}

					}



				}
				
				
				
			}
			
		}

	}

	@Override
	protected Instances determineOutputFormat(Instances inputFormat) {

		// creates a lexiconEvaluator object only if this flag is on
		if(this.labelWords){
			this.lex = new LexiconEvaluator(
					this.lexPath);
			try {
				this.lex.processDict();
			} catch (IOException e) {
				this.labelWords=false;

			}			

		}



		// calculates the word frequency vectors and the vocabulary
		this.computeWordVecsAndVoc(inputFormat);




		ArrayList<Attribute> att = new ArrayList<Attribute>();




		for(int i=0;i<this.k;i++){
			att.add(new Attribute("att"+i));
		}

		// we add the word name as an attribute
		if(this.reportWord)
			att.add(new Attribute("WORD_NAME", (ArrayList<String>) null));



		// The target label
		if(this.labelWords){
			ArrayList<String> label = new ArrayList<String>();
			label.add("negative");
			label.add("neutral");
			label.add("positive");
			att.add(new Attribute("Class", label));

		}



		Instances result = new Instances(inputFormat.relationName(), att, 0);

		return result;
	}

	@Override
	protected Instances process(Instances instances) throws Exception {

		Instances result = getOutputFormat();



		for(String word:this.wordInfo.keySet()){
			// get the word vector
			WordRep wordRep=this.wordInfo.get(word);

			// We just consider valid words
			if(wordRep.numDoc>=this.minInstDocs){

				double[] values = new double[result.numAttributes()];
				
				for(int i=0;i<this.k;i++){
					values[i]=wordRep.wordVector[i];
					
				}


				if(this.reportWord){
					int wordNameIndex=result.attribute("WORD_NAME").index();
					values[wordNameIndex]=result.attribute(wordNameIndex).addStringValue(word);					
				}


				if(this.labelWords){
					String wordPol=this.lex.retrieveValue(word);
					if(wordPol.equals("negative"))
						values[result.numAttributes()-1]=0;
					else if(wordPol.equals("neutral"))
						values[result.numAttributes()-1]=1;
					else if(wordPol.equals("positive"))
						values[result.numAttributes()-1]=2;
					else
						values[result.numAttributes()-1]= Utils.missingValue();					
				}


				Instance inst=new DenseInstance(1, values);


				inst.setDataset(result);

				result.add(inst);

			}


		}



		return result;





	}







	/**
	 * Sets the index of the string attribute
	 * 
	 * @return the index of the documents.
	 */
	public int getTextIndex() {
		return textIndex;
	}

	/**
	 * Sets the index of the string attribute
	 * 
	 * @param textIndex the index of the string attribute
	 * 
	 */
	public void setTextIndex(int textIndex) {
		this.textIndex = textIndex;
	}







	public boolean isToLowerCase() {
		return toLowerCase;
	}

	public void setToLowerCase(boolean toLowerCase) {
		this.toLowerCase = toLowerCase;
	}



	public int getMinInstDocs() {
		return minInstDocs;
	}



	public void setMinInstDocs(int minInstDocs) {
		this.minInstDocs = minInstDocs;
	}



	public String getLexPath() {
		return lexPath;
	}



	public void setLexPath(String lexPath) {
		this.lexPath = lexPath;
	}


	public boolean isLabelWords() {
		return labelWords;
	}



	public void setLabelWords(boolean labelWords) {
		this.labelWords = labelWords;
	}


	public boolean isReportWord() {
		return reportWord;
	}



	public void setReportWord(boolean reportWord) {
		this.reportWord = reportWord;
	}



	public boolean isRemoveStopWords() {
		return removeStopWords;
	}



	public void setRemoveStopWords(boolean removeStopWords) {
		this.removeStopWords = removeStopWords;
	}



	public String getStopWordsPath() {
		return stopWordsPath;
	}



	public void setStopWordsPath(String stopWordsPath) {
		this.stopWordsPath = stopWordsPath;
	}



	public boolean isCleanTokens() {
		return cleanTokens;
	}



	public void setCleanTokens(boolean cleanTokens) {
		this.cleanTokens = cleanTokens;
	}

	public int getIterations() {
		return iterations;
	}

	public void setIterations(int iterations) {
		this.iterations = iterations;
	}


	public int getK() {
		return k;
	}



	public void setK(int k) {
		this.k = k;
	}



	public double getAlpha() {
		return alpha;
	}



	public void setAlpha(double alpha) {
		this.alpha = alpha;
	}

	
	public int getMinNumWords() {
		return minNumWords;
	}



	public void setMinNumWords(int minNumWords) {
		this.minNumWords = minNumWords;
	}


	public static void main(String[] args) {
		runFilter(new WordDocRecCentRand(), args);
	}

}
