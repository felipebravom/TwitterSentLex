package weka.filters.unsupervised.attribute;


import it.unimi.dsi.fastutil.objects.Object2IntMap;
import it.unimi.dsi.fastutil.objects.Object2IntOpenHashMap;
import it.unimi.dsi.fastutil.objects.Object2ObjectMap;
import it.unimi.dsi.fastutil.objects.Object2ObjectOpenHashMap;

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
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.SparseInstance;
import weka.core.Utils;
import weka.core.Capabilities.Capability;

public class WordVSMCompactFast extends TextToVSM {

	/**
	 * 	 Given a corpus of documents creates Vector Space model for each word
	 * 
	 */



	private static final long serialVersionUID = 7553647795494402690L;


	/** the vocabulary and the WordRep */
	protected Object2ObjectMap<String, WordRep> wordInfo; 

	/** Contains a mapping of valid words to attribute indexes. */
	private Object2IntMap<String> m_Dictionary;

	/** the minimum number of documents for a word to be included. */
	protected int minNumDocs=0; 


	/** the index of the string attribute to be processed */
	protected int textIndex=1; 

	/** the index of the string attribute to be processed */
	protected String prefix="";

	/** True if all tokens should be downcased. */
	protected boolean toLowerCase=true;


	/** True if the number of documents where the word occurs is reported. */
	protected boolean numDocs=false;
	
	
	


	/** LexiconEvaluator for sentiment prefixes */
	protected LexiconEvaluator lex;


	// This class contains all the information of the word to compute the centroid
	class WordRep{
		String word; // the word
		int numDoc; // number of documents where the word occurs
		Object2IntMap<String> wordSpace; // the vector space model of the document

		
		
		public WordRep(String word){
			this.word=word;
			this.numDoc=0;
			this.wordSpace=new Object2IntOpenHashMap<String>();
		}

		public void addDoc(Object2IntMap<String> docVector){
			this.numDoc++;
			for(String vecWord:docVector.keySet()){
				int vecWordFreq=docVector.getInt(vecWord);
				// if the word was seen before we add the current frequency
				this.wordSpace.put(vecWord,vecWordFreq+this.wordSpace.getInt(vecWord));
			}	

		}

	}

	@Override
	public String globalInfo() {
		return "A simple batch filter that adds attributes for all the "
				+ "Twitter-oriented POS tags of the TwitterNLP library.  ";
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
		
		result.addElement(new Option("\t Minumum number of documents.\n"
				+ "\t(default: " + this.minNumDocs + ")", "M", 1, "-M"));

		result.addElement(new Option("\t Index of string attribute.\n"
				+ "\t(default: " + this.textIndex + ")", "I", 1, "-I"));		

		result.addElement(new Option("\t Prefix of attributes.\n"
				+ "\t(default: " + this.prefix + ")", "P", 1, "-P"));

		result.addElement(new Option("\t Lowercase content.\n"
				+ "\t(default: " + this.toLowerCase + ")", "L", 0, "-L"));
		
		result.addElement(new Option("\t Report number of documents.\n"
				+ "\t(default: " + this.numDocs + ")", "D", 0, "-D"));
		

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

		result.add("-M");
		result.add("" + this.getMinNumDocs());
		
		result.add("-I");
		result.add("" + this.getTextIndex());

		result.add("-P");
		result.add("" + this.getPrefix());

		if(this.toLowerCase)
			result.add("-L");
		
		if(this.numDocs)
			result.add("-D");

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
		
		
		String textMinNumDocsOption = Utils.getOption('M', options);
		if (textMinNumDocsOption.length() > 0) {
			String[] textMinNumDocsSpec = Utils.splitOptions(textMinNumDocsOption);
			if (textMinNumDocsSpec.length == 0) {
				throw new IllegalArgumentException(
						"Invalid index");
			}
			int minDoc = Integer.parseInt(textMinNumDocsSpec[0]);
			this.setMinNumDocs(minDoc);

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

		String prefixOption = Utils.getOption('P', options);
		if (prefixOption.length() > 0) {
			String[] prefixSpec = Utils.splitOptions(prefixOption);
			if (prefixSpec.length == 0) {
				throw new IllegalArgumentException(
						"Invalid prefix");
			}
			String pref = prefixSpec[0];
			this.setPrefix(pref);

		}

		this.toLowerCase=Utils.getFlag('L', options);
		
		this.numDocs=Utils.getFlag('D', options);

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



	public Object2IntMap<String> calculateTermFreq(List<String> tokens) {
		Object2IntMap<String> termFreq = new Object2IntOpenHashMap<String>();

		// Traverse the strings and increments the counter when the token was
		// already seen before
		for (String token : tokens) {
			termFreq.put(token, termFreq.getInt(token) + 1);			
		}

		return termFreq;
	}


	/* Calculates the vocabulary and the word vectors from an Instances object
	 * The vocabulary is only extracted the first time the filter is run.
	 * 
	 */	 
	public void computeWordVecsAndVoc(Instances inputFormat) {

		
		if (!this.isFirstBatchDone()){
			
			
			this.wordInfo = new Object2ObjectOpenHashMap<String, WordRep>();

			

			// reference to the content of the message, users index start from zero
			Attribute attrCont = inputFormat.attribute(this.textIndex-1);

			for (ListIterator<Instance> it = inputFormat.listIterator(); it
					.hasNext();) {
				Instance inst = it.next();
				String content = inst.stringValue(attrCont);
				if(this.toLowerCase)
					content=content.toLowerCase();

				// tokenises the content 
				List<String> tokens=Twokenize.tokenizeRawTweetText(content); 

				// calculates the wordVector
				Object2IntMap<String> wordFreqs = this.calculateTermFreq(tokens);



				// if the word is new we add it to the vocabulary, otherwise we
				// add the document to the vector
				for (String word : wordFreqs.keySet()) {

					if (this.wordInfo.containsKey(word)) {
						WordRep wordRep=this.wordInfo.get(word);
						wordRep.addDoc(wordFreqs); // add the document

					} else{
						WordRep wordRep=new WordRep(word);
						wordRep.addDoc(wordFreqs); // add the document
						this.wordInfo.put(word, wordRep);						
					}


				}



			}
		}

	}

	@Override
	protected Instances determineOutputFormat(Instances inputFormat) {


		// calculates the word frequency vectors and the vocabulary
		this.computeWordVecsAndVoc(inputFormat);

		
		// the dictionary of words and attribute indexes
		this.m_Dictionary=new Object2IntOpenHashMap<String>();


		ArrayList<Attribute> att = new ArrayList<Attribute>();

		int i=0;
		for(String word:this.wordInfo.keySet()){
			WordRep wordRep=this.wordInfo.get(word);

			// we only consider words occurring more times than the threshold
		//	double newThres=Math.pow(inputFormat.numInstances(),0.3);

			if(wordRep.numDoc>=this.minNumDocs){			
				Attribute a = new Attribute(this.prefix + word);
				att.add(a);		
				this.m_Dictionary.put(word, i);
				i++;
			}


			
		}
		
		// add the number of documents as an attribute
		if(this.numDocs)
			att.add(new Attribute("NUM_DOCS"));

		// we add the word name as an attribute
		att.add(new Attribute("WORD_NAME", (ArrayList<String>) null));

		
		// The target label
		ArrayList<String> label = new ArrayList<String>();
	

		label.add("negative");
		label.add("neutral");
		label.add("positive");
		
		att.add(new Attribute("Class", label));


		Instances result = new Instances(inputFormat.relationName(), att, 0);

		return result;
	}

	@Override
	protected Instances process(Instances instances) throws Exception {

		LexiconEvaluator metaLex = new LexiconEvaluator(
				"lexicons/metaLexEmo.csv");
		metaLex.processDict();
		
		Instances result = getOutputFormat();
		
		for(String word:this.m_Dictionary.keySet()){
			double[] values = new double[result.numAttributes()];
			// get the word vector
			WordRep wordRep=this.wordInfo.get(word);			
			
			for(String innerWord:wordRep.wordSpace.keySet()){
				// only include valid words
				if(this.m_Dictionary.containsKey(innerWord)){
					int attIndex=this.m_Dictionary.getInt(innerWord);
					// we normalise the value by the number of documents
					values[attIndex]=((double)wordRep.wordSpace.getInt(innerWord))/wordRep.numDoc;					
				}
			}
			
			if(this.numDocs)
				values[result.numAttributes()-3]=wordRep.numDoc;
				

			values[result.numAttributes()-2]=result.attribute(result.numAttributes()-2).addStringValue(word);

			String wordPol=metaLex.retrieveValue(word);
			if(wordPol.equals("negative"))
				values[result.numAttributes()-1]=0;
			else if(wordPol.equals("neutral"))
				values[result.numAttributes()-1]=1;
			else if(wordPol.equals("positive"))
				values[result.numAttributes()-1]=2;
			else
				values[result.numAttributes()-1]= Utils.missingValue();
			
			Instance inst=new SparseInstance(1, values);


			inst.setDataset(result);

			result.add(inst);

		}


		return result;
	}



	public int getTextIndex() {
		return textIndex;
	}


	public void setTextIndex(int textIndex) {
		this.textIndex = textIndex;
	}


	public String getPrefix() {
		return prefix;
	}


	public void setPrefix(String prefix) {
		this.prefix = prefix;
	}


	public boolean isToLowerCase() {
		return toLowerCase;
	}

	public void setToLowerCase(boolean toLowerCase) {
		this.toLowerCase = toLowerCase;
	}


	public int getMinNumDocs() {
		return minNumDocs;
	}



	public void setMinNumDocs(int minNumDocs) {
		this.minNumDocs = minNumDocs;
	}


	public static void main(String[] args) {
		runFilter(new WordVSMCompactFast(), args);
	}

}
