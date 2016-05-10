package weka.filters.unsupervised.attribute;


import it.unimi.dsi.fastutil.objects.Object2IntMap;
import it.unimi.dsi.fastutil.objects.Object2IntOpenHashMap;
import it.unimi.dsi.fastutil.objects.Object2ObjectMap;
import it.unimi.dsi.fastutil.objects.Object2ObjectOpenHashMap;
import it.unimi.dsi.fastutil.objects.ObjectOpenHashSet;
import it.unimi.dsi.fastutil.objects.ObjectSet;

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


public class WordPMI extends TextToVSM {

	/**
	 * 	 Given a corpus of documents calculates the PMI of all words
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

	/** True if instances should be sparse */
	protected boolean sparseInstances=true;




	/** LexiconEvaluator for sentiment prefixes */
	protected LexiconEvaluator lex;


	// This class contains all the information of the word to compute the centroid
	class WordRep{
		String word; // the word
		int numDoc; // number of documents where the word occurs
		Object2IntMap<String> wordSpace; // the vector space model of the document

		//Object2DoubleOpenHashMap<String> z=new Object2DoubleOpenHashMap<String>()
		
		
		public WordRep(String word){
			this.word=word;
			this.numDoc=0;
			this.wordSpace=new Object2IntOpenHashMap<String>();
		}

		public void addDoc(ObjectSet<String> terms){
			this.numDoc++;			
			for(String term:terms){
				// if the word was seen before we add the current frequency
				this.wordSpace.put(term, this.wordSpace.getInt(term)+1);
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

		result.addElement(new Option("\t Sparse instances.\n"
				+ "\t(default: " + this.sparseInstances + ")", "S", 0, "-S"));


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

		if(this.sparseInstances)
			result.add("-S");


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

		this.sparseInstances=Utils.getFlag('S', options);

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



	public ObjectSet<String> calculateTermFreq(List<String> tokens) {
		ObjectSet<String> termFreq = new ObjectOpenHashSet<String>();
		termFreq.addAll(tokens);
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
				List<String> tokens=Twokenize.tokenizeRawTweetText(content);; 

				
				ObjectSet<String> terms= new ObjectOpenHashSet<String>();
				terms.addAll(tokens);

				// if the word is new we add it to the vocabulary, otherwise we
				// add the document to the vector
				for (String word : terms) {

					if (this.wordInfo.containsKey(word)) {
						WordRep wordRep=this.wordInfo.get(word);
						wordRep.addDoc(terms); // add the document

					} else{
						WordRep wordRep=new WordRep(word);
						wordRep.addDoc(terms); // add the document
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
			if(wordRep.numDoc>=this.minNumDocs){
				// we normalise the vector
				//wordRep.normaliseVector();				
				Attribute a = new Attribute(this.prefix + word);
				att.add(a);		
				this.m_Dictionary.put(word, i);
				i++;
			}


			
		}

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
					// we compute the PMI log2(c(w1,w2))+log2(N)-log2(c(w1))-log2((c(w2))
					// We add 1 to all the counts
					double pmi=	logOfBase(wordRep.wordSpace.getInt(innerWord)+1, 2)+logOfBase(instances.size()+1,2)
							-logOfBase(this.wordInfo.get(innerWord).numDoc+1,2)-logOfBase(wordRep.numDoc+1,2);
					// We consider the Positive PointWise Mutual Information
					values[attIndex]=pmi>0?pmi:0;
											
				}
			}	

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

	
	public double logOfBase(int base, int num) {
		return Math.log(num) / Math.log(base);
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



	public boolean isSparseInstances() {
		return sparseInstances;
	}


	public void setSparseInstances(boolean sparseInstances) {
		this.sparseInstances = sparseInstances;
	}


	public int getMinNumDocs() {
		return minNumDocs;
	}



	public void setMinNumDocs(int minNumDocs) {
		this.minNumDocs = minNumDocs;
	}


	public static void main(String[] args) {
		runFilter(new WordPMI(), args);
	}

}
