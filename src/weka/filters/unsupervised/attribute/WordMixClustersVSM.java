package weka.filters.unsupervised.attribute;


import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.List;
import java.util.ListIterator;
import java.util.Map;
import java.util.TreeMap;
import java.util.Vector;

import lexexpand.core.LexiconEvaluator;
import lexexpand.core.MyUtils;
import cmu.arktweetnlp.Tagger;
import cmu.arktweetnlp.Twokenize;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.SparseInstance;
import weka.core.Utils;
import weka.core.Capabilities.Capability;
import weka.filters.SimpleBatchFilter;

public class WordMixClustersVSM extends SimpleBatchFilter {

	/**
	 * 	 Given a corpus of documents creates Vector Space model for each word
	 *    using word clusters and the words
	 * 
	 */



	private static final long serialVersionUID = 7553647795494402690L;


	/** the vocabulary and the WordRep */
	protected Map<String, WordRep> wordInfo; 

	/** Contains a mapping of valid words to attribute indexes. */
	private TreeMap<String, Integer> m_Dictionary = new TreeMap<String, Integer>();

	/** Maps the wordClusters to the number of documents in which they occur */
	protected Map<String,Integer> wordClusters;
	
	/** Maps the POStags to the number of documents in which they occur */
	protected Map<String,Integer> posTags;
	
	


	/** the minimum number of documents for a word or wordcluster to be included. */
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
		List<Map<String,Double>> listDocs; // lists of documents where the words occurs
		List<Map<String,Double>> listWordClust; // lists of worclusters
		List<Map<String,Double>> listPOS; // List of POS VSM
		
		public WordRep(String word){
			this.word=word;
			this.numDoc=0; // number of documents
			this.listDocs=new ArrayList<Map<String,Double>>();
			this.listWordClust=new ArrayList<Map<String,Double>>();
			this.listPOS=new ArrayList<Map<String,Double>>();
		}

		public void addDoc(Map<String,Double> wordVector,Map<String,Double> clustVector,
				Map<String,Double> posVector){
			this.numDoc++;
			this.listDocs.add(wordVector);
			this.listWordClust.add(clustVector);
			this.listPOS.add(posVector);
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



	public Map<String, Double> calculateTermFreq(List<String> tokens) {
		Map<String, Double> termFreq = new HashMap<String, Double>();

		// Traverse the strings and increments the counter when the token was
		// already seen before
		for (String token : tokens) {
			if (termFreq.containsKey(token))
				termFreq.put(token, termFreq.get(token) + 1.0);
			else
				termFreq.put(token, 1.0);
		}

		return termFreq;
	}

	/* Converts a sequence of words into a sequence of word-clusters
	 * 
	 */	 	
	public List<String> clustList(List<String> tokens, Map<String,String> dict){
		List<String> clusters=new ArrayList<String>();
		for(String token:tokens){
			if(dict.containsKey(token)){
				clusters.add(dict.get(token));
			}

		}	
		return clusters;
	}


	/* Calculates the vocabulary and the word vectors from an Instances object
	 * The vocabulary is only extracted the first time the filter is run.
	 * 
	 */	 
	public void computeWordVecsAndVoc(Instances inputFormat) {


		if (!this.isFirstBatchDone()){


			// The POS tagger
			Tagger tagger = new Tagger();
			try {
				tagger.loadModel("models/model.20120919");
			} catch (IOException e1) {
				// TODO Auto-generated catch block
				e1.printStackTrace();
			}
			


			Map<String,String> dict=new HashMap<String,String>();
			try {
				BufferedReader bf = new BufferedReader(new FileReader(
						"resources/50mpaths2.txt"));
				String line;
				while ((line = bf.readLine()) != null) {
					String pair[] = line.split("\t");
					dict.put(pair[1], pair[0]);


				}
				bf.close();

			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}


			this.wordInfo = new HashMap<String, WordRep>();
			this.wordClusters=new HashMap<String,Integer>();
			this.posTags=new HashMap<String,Integer>();



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

				List<String> clusters=clustList(tokens,dict);

				
				// discard tweets with no words or no cluster word
				if(tokens.isEmpty() || clusters.isEmpty())
					continue;



				// POS tags
				List<String> posTags=MyUtils.getPOStags(tokens, tagger);


				// calculates the wordVector
				Map<String, Double> wordFreqs = this.calculateTermFreq(tokens);
				// calculates the cluster Vector
				Map<String, Double> clustFreqs = this.calculateTermFreq(clusters);
				// calculates the POS Vector
				Map<String, Double> posFreqs = this.calculateTermFreq(posTags); 			
				
				
				// adds the clusters to the List of clusters
				for(String clust:clustFreqs.keySet()){
					if(this.wordClusters.containsKey(clust)){
						int prevFreq=this.wordClusters.get(clust);
						this.wordClusters.put(clust,prevFreq+1);						
					}
					else{
						this.wordClusters.put(clust,1);
					}

				}
				
				
				// adds the POS to the List of POS tags
				for(String pos:posFreqs.keySet()){
					if(this.posTags.containsKey(pos)){
						int prevFreq=this.posTags.get(pos);
						this.posTags.put(pos,prevFreq+1);						
					}
					else{
						this.posTags.put(pos,1);
					}

				}
				
				
				
				
				
				// if the word is new we add it to the vocabulary, otherwise we
				// add the cluster to the vector
				for (String word : wordFreqs.keySet()) {

					if (this.wordInfo.containsKey(word)) {
						WordRep wordRep=this.wordInfo.get(word);
						wordRep.addDoc(wordFreqs, clustFreqs, posFreqs); // add the document

					} else{
						WordRep wordRep=new WordRep(word);
						wordRep.addDoc(wordFreqs, clustFreqs, posFreqs); // add the document
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
		this.m_Dictionary=new TreeMap<String,Integer>();


		ArrayList<Attribute> att = new ArrayList<Attribute>();

		int i=0;


		for(String word:this.wordInfo.keySet()){
			WordRep wordRep=this.wordInfo.get(word);

			// we only consider words occurring more times than the threshold


			if(wordRep.numDoc>=this.minNumDocs){
				// we normalise the vector			
				Attribute a = new Attribute("TOKEN-" + word);
				att.add(a);		
				this.m_Dictionary.put("TOKEN-" + word, i);
				i++;
			}



		}
		
		// the same with the word Clusters		

		for(String cluster:this.wordClusters.keySet()){

			if(this.wordClusters.get(cluster)>=this.minNumDocs){
				Attribute a = new Attribute("CLUSTER-" + cluster);
				att.add(a);		
				this.m_Dictionary.put("CLUSTER-" + cluster, i);
				i++;

			}
		}
		
		
		for(String pos:this.posTags.keySet()){

			if(this.posTags.get(pos)>=this.minNumDocs){
				Attribute a = new Attribute("POS-" + pos);
				att.add(a);		
				this.m_Dictionary.put("POS-" + pos, i);
				i++;

			}
		}


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



		for(String word:this.wordInfo.keySet()){

			// get the word vector
			WordRep wordRep=this.wordInfo.get(word);

			// if the word is not frequent enough is not considered
			if(wordRep.numDoc>=this.minNumDocs){

				double[] values = new double[result.numAttributes()];

				for(Map<String,Double> wordMap:wordRep.listDocs){
					for(String vecWord:wordMap.keySet()){
						// we first check that the word is valid
						if(this.m_Dictionary.containsKey("TOKEN-"+vecWord)){
							int attIndex=this.m_Dictionary.get("TOKEN-"+vecWord);
							values[attIndex] += wordMap.get(vecWord)/wordRep.numDoc;	

						}
					}


				}
				
				
				for(Map<String,Double> clustMap:wordRep.listWordClust){
					for(String vecClust:clustMap.keySet()){
						// we first check that the word is valid
						if(this.m_Dictionary.containsKey("CLUSTER-"+vecClust)){
							int attIndex=this.m_Dictionary.get("CLUSTER-"+vecClust);
							values[attIndex] += clustMap.get(vecClust)/wordRep.numDoc;	

						}
					}


				}
				
				
				for(Map<String,Double> posMap:wordRep.listPOS){
					for(String vecPos:posMap.keySet()){
						// we first check that the word is valid
						if(this.m_Dictionary.containsKey("POS-"+vecPos)){
							int attIndex=this.m_Dictionary.get("POS-"+vecPos);
							values[attIndex] += posMap.get(vecPos)/wordRep.numDoc;	

						}
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
		runFilter(new WordMixClustersVSM(), args);
	}

}
