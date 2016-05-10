package weka.filters.unsupervised.attribute;


import it.unimi.dsi.fastutil.doubles.AbstractDoubleList;
import it.unimi.dsi.fastutil.objects.AbstractObjectSet;
import it.unimi.dsi.fastutil.objects.Object2DoubleMap;
import it.unimi.dsi.fastutil.objects.Object2DoubleOpenHashMap;
import it.unimi.dsi.fastutil.objects.Object2IntMap;
import it.unimi.dsi.fastutil.objects.Object2IntOpenHashMap;
import it.unimi.dsi.fastutil.objects.Object2ObjectMap;
import it.unimi.dsi.fastutil.objects.Object2ObjectOpenHashMap;
import it.unimi.dsi.fastutil.objects.ObjectArrayList;
import it.unimi.dsi.fastutil.objects.ObjectList;
import it.unimi.dsi.fastutil.objects.ObjectOpenHashSet;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Enumeration;
import java.util.List;
import java.util.ListIterator;
import java.util.Map;
import java.util.Vector;

import lexexpand.core.EmotionEvaluator;
import lexexpand.core.LexiconEvaluator;
import lexexpand.core.MyUtils;
import lexexpand.core.Word2VecDict;
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

public class WordCentroidMultiLabel extends SimpleBatchFilter {

	/**
	 * 	 Given a corpus of documents creates Vector Space model for each word using multiple attributes
	 * 
	 */



	private static final long serialVersionUID = 7553647795494402690L;


	/** the vocabulary and the WordRep */
	protected Object2ObjectMap<String, WordRep> wordInfo; 

	/** Counts the number of documents in which candidate attributes appears */
	protected Object2IntMap<String> attributeCount;


	/** Contains a mapping of valid attribute with their indexes. */
	protected Object2IntMap<String> m_Dictionary;

	/** Brown Clusters Dictionary */
	protected Object2ObjectMap<String,String> brownDict;

	/** MetaData of Numerical attributes to be transferred to the word-level */
	protected ObjectList<Attribute> metaData;

	/** the minimum number of documents for an attribute to be considered. */
	protected int minAttDocs=0; 


	/** the minimum number of documents for a word to be included. */
	protected int minInstDocs=0; 


	/** the index of the string attribute to be processed */
	protected int textIndex=1; 


	/** the prefix of the word attributes */
	protected String wordPrefix="WORD-";

	/** the prefix of the cluster-based attributes */
	protected String clustPrefix="CLUST-";


	/** the prefix of the POS-bases attributes */
	protected String posPrefix="POS-";

	/** True if all tokens should be downcased. */
	protected boolean toLowerCase=true;


	/** True for calculating word-based attributes . */
	protected boolean createWordAtts=true;


	/** True for calculating cluster-based attributes . */
	protected boolean createClustAtts=true;

	/** True for calculating POS attributes. */
	protected boolean createPosAtts=true;


	/** True if the number of documents where the word occurs is reported. */
	protected boolean reportNumDocs=false;


	/** True is the words will be labelled with the seed lexicon */
	protected boolean labelWords=true;


	/** True if the word name is included as an attribute */
	protected boolean reportWord=true;

	/** True if additional numerical attributes should be included to the centroid */
	protected boolean includeMetaData=true;

	/** True if additional word-level features provided from Word2Vector should be included */
	protected boolean includeWord2Vec=true;


	/** True is stopwords are discarded */
	protected boolean removeStopWords=false;

	/** The stopwords file */
	protected String stopWordsPath="resources/stopwords.txt";


	/** True if url, users, and repeated letters are cleaned */
	protected boolean cleanTokens=false;


	/** The path of the seed lexicon . */
	protected String lexPath="lexicons/NRC-emotion-lexicon-wordlevel-v0.92.txt";

	/** The path of the word clusters. */
	protected String clustPath="resources/50mpaths2.txt";

	/** The POS tagger. */
	protected Tagger tagger; 

	/** LexiconEvaluator for sentiment prefixes */
	protected EmotionEvaluator lex;

	/** A word2Vector dictionary */
	protected Word2VecDict word2VecDict;

	/** the word2Vector Path*/
	protected String word2VecPath="resources/edim_lab_word2Vec.csv";
	
	/** True to add a neutral dummy label */
	protected boolean addNeutralLabel=true;


	// This class contains all the information of the word to compute the centroid
	class WordRep{
		String word; // the word
		int numDoc; // number of documents where the word occurs
		Object2IntMap<String> wordSpace; // the vector space model of the word

		Object2DoubleMap<String> metaData; //



		public WordRep(String word){
			this.word=word;
			this.numDoc=0;
			this.wordSpace=new Object2IntOpenHashMap<String>();
			this.metaData=new Object2DoubleOpenHashMap<String>();
		}

		public void addDoc(Object2IntMap<String> docVector){
			this.numDoc++;
			for(String vecWord:docVector.keySet()){
				int vecWordFreq=docVector.getInt(vecWord);
				// if the word was seen before we add the current frequency
				this.wordSpace.put(vecWord,vecWordFreq+this.wordSpace.getInt(vecWord));
			}				

		}

		public void addMetaData(Object2DoubleMap<String> metaVector){
			for(String metaName:metaVector.keySet()){
				double metaVal=metaVector.getDouble(metaName);
				this.metaData.put(metaName, metaVal+this.metaData.getDouble(metaName));
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

		result.addElement(new Option("\t Minimum count of an attribute.\n"
				+ "\t(default: " + this.minAttDocs + ")", "M", 1, "-M"));

		result.addElement(new Option("\t Minimum count of an instance.\n"
				+ "\t(default: " + this.minInstDocs + ")", "N", 1, "-N"));

		result.addElement(new Option("\t Create word-based attributes.\n"
				+ "\t(default: " + this.createWordAtts + ")", "W", 0, "-W"));

		result.addElement(new Option("\t Create cluster-based attributes.\n"
				+ "\t(default: " + this.createClustAtts + ")", "C", 0, "-C"));


		result.addElement(new Option("\t Index of string attribute.\n"
				+ "\t(default: " + this.textIndex + ")", "I", 1, "-I"));		

		result.addElement(new Option("\t Prefix of word attributes.\n"
				+ "\t(default: " + this.wordPrefix + ")", "P", 1, "-P"));


		result.addElement(new Option("\t Prefix of cluster attributes.\n"
				+ "\t(default: " + this.clustPrefix + ")", "Q", 1, "-Q"));


		result.addElement(new Option("\t Lowercase content.\n"
				+ "\t(default: " + this.toLowerCase + ")", "L", 0, "-L"));

		result.addElement(new Option("\t Report number of documents.\n"
				+ "\t(default: " + this.reportNumDocs + ")", "D", 0, "-D"));


		result.addElement(new Option("\t Label words with the seed lexicon.\n"
				+ "\t(default: " + this.labelWords + ")", "K", 0, "-K"));

		result.addElement(new Option("\t The path of the seed lexicon.\n"
				+ "\t(default: " + this.lexPath + ")", "J", 1, "-J"));

		result.addElement(new Option("\t The path of the file with the word clusters.\n"
				+ "\t(default: " + this.clustPath + ")", "H", 1, "-H"));


		result.addElement(new Option("\t Include the word name as a string attribute.\n"
				+ "\t(default: " + this.reportWord + ")", "R", 0, "-R"));

		result.addElement(new Option("\t Discard stopwords.\n"
				+ "\t(default: " + this.removeStopWords + ")", "S", 0, "-S"));

		result.addElement(new Option("\t The path of the stopwords file.\n"
				+ "\t(default: " + this.stopWordsPath + ")", "T", 1, "-T"));

		result.addElement(new Option("\t Clean tokens (replace goood by good, standarise URLs and @users).\n"
				+ "\t(default: " + this.cleanTokens + ")", "O", 0, "-O"));

		result.addElement(new Option("\t Add POS unigrams and bigrams.\n"
				+ "\t(default: " + this.createPosAtts + ")", "A", 0, "-A"));

		result.addElement(new Option("\t Include metadata. All numerical attributes will be averaged at the word level. \n"
				+ "\t(default: " + this.includeMetaData + ")", "B", 0, "-B"));

		result.addElement(new Option("\t Include Word2Vecfeatures. Include word-level features trained with word2vec from a csv file.  \n"
				+ "\t(default: " + this.includeWord2Vec + ")", "E", 0, "-E"));		

		result.addElement(new Option("\t Prefix of POS attributes  \n"
				+ "\t(default: " + this.posPrefix + ")", "F", 1, "-F"));

		result.addElement(new Option("\t The path of the word2vec embedding (it must be a csv file).  \n"
				+ "\t(default: " + this.word2VecPath + ")", "G", 1, "-G"));
		
		result.addElement(new Option("\t Add a neutral category.  \n"
				+ "\t(default: " + this.addNeutralLabel + ")", "U", 0, "-U"));


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
		result.add("" + this.getMinAttDocs());

		result.add("-N");
		result.add("" + this.getMinInstDocs());

		if(this.createWordAtts)
			result.add("-W");

		if(this.createClustAtts)
			result.add("-C");

		result.add("-I");
		result.add("" + this.getTextIndex());

		result.add("-P");
		result.add("" + this.getWordPrefix());

		result.add("-Q");
		result.add("" + this.getClustPrefix());


		if(this.toLowerCase)
			result.add("-L");

		if(this.reportNumDocs)
			result.add("-D");

		if(this.labelWords)
			result.add("-K");

		result.add("-J");
		result.add("" + this.getLexPath());

		result.add("-H");
		result.add("" + this.getClustPath());

		if(this.isReportWord())
			result.add("-R");

		if(this.isRemoveStopWords())
			result.add("-S");

		result.add("-T");
		result.add("" + this.getStopWordsPath());

		if(this.isCleanTokens())
			result.add("-O");

		if(this.isCreatePosAtts())
			result.add("-A");

		if(this.isIncludeMetaData())
			result.add("-B");

		if(this.isIncludeWord2Vec())
			result.add("-E");

		result.add("-F");
		result.add("" + this.getPosPrefix());

		result.add("-G");
		result.add("" + this.getWord2VecPath());
		
		if(this.isAddNeutralLabel())
			result.add("-U");
			
			

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


		String textMinAttDocOption = Utils.getOption('M', options);
		if (textMinAttDocOption.length() > 0) {
			String[] textMinAttDocOptionSpec = Utils.splitOptions(textMinAttDocOption);
			if (textMinAttDocOptionSpec.length == 0) {
				throw new IllegalArgumentException(
						"Invalid index");
			}
			int minDocAtt = Integer.parseInt(textMinAttDocOptionSpec[0]);
			this.setMinAttDocs(minDocAtt);

		}

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

		this.createWordAtts=Utils.getFlag('W', options);

		this.createClustAtts=Utils.getFlag('C', options);




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

		String wordPrefixOption = Utils.getOption('P', options);
		if (wordPrefixOption.length() > 0) {
			String[] wordPrefixSpec = Utils.splitOptions(wordPrefixOption);
			if (wordPrefixSpec.length == 0) {
				throw new IllegalArgumentException(
						"Invalid prefix");
			}
			String wordPref = wordPrefixSpec[0];
			this.setWordPrefix(wordPref);

		}

		String clustPrefixOption = Utils.getOption('Q', options);
		if (clustPrefixOption.length() > 0) {
			String[] clustPrefixOptionSpec = Utils.splitOptions(clustPrefixOption);
			if (clustPrefixOptionSpec.length == 0) {
				throw new IllegalArgumentException(
						"Invalid prefix");
			}
			String clustPref = clustPrefixOptionSpec[0];
			this.setClustPrefix(clustPref);

		}


		this.toLowerCase=Utils.getFlag('L', options);

		this.reportNumDocs=Utils.getFlag('D', options);


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

		String clustPathOption = Utils.getOption('H', options);
		if (clustPathOption.length() > 0) {
			String[] clustPathOptionSpec = Utils.splitOptions(clustPathOption);
			if (clustPathOptionSpec.length == 0) {
				throw new IllegalArgumentException(
						"Invalid prefix");
			}
			String clustPathVal = clustPathOptionSpec[0];
			this.setClustPath(clustPathVal);

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

		this.createPosAtts=Utils.getFlag('A', options);

		this.includeMetaData=Utils.getFlag('B', options);

		this.includeWord2Vec=Utils.getFlag('E', options);


		String posPrefixOption = Utils.getOption('F', options);
		if (posPrefix.length() > 0) {
			String[] posPrefixSpec = Utils.splitOptions(posPrefixOption);
			if (posPrefixSpec.length == 0) {
				throw new IllegalArgumentException(
						"Invalid prefix");
			}
			String posPrefixVal = posPrefixSpec[0];
			this.setPosPrefix(posPrefixVal);

		}

		String word2VecPathOption = Utils.getOption('G', options);
		if (word2VecPath.length() > 0) {
			String[] word2VecPathSpec = Utils.splitOptions(word2VecPathOption);
			if (word2VecPathSpec.length == 0) {
				throw new IllegalArgumentException(
						"Invalid prefix");
			}
			String word2VecPathVal = word2VecPathSpec[0];
			this.setWord2VecPath(word2VecPathVal);

		}
		
		this.addNeutralLabel=Utils.getFlag('U', options);


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


	public Object2IntMap<String> calculateTermFreq(List<String> tokens, String prefix) {
		Object2IntMap<String> termFreq = new Object2IntOpenHashMap<String>();

		// Traverse the strings and increments the counter when the token was
		// already seen before
		for (String token : tokens) {
			termFreq.put(prefix+token, termFreq.getInt(prefix+token) + 1);			
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

	public List<String> calculateBigrams(List<String> tokens){
		List<String> bigrams=new ArrayList<String>();
		if(tokens.size()>=2){			
			for(int i=0;i<tokens.size()-1;i++){
				String bigram="BIG-"+tokens.get(i)+"-"+tokens.get(i+1);
				bigrams.add(bigram);
			}
		}
		return bigrams;		

	}






	public Object2IntMap<String> calculateDocVec(List<String> tokens) {

		Object2IntMap<String> docVec = new Object2IntOpenHashMap<String>();
		// add the word-based vector
		if(this.createWordAtts){
			docVec.putAll(calculateTermFreq(tokens,this.wordPrefix));			
		}

		if(this.createClustAtts){
			// calcultates the vector of clusters
			List<String> brownClust=clustList(tokens,brownDict);
			docVec.putAll(calculateTermFreq(brownClust,this.clustPrefix));	
		}

		if(this.createPosAtts){
			List<String> posTags=MyUtils.getPOStags(tokens, this.tagger);
			docVec.putAll(calculateTermFreq(posTags,this.posPrefix));
			// add POS bigrams as well
			docVec.putAll(calculateTermFreq(calculateBigrams(posTags),this.posPrefix));

		}

		return docVec;

	}






	/* Calculates the vocabulary and the word vectors from an Instances object
	 * The vocabulary is only extracted the first time the filter is run.
	 * 
	 */	 
	public void computeWordVecsAndVoc(Instances inputFormat) {


		if (!this.isFirstBatchDone()){


			this.wordInfo = new Object2ObjectOpenHashMap<String, WordRep>();

			this.attributeCount= new Object2IntOpenHashMap<String>(); 



			// the Dictionary of the brown Clusters
			if(this.createClustAtts){
				this.brownDict=new Object2ObjectOpenHashMap<String,String>();
				try {
					BufferedReader bf = new BufferedReader(new FileReader(
							this.clustPath));
					String line;
					while ((line = bf.readLine()) != null) {
						String pair[] = line.split("\t");
						brownDict.put(pair[1], pair[0]);


					}
					bf.close();

				} catch (IOException e) {
					// do not create clusters attributes
					this.setCreateClustAtts(false);
				}

			}

			// Loads the POS tagger model
			if(this.createPosAtts){				
				try {
					this.tagger= new Tagger();
					this.tagger.loadModel("models/model.20120919");
				} catch (IOException e) {
					this.createPosAtts=false;
				}

			}


			if(this.includeWord2Vec){
				this.word2VecDict=new Word2VecDict(this.word2VecPath);
				try {
					this.word2VecDict.createDict();
				} catch (Exception e) {
					this.includeWord2Vec=false;
				}
			}




			// reference to the content of the message, users index start from zero
			Attribute attrCont = inputFormat.attribute(this.textIndex-1);


			// if MetaData is set to true we will include additional features in the centroids
			if(this.includeMetaData){
				this.metaData=new  ObjectArrayList<Attribute>();
				for(int i=0;i<inputFormat.numAttributes();i++){
					if(i!=this.textIndex && inputFormat.attribute(i).type()==Attribute.NUMERIC){
						this.metaData.add(inputFormat.attribute(i));						
					}

				}
			}



			for (ListIterator<Instance> it = inputFormat.listIterator(); it
					.hasNext();) {
				Instance inst = it.next();
				String content = inst.stringValue(attrCont);


				// tokenises the content 
				List<String> tokens=this.tokenize(content); 

				// Identifies the distinct terms
				AbstractObjectSet<String> terms=new  ObjectOpenHashSet<String>(); 
				terms.addAll(tokens);


				Object2IntMap<String> docVec=this.calculateDocVec(tokens);			




				// adds the attributes to the List of attributes
				for(String docAtt:docVec.keySet()){
					if(this.attributeCount.containsKey(docAtt)){
						int prevFreq=this.attributeCount.getInt(docAtt);
						this.attributeCount.put(docAtt,prevFreq+1);						
					}
					else{
						this.attributeCount.put(docAtt,1);
					}

				}



				// if the word is new we add it to the vocabulary, otherwise we
				// add the document to the vector
				for (String word : terms) {

					// if the word has no vector is discarded
					if(this.includeWord2Vec)
						if(!this.word2VecDict.getWordMap().containsKey(word))
							continue;

					WordRep wordRep;

					if (this.wordInfo.containsKey(word)) {
						wordRep=this.wordInfo.get(word);
						wordRep.addDoc(docVec); // add the document

					} else{
						wordRep=new WordRep(word);
						wordRep.addDoc(docVec); // add the document
						this.wordInfo.put(word, wordRep);						
					}

					if(this.includeMetaData){
						Object2DoubleMap<String> metaValues=new Object2DoubleOpenHashMap<String>();
						for(Attribute att:this.metaData){
							metaValues.put(att.name(),inst.value(att));
						}
						wordRep.addMetaData(metaValues);						
					}					


				}


			}





		}
	}



	@Override
	protected Instances determineOutputFormat(Instances inputFormat) {

		// creates a lexiconEvaluator object only if this flag is on
		if(this.labelWords){
			this.lex = new EmotionEvaluator(
					this.lexPath);
			try {
				this.lex.processDict();
				if(this.addNeutralLabel)
					this.lex.addNeutralDimension();
			} catch (IOException e) {
				this.labelWords=false;

			}			

		}



		// calculates the word frequency vectors and the vocabulary
		this.computeWordVecsAndVoc(inputFormat);


		// the dictionary of words and attribute indexes
		this.m_Dictionary=new Object2IntOpenHashMap<String>();


		ArrayList<Attribute> att = new ArrayList<Attribute>();

		int i=0;

		// The target label
		if(this.labelWords){

			ArrayList<String> values = new ArrayList<String>();
			values.add("0");
			values.add("1");

			att.add(new Attribute("anger",values));
			att.add(new Attribute("anticipation",values));
			att.add(new Attribute("disgust",values));
			att.add(new Attribute("fear",values));
			att.add(new Attribute("joy",values));
			att.add(new Attribute("negative",values));
			att.add(new Attribute("positive",values));
			att.add(new Attribute("sadness",values));
			att.add(new Attribute("surprise",values));
			att.add(new Attribute("trust",values));	
			i+=10;	
			
			if(this.addNeutralLabel){
				att.add(new Attribute("neutral",values));
				i++;
			}
			
					
		}

		if(this.includeMetaData){
			for(Attribute metaAtt:this.metaData){
				att.add(metaAtt);
				i++;
			}
		}

		if(this.includeWord2Vec){
			for(int j=0;j<this.word2VecDict.getDimensions();j++){
				att.add(new Attribute("Word2Vec-"+j));
				i++;
			}


		}

		for(String attribute:this.attributeCount.keySet()){
			if(this.attributeCount.get(attribute)>=this.minAttDocs){
				Attribute a = new Attribute(attribute);
				att.add(a);		
				this.m_Dictionary.put(attribute, i);
				i++;

			}
		}


		// add the number of documents as an attribute
		if(this.reportNumDocs)
			att.add(new Attribute("NUM_DOCS"));

		// we add the word name as an attribute
		if(this.reportWord)
			att.add(new Attribute("WORD_NAME", (ArrayList<String>) null));






		Instances result = new Instances(inputFormat.relationName(), att, 0);

		return result;
	}

	@Override
	protected Instances process(Instances instances) throws Exception {



		Instances result = getOutputFormat();

		String[] words=this.wordInfo.keySet().toArray(new String[0]);
		Arrays.sort(words);

		for(String word:words){
			// get the word vector
			WordRep wordRep=this.wordInfo.get(word);

			// We just consider valid words
			if(wordRep.numDoc>=this.minInstDocs){
				double[] values = new double[result.numAttributes()];


				for(String wordFeat:wordRep.wordSpace.keySet()){
					// only include valid features
					if(this.m_Dictionary.containsKey(wordFeat)){
						int attIndex=this.m_Dictionary.getInt(wordFeat);
						// we normalise the value by the number of documents
						values[attIndex]=((double)wordRep.wordSpace.getInt(wordFeat))/wordRep.numDoc;					
					}

					if(this.includeMetaData){
						for(Attribute metaAtt:this.metaData){
							String metaAttName=metaAtt.name();
							values[result.attribute(metaAttName).index()]= wordRep.metaData.getDouble(metaAtt.name())/wordRep.numDoc;
						}


					}

					if(this.includeWord2Vec){
						AbstractDoubleList word2VecVals=this.word2VecDict.getWordMap().get(word);
						int i=0;
						for(double w2val:word2VecVals){
							values[result.attribute("Word2Vec-"+i).index()]=w2val;
							i++;
						}
					}

				}

				if(this.reportNumDocs)
					values[result.numAttributes()-3]=wordRep.numDoc;

				if(this.reportWord){
					int wordNameIndex=result.attribute("WORD_NAME").index();
					values[wordNameIndex]=result.attribute(wordNameIndex).addStringValue(word);					
				}


				if(this.labelWords){
					int angerIndex=result.attribute("anger").index();
					int anticipationIndex=result.attribute("anticipation").index();
					int disgustIndex=result.attribute("disgust").index();
					int fearIndex=result.attribute("fear").index();
					int joyIndex=result.attribute("joy").index();
					int negativeIndex=result.attribute("negative").index();
					int positiveIndex=result.attribute("positive").index();
					int sadnessIndex=result.attribute("sadness").index();
					int surpriseIndex=result.attribute("surprise").index();
					int trustIndex=result.attribute("trust").index();


					Map<String, Integer> emoLabel=this.lex.getWord(word);
					if(emoLabel==null){
						values[angerIndex]= Utils.missingValue();
						values[anticipationIndex]= Utils.missingValue();
						values[disgustIndex]= Utils.missingValue();
						values[fearIndex]= Utils.missingValue();
						values[joyIndex]= Utils.missingValue();
						values[negativeIndex]= Utils.missingValue();
						values[positiveIndex]= Utils.missingValue();
						values[sadnessIndex]= Utils.missingValue();
						values[surpriseIndex]= Utils.missingValue();
						values[trustIndex]= Utils.missingValue();
						
						if(this.addNeutralLabel){
							int neutralIndex=result.attribute("neutral").index();
							values[neutralIndex]=Utils.missingValue();							
						}
							

					}
					else{
						values[angerIndex]= emoLabel.get("anger");
						values[anticipationIndex]= emoLabel.get("anticipation");
						values[disgustIndex]= emoLabel.get("disgust");
						values[fearIndex]= emoLabel.get("fear");
						values[joyIndex]= emoLabel.get("joy");
						values[negativeIndex]= emoLabel.get("negative");
						values[positiveIndex]= emoLabel.get("positive");
						values[sadnessIndex]= emoLabel.get("sadness");
						values[surpriseIndex]= emoLabel.get("surprise");
						values[trustIndex]= emoLabel.get("trust");
						
						if(this.addNeutralLabel){
							int neutralIndex=result.attribute("neutral").index();
							values[neutralIndex]=emoLabel.get("neutral");
						}
							
					}


				}


				Instance inst=new SparseInstance(1, values);


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


	public String getWordPrefix() {
		return wordPrefix;
	}


	public void setWordPrefix(String prefix) {
		this.wordPrefix = prefix;
	}


	public String getClustPrefix() {
		return clustPrefix;
	}



	public void setClustPrefix(String clustPrefix) {
		this.clustPrefix = clustPrefix;
	}


	public boolean isToLowerCase() {
		return toLowerCase;
	}

	public void setToLowerCase(boolean toLowerCase) {
		this.toLowerCase = toLowerCase;
	}


	public int getMinAttDocs() {
		return minAttDocs;
	}



	public void setMinAttDocs(int minAttDocs) {
		this.minAttDocs = minAttDocs;
	}

	public int getMinInstDocs() {
		return minInstDocs;
	}



	public void setMinInstDocs(int minInstDocs) {
		this.minInstDocs = minInstDocs;
	}


	public boolean isCreateWordAtts() {
		return createWordAtts;
	}



	public void setCreateWordAtts(boolean createWordAtts) {
		this.createWordAtts = createWordAtts;
	}

	public void setCreateClustAtts(boolean createClustAtts) {
		this.createClustAtts = createClustAtts;
	}


	public boolean isCreateClustAtts() {
		return createClustAtts;
	}




	public boolean isReportNumDocs() {
		return reportNumDocs;
	}



	public void setReportNumDocs(boolean reportNumDocs) {
		this.reportNumDocs = reportNumDocs;
	}


	public String getLexPath() {
		return lexPath;
	}



	public void setLexPath(String lexPath) {
		this.lexPath = lexPath;
	}



	public String getClustPath() {
		return clustPath;
	}



	public void setClustPath(String clustPath) {
		this.clustPath = clustPath;
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


	public String getPosPrefix() {
		return posPrefix;
	}



	public void setPosPrefix(String posPrefix) {
		this.posPrefix = posPrefix;
	}



	public boolean isCreatePosAtts() {
		return createPosAtts;
	}



	public void setCreatePosAtts(boolean createPosAtts) {
		this.createPosAtts = createPosAtts;
	}



	public boolean isIncludeMetaData() {
		return includeMetaData;
	}



	public void setIncludeMetaData(boolean includeMetaData) {
		this.includeMetaData = includeMetaData;
	}



	public boolean isIncludeWord2Vec() {
		return includeWord2Vec;
	}



	public void setIncludeWord2Vec(boolean includeWord2Vec) {
		this.includeWord2Vec = includeWord2Vec;
	}



	public String getWord2VecPath() {
		return word2VecPath;
	}



	public void setWord2VecPath(String word2VecPath) {
		this.word2VecPath = word2VecPath;
	}
	
	
	public boolean isAddNeutralLabel() {
		return addNeutralLabel;
	}



	public void setAddNeutralLabel(boolean addNeutralLabel) {
		this.addNeutralLabel = addNeutralLabel;
	}



	public static void main(String[] args) {
		runFilter(new WordCentroidMultiLabel(), args);
	}

}
