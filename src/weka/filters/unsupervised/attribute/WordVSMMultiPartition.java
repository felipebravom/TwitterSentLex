package weka.filters.unsupervised.attribute;


import it.unimi.dsi.fastutil.objects.AbstractObjectSet;
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
import java.util.Collections;
import java.util.Enumeration;
import java.util.List;
import java.util.ListIterator;
import java.util.Map;
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

public class WordVSMMultiPartition extends DistantSupervisionFilter{

	/**
	 * 	 Given a corpus of documents creates Vector Space model for each labelled word using multiple attributes
	 *  	 We just consider words labelled according to a lexicon
	 *   We create multiple instances for each word by partionating the posting list
	 *   Idea 1: Partionating I will end up having more instances for frequent words
	 *   Idea 2: Partionaning and setting a maximum 
	 *   Idea 3: Same number of instances per word, using sampling
	 */ 



	private static final long serialVersionUID = 7553647795494402690L;


	/** the vocabulary and the WordRep */
	protected Object2ObjectMap<String, WordRep> wordInfo; 

	/** Counts the number of documents in which candidate attributes appear */
	protected Object2IntMap<String> attributeCount;


	/** Contains a mapping of valid attribute with their indexes. */
	protected Object2IntMap<String> m_Dictionary;

	/** Brown Clusters Dictionary */
	protected Object2ObjectMap<String,String> brownDict;

	/** the minimum number of documents for an attribute to be considered. */
	protected int minAttDocs=0; 


	/** the minimum number of documents for a word to be included. */
	protected int minInstDocs=0; 


	/** the index of the string attribute to be processed */
	protected int textIndex=1; 
	
	/** the number of parititions in each centrod */
	protected int partNumber=-1;
	
	/** the prefix of the word attributes */
	protected String wordPrefix="WORD-";

	/** the prefix of the cluster-based attributes */
	protected String clustPrefix="CLUST-";

	/** the prefix of the POS-bases attributes */
	protected String posPrefix="POS-";


	/** True for calculating POS attributes. */
	protected boolean createPosAtts=true;


	/** True if all tokens should be downcased. */
	protected boolean toLowerCase=true;


	/** True for calculating word-based attributes . */
	protected boolean createWordAtts=true;


	/** True for calculating cluster-based attributes . */
	protected boolean createClustAtts=true;


	/** True if the number of documents where the word occurs is reported. */
	protected boolean reportNumDocs=false;



	/** True if the word name is included as an attribute */
	protected boolean reportWord=true;


	/** True is stopwords are discarded */
	protected boolean removeStopWords=false;

	/** The stopwords file */
	protected String stopWordsPath="resources/stopwords.txt";


	/** True if url, users, and repeated letters are cleaned */
	protected boolean cleanTokens=false;


	/** The path of the seed lexicon . */
	protected String lexPath="lexicons/AFINN-posneg.txt";

	/** The path of the word clusters. */
	protected String clustPath="resources/50mpaths2.txt";


	/** LexiconEvaluator for sentiment prefixes */
	protected LexiconEvaluator lex;

	/** The POS tagger. */
	protected Tagger tagger; 


	// This class contains all the information of the word to compute the centroid
	class WordRep{
		String word; // the word
		int numDoc; // number of documents where the word occurs		
		ObjectList<Object2IntMap<String>> postingList; 


		public WordRep(String word){
			this.word=word;
			this.numDoc=0;
			this.postingList=new ObjectArrayList<Object2IntMap<String>>();
		}

		public void addDoc(Object2IntMap<String> docVector){
			this.postingList.add(docVector);
			this.numDoc++;
		}


		// returns a list of partions of the posting List
		public ObjectList<ObjectList<Object2IntMap<String>>> partitionate(int partSize){

			ObjectList<ObjectList<Object2IntMap<String>>> resList= new ObjectArrayList<ObjectList<Object2IntMap<String>>>();

			// if the partition size is larger than the posting list, then put the whole list into one partition
			// if partsize is less or equal than zero we create one single partition too, which is equivalent as the full
			// tweet centroid model
			if(partSize>=this.postingList.size() || partSize <=0){
				resList.add(this.postingList);
			}
			else{
				int i=0;
				while(i+partSize<=this.postingList.size()){
					resList.add(this.postingList.subList(i, i+partSize));
					i+=partSize;				
				}
				if(i<this.postingList.size()&& i+partSize>this.postingList.size() ){
					resList.add(this.postingList.subList(i, this.postingList.size()));
				}
				
			}
			
			

			return resList;

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

		
		result.addElement(new Option("\t The size of the partitions.\n"
				+ "\t(default: " + this.partNumber + ")", "U", 1, "-U"));

		

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
		
		result.add("-U");
		result.add(""+this.getPartNumber());
		

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

		
		String partNumberOption = Utils.getOption('U', options);
		if (textIndexOption.length() > 0) {
			String[] partNumberSpec = Utils.splitOptions(partNumberOption);
			if (partNumberSpec.length == 0) {
				throw new IllegalArgumentException(
						"Invalid index");
			}
			int partNumberValue = Integer.parseInt(partNumberSpec[0]);
			this.setPartNumber(partNumberValue);

		}
		
		


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


	// Maps a given instance with an attribute called content into a target instance with the same dimensions
	public Instances mapTargetInstance(Instances inp){

		// Creates instances with the same format
		Instances result=getOutputFormat();

		Attribute contentAtt=inp.attribute("content");
		Attribute attClassInp=inp.attribute("Class");

		Attribute attClassOut=result.attribute("Class");


		for(Instance inst:inp){
			String content=inst.stringValue(contentAtt);

			String classValue=attClassInp.value((int)inst.value(attClassInp));


			List<String> tokens=this.tokenize(content); 

			// Identifies the distinct terms
			AbstractObjectSet<String> terms=new  ObjectOpenHashSet<String>(); 
			terms.addAll(tokens);


			Object2IntMap<String> docVec=this.calculateDocVec(tokens);

			double[] values = new double[result.numAttributes()];


			values[attClassOut.index()]= attClassOut.indexOfValue(classValue);

			for(String att:docVec.keySet()){

				if(this.m_Dictionary.containsKey(att)){
					int attIndex=this.m_Dictionary.getInt(att);
					// we normalise the value by the number of documents
					values[attIndex]=docVec.getInt(att);					
				}


			}



			int wordNameIndex=result.attribute("WORD_NAME").index();
			values[wordNameIndex]=result.attribute(wordNameIndex).addStringValue(content);	

			Instance outInst=new SparseInstance(1, values);

			inst.setDataset(result);

			result.add(outInst);

		}

		return result;

	}


	public Object2IntMap<String> calculateDocVec(List<String> tokens) {

		Object2IntMap<String> docVec = new Object2IntOpenHashMap<String>();
		// add the word-based vector
		if(this.createWordAtts)
			docVec.putAll(calculateTermFreq(tokens,this.wordPrefix));

		if(this.createClustAtts){
			// calcultates the vector of clusters
			List<String> brownClust=clustList(tokens,brownDict);
			docVec.putAll(calculateTermFreq(brownClust,this.clustPrefix));			
		}	

		if(this.createPosAtts){
			List<String> posTags=MyUtils.getPOStags(tokens, this.tagger);
			docVec.putAll(calculateTermFreq(posTags,this.posPrefix));
		}


		return docVec;

	}






	/* Calculates the vocabulary and the word vectors from an Instances object
	 * The vocabulary is only extracted the first time the filter is run.
	 * 
	 */	 
	public void computeWordVecsAndVoc(Instances inputFormat) {

		this.lex = new LexiconEvaluator(	this.lexPath);
		
		try {
			this.lex.processDict();
		} catch (IOException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}

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




			// reference to the content of the message, users index start from zero
			Attribute attrCont = inputFormat.attribute(this.textIndex-1);

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


					String value=this.lex.retrieveValue(word);
										
					if(value.equals("positive")||value.equals("negative")){
							
						if (this.wordInfo.containsKey(word)) {
							WordRep wordRep=this.wordInfo.get(word);
							wordRep.addDoc(docVec); // add the document

						} else{
							WordRep wordRep=new WordRep(word);
							wordRep.addDoc(docVec); // add the document
							this.wordInfo.put(word, wordRep);						
						}

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


		// The target label

		ArrayList<String> label = new ArrayList<String>();
		label.add("negative");
		label.add("positive");
		att.add(new Attribute("Class", label));





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

				// a list of lists of tweet vectors
				ObjectList<ObjectList<Object2IntMap<String>>> partitions=wordRep.partitionate(this.getPartNumber());
				
				// traverse the partitions
				for(ObjectList<Object2IntMap<String>> tweetPartition:partitions){

					// create one instance per partition	
					double[] values = new double[result.numAttributes()];


					// average the vectors of the tweets in the partition
					// traverse each feature space in the partition
					for(Object2IntMap<String> wordSpace:tweetPartition){

						for(String innerWord:wordSpace.keySet()){
							// only include valid words
							if(this.m_Dictionary.containsKey(innerWord)){
								int attIndex=this.m_Dictionary.getInt(innerWord);
								// we normalise the value by the number of documents
								values[attIndex]+=((double)wordSpace.getInt(innerWord))/tweetPartition.size();					
							}
						}
					}


					if(this.reportNumDocs)
						values[result.numAttributes()-3]=wordRep.numDoc;

					if(this.reportWord){
						int wordNameIndex=result.attribute("WORD_NAME").index();
						values[wordNameIndex]=result.attribute(wordNameIndex).addStringValue(word);					
					}


					String wordPol=this.lex.retrieveValue(word);
					if(wordPol.equals("negative"))
						values[result.numAttributes()-1]=0;
					else if(wordPol.equals("positive"))
						values[result.numAttributes()-1]=1;
					else
						values[result.numAttributes()-1]= Utils.missingValue();					



					Instance inst=new SparseInstance(1, values);


					inst.setDataset(result);

					result.add(inst);




				}



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

	

	public int getPartNumber() {
		return partNumber;
	}



	public void setPartNumber(int partNumber) {
		this.partNumber = partNumber;
	}



	public static void main(String[] args) {
		runFilter(new WordVSMMultiPartition(), args);
	}

}
