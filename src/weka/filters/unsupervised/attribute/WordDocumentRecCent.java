package weka.filters.unsupervised.attribute;


import it.unimi.dsi.fastutil.objects.AbstractObjectList;
import it.unimi.dsi.fastutil.objects.AbstractObjectSet;
import it.unimi.dsi.fastutil.objects.Object2DoubleMap;
import it.unimi.dsi.fastutil.objects.Object2DoubleOpenHashMap;
import it.unimi.dsi.fastutil.objects.Object2IntMap;
import it.unimi.dsi.fastutil.objects.Object2IntOpenHashMap;
import it.unimi.dsi.fastutil.objects.Object2ObjectMap;
import it.unimi.dsi.fastutil.objects.Object2ObjectOpenHashMap;
import it.unimi.dsi.fastutil.objects.ObjectArrayList;
import it.unimi.dsi.fastutil.objects.ObjectOpenHashSet;
import it.unimi.dsi.fastutil.objects.ObjectSet;

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
import weka.filters.SimpleBatchFilter;

public class WordDocumentRecCent extends SimpleBatchFilter {

	/**
	 * 	 Given a corpus of documents creates Vector Space model for each word using bag-of-words and cluster-based attributes
	 * 	 words are labelled to sentiment labels using a seed lexicon	
	 * 
	 */



	private static final long serialVersionUID = 7553647795494402690L;


	/** the vocabulary and the WordRep */
	protected Object2ObjectMap<String, WordRep> wordInfo; 

	/** Counts the number of documents in which candidate attributes appear */
	protected Object2IntMap<String> attributeCount;


	/** Contains a mapping of valid attribute with their indexes. */
	protected Object2IntMap<String> m_Dictionary;
	
	/** the corpus of documents. */
	protected AbstractObjectList<DocRep> corpus; 

	/** Brown Clusters Dictionary */
	protected Object2ObjectMap<String,String> brownDict;
	
	/** Number of iterations */
	protected int iterations=1;


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


	/** True if all tokens should be downcased. */
	protected boolean toLowerCase=true;


	/** True for calculating word-based attributes . */
	protected boolean createWordAtts=true;


	/** True for calculating cluster-based attributes . */
	protected boolean createClustAtts=true;


	/** True if the number of documents where the word occurs is reported. */
	protected boolean reportNumDocs=false;


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

	/** The path of the word clusters. */
	protected String clustPath="resources/50mpaths2.txt";


	/** LexiconEvaluator for sentiment prefixes */
	protected LexiconEvaluator lex;


	public String minAttDocsTipText() {
		return "The minimum number of documents for an attribute to be considered.";
	}
	
	public String minInstDocsTipText() {
		return "The minimum number of documents for a word to be included. ";
	}
	
	public String textIndexTipText() {
		return "The index of the string attribute to be processed.";
	}

	public String wordPrefixTipText() {
		return "The prefix of the word attributes.";
	}
	
	public String clustPrefixTipText() {
		return "The prefix of the cluster-based attributes.";
	}
	
	public String toLowerCaseTipText() {
		return "True if all tokens should be downcased.";
	}
	
	public String createWordAttsTipText() {
		return "True for calculating word-based attributes.";
	}

	public String createClustAttsTipText() {
		return "True for calculating cluster-based attributes.";
	}
	
	public String reportNumDocsTipText() {
		return "True if the number of documents where the word occurs is reported.";
	}
	
	public String labelWordTipText() {
		return "True is the words will be labelled with the seed lexicon.";
	}
	

	public String reportWordTipText() {
		return "True if the word name is included as an attribute.";
	}
	
	public String removeStopWordsTipText() {
		return "True is stopwords are discarded.";
	}
	
	public String stopWordsPathTipText() {
		return "The stopwords file.";
	}
	
	public String cleanTokensTipText() {
		return "True if url, users, and repeated letters are cleaned.";
	}
	
	public String lexPathTipText() {
		return "The path of the seed lexicon.";
	}
	
	public String clustPathTipText() {
		return "The path of the word clusters.";
	}
	
	
	// This class contains all the information of the word to compute the centroid
		class WordRep{
			String word; // the word
			AbstractObjectList<DocRep> docList; 
			Object2DoubleMap<String> wordVector; // the vector space model of the word



			public WordRep(String word){
				this.word=word;
				this.docList=new ObjectArrayList<DocRep>();
				this.wordVector=new Object2DoubleOpenHashMap<String>();
			}
			
			
			public void addDoc(DocRep doc){
				this.docList.add(doc);

			}
			
			

			// Calculates the centroid considering only valid Attributes			
			public void calcCent(ObjectSet<String> validAtt){
				
				this.wordVector.clear();
				
				for(DocRep doc:this.docList){					
					for(String innerAtt:doc.docVector.keySet()){
						
						if(validAtt.contains(innerAtt)){
							double innerAttValue=doc.docVector.getDouble(innerAtt)/this.docList.size();
							// if the word was seen before we add the current frequency
							this.wordVector.put(innerAtt,innerAttValue+this.wordVector.getDouble(innerAtt));							
						}
						
					
					}	

				}
				
				
			}
			
			



		}
		
		
		class DocRep{

			AbstractObjectList<WordRep> wordList; 
			Object2DoubleMap<String> docVector; // the vector space model of the document

			
			public DocRep(Object2DoubleMap<String> docVector){
				this.docVector=docVector;
				this.wordList=new ObjectArrayList<WordRep>();
								
			}
			
			public void addWord(WordRep wordRep){
				this.wordList.add(wordRep);
			}
			
			
			
			// calculate the centroid of the document
			public void calcCent(int minFreqDoc){			
				
				docVector.clear();
				
				// calculate number of valid words
				int validWords=0;
				for(WordRep wordRep:this.wordList){
					if(wordRep.docList.size()>=minFreqDoc)
						validWords++;
				}
				
				for(WordRep wordRep:this.wordList){
					
					if(wordRep.docList.size()>=minFreqDoc){
						for(String innerAtt:wordRep.wordVector.keySet()){
							double innerAttValue=wordRep.wordVector.getDouble(innerAtt);
							// if the word was seen before we add the current frequency
							this.docVector.put(innerAtt,innerAttValue/validWords+this.docVector.getDouble(innerAtt)); // SLOW rehash
						}	
						
					}
					
		

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


	public Object2DoubleMap<String> calculateTermFreq(List<String> tokens, String prefix) {
		Object2DoubleMap<String> termFreq = new Object2DoubleOpenHashMap<String>();

		// Traverse the strings and increments the counter when the token was
		// already seen before
		for (String token : tokens) {
			termFreq.put(prefix+token, termFreq.getDouble(prefix+token) + 1);			
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


	public Object2DoubleMap<String> calculateDocVec(List<String> tokens) {

		Object2DoubleMap<String> docVec = new Object2DoubleOpenHashMap<String>();
		// add the word-based vector
		if(this.createWordAtts)
			docVec.putAll(calculateTermFreq(tokens,this.wordPrefix));

		if(this.createClustAtts){
			// calcultates the vector of clusters
			List<String> brownClust=clustList(tokens,brownDict);
			docVec.putAll(calculateTermFreq(brownClust,this.clustPrefix));			
		}	

		return docVec;

	}






	/* Calculates the vocabulary and the word vectors from an Instances object
	 * The vocabulary is only extracted the first time the filter is run.
	 * 
	 */	 
	public void computeWordVecsAndVoc(Instances inputFormat) {


		if (!this.isFirstBatchDone()){


			// the list of documents
			this.corpus= new ObjectArrayList<DocRep> (); 
			
			// the list of words using the string value as key
			this.wordInfo = new Object2ObjectOpenHashMap<String, WordRep>();
			
			// counts the numbers of objects where the sparse attribute is observed
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
					// do not create clusters attributes in case of exception
					this.setCreateClustAtts(false);
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


				Object2DoubleMap<String> docVec=this.calculateDocVec(tokens);			


				// creates a DocRep object
				DocRep docRep=new DocRep(docVec);
				this.corpus.add(docRep);


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



				// if the word is new we add it to the vocabulary		
				for (String word : terms) {
					
					if (this.wordInfo.containsKey(word)) {
						WordRep wordRep=this.wordInfo.get(word);
						wordRep.addDoc(docRep); // add the word to the list of words
						docRep.addWord(wordRep); // add the wordRep to the DocRep

					} else{
						WordRep wordRep=new WordRep(word);
						wordRep.addDoc(docRep); // add the document
						this.wordInfo.put(word, wordRep);	
						docRep.addWord(wordRep);
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


		// the dictionary of valid attribute and their indexes
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

		// this loop should be repeated  mutiple times
		for(int i=0;i<this.iterations;i++){
			for(WordRep wordRep:this.wordInfo.values()){
				if(wordRep.docList.size()>=this.minInstDocs){
					wordRep.calcCent(this.m_Dictionary.keySet());
				}			
			}
			for(DocRep docRep:this.corpus){
				docRep.calcCent(this.minInstDocs);
			}			
		}
		

		
		
		
		for(String word:this.wordInfo.keySet()){
			// get the word vector
			WordRep wordRep=this.wordInfo.get(word);

			// We just consider valid words
			if(wordRep.docList.size()>=this.minInstDocs){
		
				double[] values = new double[result.numAttributes()];


				
				for(String innerAtt:wordRep.wordVector.keySet()){
					// only include valid words
					if(this.m_Dictionary.containsKey(innerAtt)){
						int attIndex=this.m_Dictionary.get(innerAtt);
						values[attIndex] = wordRep.wordVector.getDouble(innerAtt);	
										
					}
				}

				if(this.reportNumDocs)
					values[result.numAttributes()-3]=wordRep.docList.size();

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
	
	public int getIterations() {
		return iterations;
	}

	public void setIterations(int iterations) {
		this.iterations = iterations;
	}




	public static void main(String[] args) {
		runFilter(new WordDocumentRecCent(), args);
	}

}
