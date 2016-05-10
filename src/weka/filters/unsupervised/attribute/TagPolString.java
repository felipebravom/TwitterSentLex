package weka.filters.unsupervised.attribute;


import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Enumeration;
import java.util.List;
import java.util.ListIterator;
import java.util.Map;
import java.util.TreeMap;
import java.util.Vector;

import lexexpand.core.LexiconEvaluator;
import lexexpand.core.MyUtils;
import cmu.arktweetnlp.Tagger;
import cmu.arktweetnlp.Tagger.TaggedToken;
import cmu.arktweetnlp.Twokenize;
import cmu.arktweetnlp.impl.ModelSentence;
import cmu.arktweetnlp.impl.Sentence;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.DenseInstance;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.SparseInstance;
import weka.core.Utils;
import weka.core.Capabilities.Capability;
import weka.filters.SimpleBatchFilter;

public class TagPolString extends SimpleBatchFilter {

	/**  Converts one String attribute into a set of attributes
	 * representing word occurrence based on the TwitterNLP tokenizer.
	 * 
	 */


	/** for serialization */
	private static final long serialVersionUID = 3635946466523698211L;


	/** the index of the string attribute to be processed */
	protected int textIndex=1; 


	/** True if all words should be downcased. */
	protected boolean toLowerCase=true;



	/** True if a part-of-speech prefix should be included to each word */
	protected boolean posPrefix=true;


	/** True if a Sentiment prefix calculatef from a Lexicon should be included to each word */
	protected boolean sentPrefix=false;


	/** TwitterNLP Tagger model */
	protected Tagger tagger;


	/** LexiconEvaluator for sentiment prefixes */
	protected LexiconEvaluator lex;



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

		result.addElement(new Option("\t Index of string attribute.\n"
				+ "\t(default: " + this.textIndex + ")", "I", 1, "-I"));		



		result.addElement(new Option("\t Lowercase content.\n"
				+ "\t(default: " + this.toLowerCase + ")", "L", 0, "-L"));

		result.addElement(new Option("\t POS prefix.\n"
				+ "\t(default: " + this.posPrefix + ")", "K", 0, "-K"));


		result.addElement(new Option("\t Sent prefix.\n"
				+ "\t(default: " + this.sentPrefix + ")", "H", 0, "-H"));


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

		result.add("-I");
		result.add("" + this.getTextIndex());


		if(this.toLowerCase)
			result.add("-L");


		if(this.posPrefix)
			result.add("-K");

		if(this.sentPrefix)
			result.add("-H");


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


		this.posPrefix=Utils.getFlag('K', options);

		this.sentPrefix=Utils.getFlag('H', options);

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



	@Override
	protected Instances determineOutputFormat(Instances inputFormat) {

		Instances result = new Instances(inputFormat, 0);
		result.setClassIndex(inputFormat.classIndex());
		return result;

		// set the class index

	}

	@Override
	protected Instances process(Instances instances) throws Exception {

		Instances result = getOutputFormat();

		// process the Tagger
		if(this.posPrefix){
			try {
				this.tagger= new Tagger();
				this.tagger.loadModel("models/model.20120919");
			} catch (IOException e) {
				this.posPrefix=false;
				System.err.println("Warning: TwitterNLP model couldn't be read.");
			}	

		}
		
		// process the LexiconEvaluator
		if(this.sentPrefix){	
			try {
				this.lex= new LexiconEvaluator("lexicons/cleanLex.csv");
				this.lex.processDict();
			} catch (IOException e) {
				this.sentPrefix=false;
				System.err.println("Warning: Lexicon couldn't be read.");
			}
		}

		for (int i = 0; i < instances.numInstances(); i++) {
			double[] values = new double[result.numAttributes()];
			for (int n = 0; n < instances.numAttributes(); n++){
				if(n==this.textIndex-1){
					
					String content=instances.instance(i).stringValue(n);

					if(this.toLowerCase)
						content=content.toLowerCase();

					// tokenises the content 
					List<String> tokens=Twokenize.tokenizeRawTweetText(content);; 
					List<String> posTokens = null;
					List<String> sentTokens = null;


					if(this.posPrefix){
						try{
							posTokens=MyUtils.getPOStags(tokens, tagger);
						}
						catch(Exception E){

						}
					}


					if(this.sentPrefix){
						sentTokens=new ArrayList<String>();
						for(String token:tokens){
							String sentToken="";
							if(this.lex.getDict().containsKey(token)){
								String value=this.lex.getDict().get(token);
								sentToken +=  value+"-";						
							}

							sentTokens.add(sentToken);
						}


					}

					if(this.posPrefix){
						for(int j=0; j<tokens.size();j++){
							tokens.set(j, posTokens.get(j)+"-"+tokens.get(j));
						}
					}

					if(this.sentPrefix){
						for(int j=0; j<tokens.size();j++){
							tokens.set(j, sentTokens.get(j)+tokens.get(j));
						}
					}

					StringBuilder build=new StringBuilder();
					for(String t:tokens){
						build.append(t+" ");
					}							


					values[n] = result.attribute(n).addStringValue(build.toString());
				}
				else{
					values[n] = instances.instance(i).value(n);					
				}


			}	

			Instance inst=new DenseInstance(1, values);


			//inst.setDataset(result);
			// copy possible strings, relational values...
			//copyValues(inst, false, instances, result);

			result.add(inst);




		}

		return result;
	}

	protected String taggedString(String sentence){
		StringBuilder build=new StringBuilder();
		List<TaggedToken> tokens=this.tagger.tokenizeAndTag(sentence);
		for(TaggedToken t:tokens){
			build.append(t.tag+"-"+t.token+" ");
		}		
		return build.toString();
	}




	public int getTextIndex() {
		return textIndex;
	}


	public void setTextIndex(int textIndex) {
		this.textIndex = textIndex;
	}



	public boolean isToLowerCase() {
		return toLowerCase;
	}

	public void setToLowerCase(boolean toLowerCase) {
		this.toLowerCase = toLowerCase;
	}







	public boolean isPosPrefix() {
		return posPrefix;
	}



	public void setPosPrefix(boolean posPrefix) {
		this.posPrefix = posPrefix;
	}

	public boolean isSentPrefix() {
		return sentPrefix;
	}



	public void setSentPrefix(boolean sentPrefix) {
		this.sentPrefix = sentPrefix;
	}




	public static void main(String[] args) {
		runFilter(new TagPolString(), args);
	}

}
