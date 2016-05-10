package weka.filters.unsupervised.attribute;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Enumeration;
import java.util.List;
import java.util.Map;
import java.util.Vector;

import lexexpand.core.EmotionEvaluator;
import lexexpand.core.ExpandLexEvaluator;
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

public class EmoLexExpandTestFilter extends SimpleBatchFilter {

	/**
	 * This Filter receives two path Files 
	 */
	private static final long serialVersionUID = 4983739424598292130L;
	
	protected String seedPath="lexicons/NRC-emotion-lexicon-wordlevel-v0.92.txt"; // Path of seed Lexicon
	protected String exPath="lexBR.csv"; // Path of expanded Lexicon
	
	
	


	@Override
	public String globalInfo() {
		return "A batch filter that calcuates attributes from different lexical resources for Sentiment Analysis ";

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
		
		result.addElement(new Option("\t Seed lexicon path.\n"
				+ "\t(default: " + this.seedPath + ")", "S", 1, "-S"));

		result.addElement(new Option("\t Expanded lexicon path.\n"
				+ "\t(default: " + this.exPath + ")", "E", 1, "-E"));		

			

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

		result.add("-S");
		result.add("" + this.getSeedPath());
		
		result.add("-E");
		result.add("" + this.getExPath());
		


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
		
		
		String seedPathOption = Utils.getOption('S', options);
		if (seedPathOption.length() > 0) {
			String[] seedPathOptionSpec = Utils.splitOptions(seedPathOption);
			if (seedPathOptionSpec.length == 0) {
				throw new IllegalArgumentException(
						"Invalid index");
			}
			this.setSeedPath(seedPathOptionSpec[0]);
		}
		
		
		String exPathOption = Utils.getOption('E', options);
		if (exPathOption.length() > 0) {
			String[] exPathOptionSpec = Utils.splitOptions(exPathOption);
			if (exPathOptionSpec.length == 0) {
				throw new IllegalArgumentException(
						"Invalid index");
			}
			this.setExPath(exPathOptionSpec[0]);
		}
	
		

		super.setOptions(options);

		Utils.checkForRemainingOptions(options);


	}


	@Override
	protected Instances determineOutputFormat(Instances inputFormat)
			throws Exception {

		ArrayList<Attribute> att = new ArrayList<Attribute>();

		// Adds all attributes of the inputformat
		for (int i = 0; i < inputFormat.numAttributes(); i++) {
			att.add(inputFormat.attribute(i));
		}

		att.add(new Attribute("SEED-anger")); // Seed Lexicon Positive words
		att.add(new Attribute("SEED-anticipation")); // Seed Lexicon Negative words
		att.add(new Attribute("SEED-disgust")); // Seed Lexicon Positive words
		att.add(new Attribute("SEED-fear")); // Seed Lexicon Negative words
		att.add(new Attribute("SEED-joy")); // Seed Lexicon Positive words
		att.add(new Attribute("SEED-sadness")); // Seed Lexicon Negative words
		att.add(new Attribute("SEED-surprise")); // Seed Lexicon Positive words
		att.add(new Attribute("SEED-trust")); // Seed Lexicon Negative words
		att.add(new Attribute("SEED-negative")); // Seed Lexicon Positive words
		att.add(new Attribute("SEED-positive")); // Seed Lexicon Negative words
		
		att.add(new Attribute("EX-anger")); // EX Lexicon Positive words
		att.add(new Attribute("EX-anticipation")); // EX Lexicon Negative words
		att.add(new Attribute("EX-disgust")); // EX Lexicon Positive words
		att.add(new Attribute("EX-fear")); // EX Lexicon Negative words
		att.add(new Attribute("EX-joy")); // EX Lexicon Positive words
		att.add(new Attribute("EX-sadness")); // EX Lexicon Negative words
		att.add(new Attribute("EX-surprise")); // EX Lexicon Positive words
		att.add(new Attribute("EX-trust")); // EX Lexicon Negative words
		att.add(new Attribute("EX-negative")); // EX Lexicon Positive words
		att.add(new Attribute("EX-positive")); // EX Lexicon Negative words
		

		Instances result = new Instances("Twitter Emotion Analysis", att, 0);

		// set the class index
		result.setClassIndex(inputFormat.classIndex());

		return result;
	}

	@Override
	protected Instances process(Instances instances) throws Exception {
		// Instances result = new Instances(determineOutputFormat(instances),
		// 0);

		Instances result = getOutputFormat();

		// reference to the content of the tweet
		Attribute attrCont = instances.attribute("content");

		EmotionEvaluator seedLex = new EmotionEvaluator(
				this.seedPath);
		seedLex.processDict();
		
		ExpandLexEvaluator expandedLex=new ExpandLexEvaluator(this.exPath);
		expandedLex.processDict();
		

		
	
		for (int i = 0; i < instances.numInstances(); i++) {
			double[] values = new double[result.numAttributes()];
			for (int n = 0; n < instances.numAttributes(); n++)
				values[n] = instances.instance(i).value(n);

			String content = instances.instance(i).stringValue(attrCont);
			content=content.toLowerCase();
			
			List<String> words =Twokenize.tokenizeRawTweetText(content);

			Map<String, Integer> seedLexEmo = seedLex
					.evaluateEmotion(words);
			
			values[result.attribute("SEED-anger").index()] = seedLexEmo
					.get("anger");
			values[result.attribute("SEED-anticipation").index()] = seedLexEmo
					.get("anticipation");
			values[result.attribute("SEED-disgust").index()] = seedLexEmo
					.get("disgust");
			values[result.attribute("SEED-fear").index()] = seedLexEmo
					.get("fear");
			values[result.attribute("SEED-joy").index()] = seedLexEmo
					.get("joy");
			values[result.attribute("SEED-sadness").index()] = seedLexEmo
					.get("sadness");
			values[result.attribute("SEED-surprise").index()] = seedLexEmo
					.get("surprise");
			values[result.attribute("SEED-trust").index()] = seedLexEmo
					.get("trust");
			values[result.attribute("SEED-negative").index()] = seedLexEmo
					.get("negative");
			values[result.attribute("SEED-positive").index()] = seedLexEmo
					.get("positive");

						
	
				
			Map<String,Double> exLexEmo=expandedLex.evaluateEmotion(words);
			
			values[result.attribute("EX-anger").index()] = exLexEmo
					.get("anger");
			values[result.attribute("EX-anticipation").index()] = exLexEmo
					.get("anticipation");
			values[result.attribute("EX-disgust").index()] = exLexEmo
					.get("disgust");
			values[result.attribute("EX-fear").index()] = exLexEmo
					.get("fear");
			values[result.attribute("EX-joy").index()] = exLexEmo
					.get("joy");
			values[result.attribute("EX-sadness").index()] = exLexEmo
					.get("sadness");
			values[result.attribute("EX-surprise").index()] = exLexEmo
					.get("surprise");
			values[result.attribute("EX-trust").index()] = exLexEmo
					.get("trust");
			values[result.attribute("EX-negative").index()] = exLexEmo
					.get("negative");
			values[result.attribute("EX-positive").index()] = exLexEmo
					.get("positive");

			Instance inst = new DenseInstance(1, values);

			inst.setDataset(result);

			// copy possible strings, relational values...
			copyValues(inst, false, instances, result);

			result.add(inst);

		}

		return result;
	}
	
	
	public String getSeedPath() {
		return seedPath;
	}

	public void setSeedPath(String seedPath) {
		this.seedPath = seedPath;
	}

	public String getExPath() {
		return exPath;
	}

	public void setExPath(String exPath) {
		this.exPath = exPath;
	}


	
	
	
	
	

}
