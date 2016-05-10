package weka.filters.unsupervised.attribute;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Enumeration;
import java.util.List;
import java.util.Map;
import java.util.Vector;

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

public class LexExpandTestFilter extends SimpleBatchFilter {

	/**
	 * This Filter receives two path Files 
	 */
	private static final long serialVersionUID = 4983739424598292130L;
	
	protected String seedPath="lexicons/metaLexEmo.csv"; // Path of seed Lexicon
	protected String exPath="lexGen.csv"; // Path of expanded Lexicon
	
	protected int posThres=0; // Positive threshold for including positive words
	protected int negThres=0; // Negative threshold for including negative words

	
	


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

		result.addElement(new Option("\t Positive count threshold.\n"
				+ "\t(default: " + this.posThres + ")", "P", 1, "-P"));
		
		result.addElement(new Option("\t Negative count threshold.\n"
				+ "\t(default: " + this.negThres + ")", "N", 1, "-N"));


		

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
		
		result.add("-P");
		result.add("" + this.getPosThres());
		
		result.add("-N");
		result.add("" + this.getNegThres());



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
		
		String posThresPathOption = Utils.getOption('P', options);
		if (posThresPathOption.length() > 0) {
			String[] posThresPathOptionSpec = Utils.splitOptions(posThresPathOption);
			if (posThresPathOptionSpec.length == 0) {
				throw new IllegalArgumentException(
						"Invalid index");
			}
			int posThresValue=Integer.parseInt(posThresPathOptionSpec[0]);
			this.setPosThres(posThresValue);
		}
		
		String negThresPathOption = Utils.getOption('N', options);
		if (negThresPathOption.length() > 0) {
			String[] negThresPathOptionSpec = Utils.splitOptions(negThresPathOption);
			if (negThresPathOptionSpec.length == 0) {
				throw new IllegalArgumentException(
						"Invalid index");
			}
			int negThresValue=Integer.parseInt(negThresPathOptionSpec[0]);
			this.setNegThres(negThresValue);
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

		att.add(new Attribute("SEED-PW")); // Seed Lexicon Positive words
		att.add(new Attribute("SEED-NW")); // Seed Lexicon Negative words
		
		att.add(new Attribute("EX-PS")); // Expanded Pos Score
		att.add(new Attribute("EX-NS")); // Expanded Negative Score
		
		

		Instances result = new Instances("Twitter Sentiment Analysis", att, 0);

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

		LexiconEvaluator seedLex = new LexiconEvaluator(
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

			Map<String, Integer> seedLexCounts = seedLex
					.evaluatePolarityLexicon(words);
			values[result.attribute("SEED-PW").index()] = seedLexCounts
					.get("posCount");
			values[result.attribute("SEED-NW").index()] = seedLexCounts
					.get("negCount");

			
	
				
			Map<String,Double> edScores=expandedLex.evaluatePolThres(words,this.getPosThres(),this.getNegThres());
			
			values[result.attribute("EX-PS").index()] = edScores.get("posScore");
			values[result.attribute("EX-NS").index()] = edScores.get("negScore");
			

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


	public int getPosThres() {
		return posThres;
	}


	public void setPosThres(int posThres) {
		this.posThres = posThres;
	}


	public int getNegThres() {
		return negThres;
	}


	public void setNegThres(int negThres) {
		this.negThres = negThres;
	}
	
	
	
	
	

}
