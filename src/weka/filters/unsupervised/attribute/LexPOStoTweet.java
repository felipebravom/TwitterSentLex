package weka.filters.unsupervised.attribute;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import lexexpand.core.ExpandLexEvaluator;
import lexexpand.core.LexiconEvaluator;
import lexexpand.core.MyUtils;
import cmu.arktweetnlp.Tagger;
import cmu.arktweetnlp.Twokenize;
import cmu.arktweetnlp.Tagger.TaggedToken;
import uk.ac.wlv.sentistrength.SentiStrength;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SparseInstance;
import weka.core.Capabilities.Capability;
import weka.filters.SimpleBatchFilter;

public class LexPOStoTweet extends SimpleBatchFilter {

	/**
	 * 
	 */
	private static final long serialVersionUID = 4983739424598292130L;

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
	protected Instances determineOutputFormat(Instances inputFormat)
			throws Exception {

		ArrayList<Attribute> att = new ArrayList<Attribute>();

		// Adds all attributes of the inputformat
		for (int i = 0; i < inputFormat.numAttributes(); i++) {
			att.add(inputFormat.attribute(i));
		}

		att.add(new Attribute("LEX-CL-PW")); // MetaLex Positive words
		att.add(new Attribute("LEX-CL-NW")); // MetaLex Negative words

		att.add(new Attribute("LEX-SS-PS")); // SentiStrength Positive Score
		att.add(new Attribute("LEX-SS-NS")); // SentiStregtn Negative Score
		
		
		att.add(new Attribute("LEX-SW-PS")); // SW Positive Score
		att.add(new Attribute("LEX-SW-NS")); // SW Negative Score
		
		att.add(new Attribute("LEX-EDEM-PS")); // Edinburgh EM Pos Score
		att.add(new Attribute("LEX-EDEM-NS")); // EdinBurgh EM Negative Score		

		att.add(new Attribute("LEX-EDFU-PS")); // Edinburgh FUZZ Pos Score
		att.add(new Attribute("LEX-EDFU-NS")); // EdinBurgh FUZZ Negative Score


		att.add(new Attribute("LEX-EDT07-PS")); // Edinburgh T07 Pos Score
		att.add(new Attribute("LEX-EDT07-NS")); // EdinBurgh T07 Negative Score

		att.add(new Attribute("LEX-S140-PS")); // s140 Pos Score
		att.add(new Attribute("LEX-S140-NS")); // s140 Negative Score




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

		LexiconEvaluator cleanLex = new LexiconEvaluator(
				"lexicons/MetaLex.csv");
		cleanLex.processDict();

		ExpandLexEvaluator swLex=new ExpandLexEvaluator("lexicons/SentiWordNetLexFormat.csv");
		swLex.processDict();
		
		 SentiStrength sentiStrength = new SentiStrength();
		 String sentiParams[] = {"sentidata", "lexicons/SentiStrength/", "trinary"};
		 sentiStrength.initialise(sentiParams);	
		
		
		ExpandLexEvaluator edEMLex=new ExpandLexEvaluator("lexicons/EDLex.csv");
		//ExpandLexEvaluator edEMLex=new ExpandLexEvaluator("lexicons/EDLexNoOut.csv");
		edEMLex.processDict();

		ExpandLexEvaluator edFuzzLex=new ExpandLexEvaluator("lexicons/edimFuzzLex.csv");
		//ExpandLexEvaluator edFuzzLex=new ExpandLexEvaluator("lexicons/edimFuzzLexNOOUT.csv");
		edFuzzLex.processDict();

		ExpandLexEvaluator edTh07Lex=new ExpandLexEvaluator("lexicons/edimThres07Lex.csv");
		//ExpandLexEvaluator edTh07Lex=new ExpandLexEvaluator("lexicons/edimThres07LexNOOUT.csv");
		edTh07Lex.processDict();


		ExpandLexEvaluator s140Lex=new ExpandLexEvaluator("lexicons/STSLex.csv");
		//ExpandLexEvaluator s140Lex=new ExpandLexEvaluator("lexicons/STSLexNoOut.csv");
		s140Lex.processDict();


		Tagger tagger = new Tagger();
		tagger.loadModel("models/model.20120919");





		for (int i = 0; i < instances.numInstances(); i++) {
			double[] values = new double[result.numAttributes()];
			for (int n = 0; n < instances.numAttributes(); n++)
				values[n] = instances.instance(i).value(n);

			String content = instances.instance(i).stringValue(attrCont);
			content=content.toLowerCase();

			List<String> words =Twokenize.tokenizeRawTweetText(content);

			Map<String, Integer> cleanLexCounts = cleanLex
					.evaluatePolarityLexicon(words);
			values[result.attribute("LEX-CL-PW").index()] = cleanLexCounts
					.get("posCount");
			values[result.attribute("LEX-CL-NW").index()] = cleanLexCounts
					.get("negCount");


			Map<String,Double> ssScores=MyUtils.evaluateSentiStrength(sentiStrength, words);
			values[result.attribute("LEX-SS-PS").index()] = ssScores.get("posScore");
			values[result.attribute("LEX-SS-NS").index()] = ssScores.get("negScore");



			List<String> tagWords=new ArrayList<String>();

			List<TaggedToken> tagTokens=tagger.tokenizeAndTag(content.toLowerCase());
			for(TaggedToken tt:tagTokens){
				tagWords.add(tt.tag+"-"+tt.token);
			}


			Map<String,Double> swScores=swLex.getPosNegScores(tagWords);

			values[result.attribute("LEX-SW-PS").index()] = swScores.get("posScore");
			values[result.attribute("LEX-SW-NS").index()] = swScores.get("negScore");	

			Map<String,Double> edEMScores=edEMLex.evaluatePolarity(tagWords);

			values[result.attribute("LEX-EDEM-PS").index()] = edEMScores.get("posScore");
			values[result.attribute("LEX-EDEM-NS").index()] = edEMScores.get("negScore");			

			Map<String,Double> edFuzzScores=edFuzzLex.evaluatePolarity(tagWords);
			
			values[result.attribute("LEX-EDFU-PS").index()] = edFuzzScores.get("posScore");
			values[result.attribute("LEX-EDFU-NS").index()] = edFuzzScores.get("negScore");	


			Map<String,Double> edTh07Scores=edTh07Lex.evaluatePolarity(tagWords);

			values[result.attribute("LEX-EDT07-PS").index()] = edTh07Scores.get("posScore");
			values[result.attribute("LEX-EDT07-NS").index()] = edTh07Scores.get("negScore");	




			Map<String,Double> s140Scores=s140Lex.evaluatePolarity(tagWords);

			values[result.attribute("LEX-S140-PS").index()] = s140Scores.get("posScore");
			values[result.attribute("LEX-S140-NS").index()] = s140Scores.get("negScore");


			Instance inst = new SparseInstance(1, values);

			inst.setDataset(result);

			// copy possible strings, relational values...
			copyValues(inst, false, instances, result);

			result.add(inst);

		}

		return result;
	}

}
