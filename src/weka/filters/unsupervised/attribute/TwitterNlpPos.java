package weka.filters.unsupervised.attribute;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import lexexpand.core.MyUtils;
import cmu.arktweetnlp.Tagger;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SparseInstance;
import weka.filters.SimpleBatchFilter;

public class TwitterNlpPos extends SimpleBatchFilter {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	@Override
	public String globalInfo() {
		return "A simple batch filter that adds attributes for all the "
				+ "Twitter-oriented POS tags of the TwitterNLP library.  ";
	}

	public boolean allowAccessToFullInputFormat() {
		return true;
	}

	@Override
	protected Instances determineOutputFormat(Instances inputFormat)
			throws Exception {

		ArrayList<Attribute> att = new ArrayList<Attribute>();

		// Adds all attributes of the inputformat
		for (int i = 0; i < inputFormat.numAttributes(); i++) {
			att.add(inputFormat.attribute(i));
		}

		att.add(new Attribute("POS-N")); // common noun
		att.add(new Attribute("POS-O")); // personal or WH, not possessive
		att.add(new Attribute("POS-S")); // nominal + possessive
		att.add(new Attribute("POS-^")); // proper noun
		att.add(new Attribute("POS-Z")); // proper noun + possessive
		att.add(new Attribute("POS-L")); // nominal + verbal
		att.add(new Attribute("POS-M")); // proper noun + verbal
		att.add(new Attribute("POS-V")); // verb or auxiliary
		att.add(new Attribute("POS-A")); // adjective
		att.add(new Attribute("POS-R")); // adverb
		att.add(new Attribute("POS-!")); // interjection
		att.add(new Attribute("POS-D")); // determiner
		att.add(new Attribute("POS-P")); // preposition or subordinating
											// conjunction
		att.add(new Attribute("POS-&")); // coordinating conjunction
		att.add(new Attribute("POS-T")); // verb particle
		att.add(new Attribute("POS-X")); // existential "there" or predeterminer
		att.add(new Attribute("POS-Y")); // X + verbal
		att.add(new Attribute("POS-#")); // hashtag
		att.add(new Attribute("POS-@")); // at-mention
		att.add(new Attribute("POS-~")); // Twitter discourse function word
		att.add(new Attribute("POS-U")); // URL or email address
		att.add(new Attribute("POS-E")); // emoticon
		att.add(new Attribute("POS-$")); // numeral
		att.add(new Attribute("POS-,")); // punctuation
		att.add(new Attribute("POS-G")); // other abbreviation, foreign word,
											// possessive ending, symbol, or
											// garbage
		att.add(new Attribute("POS-?")); // unsure

		Instances result = new Instances("Twitter Sentiment Analysis", att, 0);

		// set the class index
		result.setClassIndex(inputFormat.classIndex());

		return result;
	}

	@Override
	protected Instances process(Instances instances) throws Exception {
		Instances result = new Instances(determineOutputFormat(instances), 0);

		Tagger tagger = new Tagger();
		tagger.loadModel("models/model.20120919");

		// reference to the content of the tweet
		Attribute attrCont = instances.attribute("content");

		for (int i = 0; i < instances.numInstances(); i++) {
			double[] values = new double[result.numAttributes()];
			for (int n = 0; n < instances.numAttributes(); n++)
				values[n] = instances.instance(i).value(n);

			String content = instances.instance(i).stringValue(attrCont);
			List<String> words = MyUtils.cleanTokenize(content);
			List<String> posTags = MyUtils.getPOStags(words, tagger);

			// calculate frequencies of different POS tags
			Map<String, Integer> posFreqs = MyUtils.calculateTermFreq(posTags);

			// add POS values
			for (String posTag : posFreqs.keySet()) {
				int index = result.attribute("POS-"+posTag).index();
				values[index] = posFreqs.get(posTag);
			}

			Instance inst = new SparseInstance(1, values);
			result.add(inst);

		}
		return result;

	}

}
