
import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Random;

import lexexpand.core.MyUtils;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ArffSaver;
import weka.experiment.PairedStats;
import weka.experiment.PairedStatsCorrected;
import weka.filters.Filter;
import weka.filters.SimpleBatchFilter;
import weka.filters.unsupervised.attribute.TwitterNlpWordToVector;
import weka.filters.unsupervised.attribute.WordClustersVSM;
import weka.filters.unsupervised.attribute.WordDocRecCentRand;
import weka.filters.unsupervised.attribute.WordDocumentRecCent;
import weka.filters.unsupervised.attribute.WordMixClustersVSM;
import weka.filters.unsupervised.attribute.WordPMI;
import weka.filters.unsupervised.attribute.WordVSMCompact;
import weka.filters.unsupervised.attribute.WordVSM;
import weka.filters.unsupervised.attribute.WordVSMCompactFast;
import weka.filters.unsupervised.attribute.WordVSMMulti;


public class Test {
	

	
	
	static public void main(String args[]) throws Exception{
		
		
		PairedStats ps=new PairedStatsCorrected(0.05,0.111);
		double[] a={1.0,2.3,4.0};
		double[] b={1.0,2.3,4.0};
		ps.add(a,b);
		ps.calculateDerived();
		System.out.println(ps.toString());
		
		
		int numbTweets=100;
		int centSize=9;
		
		
		int[] nums=MyUtils.getRandomPermutation (100,new Random());
		System.out.println(Arrays.toString(nums));
		
		
//		BufferedReader stop=new BufferedReader(new FileReader("lexicons/stopwords.txt"));
//		HashSet<String> la=new HashSet<String>();
//		String line;
//		while((line=stop.readLine())!=null){
//			System.out.println(line);
//			la.add(line);
//		}
//		
//		System.out.println(la.contains("the"));
				
		
	
	//	String path="/Users/admin/workspace/SentimentDomain/arff_data/sandersShort.arff";
//		String path="/Users/admin/edinHead.arff";
//		
//		BufferedReader reader = new BufferedReader(new FileReader(path));
//		Instances train = new Instances(reader);
//		reader.close();
//		
//	
//		
//		long startTime=System.nanoTime();
//		
//		SimpleBatchFilter f=new WordDocRecCentRand();
//		f.setOptions(Utils.splitOptions("-N 0  -I 3  -L -K -J lexicons/metaLexEmo.csv -R -S -T resources/stopwords.txt -O "));
//
//		f.setInputFormat(train);
//
//
//		train=Filter.useFilter(train, f);
//		
//		
//		System.out.println(train.toSummaryString());
//		System.out.println(train.toString());
//		
//		System.out.println(System.nanoTime()-startTime);





//
//
//		ArffSaver saver = new ArffSaver();
//		saver.setInstances(train);
//		saver.setFile(new File("TESTING.arff"));
//		saver.writeBatch();

		
	}

}
