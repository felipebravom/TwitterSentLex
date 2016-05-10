

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;

import weka.clusterers.ClusterEvaluation;
import weka.clusterers.LdaCluster;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.Utils;

public class TestLDAClusterer {
	
	static public void main(String args[]) throws Exception{
	
		BufferedReader reader = new BufferedReader(
				new FileReader("edimEx.arff"));
		Instances data = new Instances(reader);
		reader.close();
		
		
		LdaCluster lda = new LdaCluster();
		lda.setOptions(Utils.splitOptions("-D 2 -N 8 -A 1 -B 0.01 -I 1000 -P 4"));	
		lda.buildClusterer(data);
		
				
		Attribute attrCont = data.attribute("text");
		
		for (int i = 0; i < data.numInstances(); i++) {
			
			
			
			double[] probs=lda.distributionForInstance(data.instance(i));
			int clust=lda.clusterInstance(data.instance(i));
			
			
			System.out.println(data.instance(i).stringValue(attrCont));
			System.out.println("Cluster:"+clust);
			System.out.println("Probs:");
			for(double j:probs){
				System.out.print(j+" , ");
			}
			System.out.println("");
			
			
			
			
			
		}
		
		
//	    ClusterEvaluation eval = new ClusterEvaluation();
//	    eval.setClusterer(lda);
//	    eval.evaluateClusterer(new Instances(data));
//	    System.out.println("# of clusters: " + eval.getNumClusters());
	    

		
		
		
	}

}
