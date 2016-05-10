package tests;

import java.io.BufferedReader;
import java.io.FileReader;

import weka.core.Instances;
import weka.core.Utils;
import weka.filters.Filter;
import weka.filters.SimpleBatchFilter;
import weka.filters.unsupervised.attribute.WordVSM;
import weka.filters.unsupervised.attribute.WordVSMCompact;
import weka.filters.unsupervised.attribute.WordVSMCompactFast;

public class FilterTest {

	// A simple Test to compare the performance of WordVSM, WordVSMCompact, and WordVSMCompactFast
	static public void main(String args[]) throws Exception{

		int listSize=53;
		int partSize=10;
		
		int j=0;
		
		for(int i=0;i<listSize;i++){
			
			j += 3;
			System.out.println(j);
			System.out.println("El valor de i"+i);
			System.out.println("El valor de i%partSize "+i%partSize);

			
		}

		
		
		



	}

}
