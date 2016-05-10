import it.unimi.dsi.fastutil.doubles.AbstractDoubleList;
import it.unimi.dsi.fastutil.doubles.DoubleArrayList;
import it.unimi.dsi.fastutil.objects.Object2ObjectMap;
import it.unimi.dsi.fastutil.objects.Object2ObjectOpenHashMap;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Arrays;
import java.util.zip.GZIPInputStream;


public class TestGZ {
	
	static public void main(String args[]) throws FileNotFoundException, IOException{
		
		Object2ObjectMap<String, AbstractDoubleList> map=new Object2ObjectOpenHashMap<String, AbstractDoubleList>() ;
		
		GZIPInputStream  gzip = new GZIPInputStream(new FileInputStream("resources/glove.twitter.27B.200d.txt.gz"));
		
		
		BufferedReader br = new BufferedReader(new InputStreamReader(gzip));
		
		String line;
		int a=0;
		while((line=br.readLine())!=null && a<100){
			String[] words=line.split(" ");
			
			AbstractDoubleList wordVector=new DoubleArrayList();
			for(int i=1;i<words.length;i++){
				wordVector.add(Double.parseDouble(words[i]));
			}
			
			map.put(words[0], wordVector);
			
			a++;
		}
		
		br.close();
		
		
		for(String word:map.keySet()){
			System.out.println(word);
			AbstractDoubleList weights=map.get(word);
			
			System.out.println(weights.toString());
			
		}
		
		
	//	gzipInputStream.
		
	}

}
