import it.unimi.dsi.fastutil.objects.ObjectOpenHashSet;
import it.unimi.dsi.fastutil.objects.ObjectSet;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import cmu.arktweetnlp.Twokenize;


public class ValueTest {

	protected HashMap<Integer,String> mymaper;
	public String word;

	public ValueTest(HashMap<Integer,String> maper,String word){
		this.mymaper=maper;
		this.word=word;
	}

	public void modify(){
		this.word += "change";
		for(int val:this.mymaper.keySet()){
			this.mymaper.put(val,"Cambie"+val);
		}
	}



	static public List<String> clustList(List<String> tokens, Map<String,String> dict){
		List<String> clusters=new ArrayList<String>();
		for(String token:tokens){
			if(dict.containsKey(token)){
				clusters.add(dict.get(token));
			}

		}	
		return clusters;
	}



	static public void main(String args[]) throws IOException{

		Map<String,String> dict=new HashMap<String,String>();
		BufferedReader bf = new BufferedReader(new FileReader(
				"resources/50mpaths2.txt"));
		String line;
		while ((line = bf.readLine()) != null) {
			String pair[] = line.split("\t");
			dict.put(pair[1], pair[0]);

		}
		bf.close();

		List<String> tokens=Twokenize.tokenizeRawTweetText("fuck fuuuck love loove you you you I am happy");
		List<String> clusters=clustList(tokens,dict);
		for(String w:clusters){
			System.out.print(w+" ");
		}


		System.out.println(dict.get("love")+" "+dict.get("loove") );
		
		
		
		System.out.println("LALALALLQA");
		ObjectSet<String> termFreq = new ObjectOpenHashSet<String>();
		termFreq.addAll(tokens);
		for(String la:termFreq){
			System.out.println(la);
		}
		
		
		System.out.println(Math.log(99));
		

		//		HashMap<Integer,String> maper=new HashMap<Integer,String>();
		//
		//		for(int i=0;i<100;i++){
		//			maper.put(i, "original "+i);
		//		}
		//
		//		for(int val:maper.keySet()){
		//			System.out.println(val+" "+maper.get(val));
		//		}
		//		
		//		String myword="no change";
		//		
		//		ValueTest a=new ValueTest(maper,myword);
		//		
		//		a.modify();
		//
		//		for(int val:maper.keySet()){
		//			System.out.println(val+" "+maper.get(val));
		//		}
		//		
		//		System.out.println(myword);
		//		System.out.println(a.word);





	}






}


