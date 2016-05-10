# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import word2vec
import sys

class Word2VecToEmoArff:
    
    def __init__(self,n):
        self.word_dic={}
        self.n=n
    
    
    def create_lex(self,lex_file):       
        f=open(lex_file, "rb")
        for line in f.readlines():
            parts=line.strip().split('\t')
            if self.word_dic.has_key(parts[0]):
                emo_words=self.word_dic.get(parts[0])
                emo_words[parts[1]]=int(parts[2])
            else:
                emo_words={}
                emo_words[parts[1]]=int(parts[2])
                self.word_dic[parts[0]]=emo_words
        f.close()
        
        
    def build_word2vec_model(self,input_file,model_file):
         word2vec.word2vec(input_file, model_file, size=self.n, verbose=True, binary=1)
         word2vec.word2vec
           
   
        
        
    def load_word2vec_model(self,model_file):
        self.model = word2vec.load(model_file)  
        



    def create_arff(self,output_file):
        header='@relation \'Edinburgh Corpus: -C 10\' \n\n'
        header+='@attribute anger {0,1}\n@attribute anticipation {0,1}\n'   
        header+='@attribute disgust {0,1}\n@attribute fear {0,1}\n'
        header+='@attribute joy {0,1}\n@attribute negative {0,1}\n'
        header+='@attribute positive {0,1}\n@attribute sadness {0,1}\n'
        header+='@attribute surprise {0,1}\n@attribute trust {0,1}\n'    
    
        
        for i in range(0,n):
            header+='@attribute dim'+str(i)+' numeric\n'
            
        header+='@attribute word_name string\n\n@data\n'
            
        out=open(output_file,"w")
             
        out.write(header)
    
            
        for word_vec in self.model.vocab:
            line=str()
            word_clean=word_vec.replace('\'','')
            if self.word_dic.has_key(word_clean):
                emo_word=self.word_dic[word_clean]
                line+=str(emo_word['anger'])+','            
                line+=str(emo_word['anticipation'])+',' 
                line+=str(emo_word['disgust'])+',' 
                line+=str(emo_word['fear'])+',' 
                line+=str(emo_word['joy'])+',' 
                line+=str(emo_word['negative'])+','
                line+=str(emo_word['positive'])+',' 
                line+=str(emo_word['sadness'])+',' 
                line+=str(emo_word['surprise'])+',' 
                line+=str(emo_word['trust'])+','
            else:
                line+='?,?,?,?,?,?,?,?,?,?,'           
       
                 
            
            for value in self.model[word_vec]:
                line+=str(value)+','        
            
            line+="\'"+word_clean+'\'\n'
            out.write(line.encode('utf8') )
        
        
        out.close()
        
        
    def create_csv(self,output_file):
        out=open(output_file,"w")            
        for word_vec in self.model.vocab:
            line=str()
            for value in self.model[word_vec]:
                line+=str(value)+'\t'
            line+=word_vec.replace('\t','')+'\n'
            out.write(line.encode('utf8'))
        out.close()
                
            
            

if __name__ == '__main__':
    
    
    n=100
    lex_file="/Users/admin/workspace/TwitterSentLex/lexicons/NRC-emotion-lexicon-wordlevel-v0.92.txt"    
    
    print sys.argv
    print sys.argv[1],sys.argv[2]
    
#    input_corpus="/Users/admin/workspace/TwitterSentLex/edim_lab_short.txt" 
#    model_file="/Users/admin/workspace/word2vec/edim_lab_short.bin"
#    arff_file="/Users/admin/workspace/word2vec/edim_lab_WordsEmo.arff"
#    csv_file="/Users/admin/workspace/word2vec/edim_lab_WordsEmo.csv"
 
#    input_corpus="/Users/admin/workspace/TwitterSentLex/edimEx.txt" 
#    model_file="/Users/admin/workspace/word2vec/edimEx.bin"
#    arff_file="/Users/admin/workspace/word2vec/edim_lab_WordsEmo.arff"
#    csv_file="/Users/admin/workspace/word2vec/edim_lab_word2Vec.csv"
      
    model_file=sys.argv[1]
    csv_file=sys.argv[2]
    
    wtarff=Word2VecToEmoArff(n)
#    wtarff.create_lex(lex_file)
#    wtarff.build_word2vec_model(input_corpus,model_file)
    wtarff.load_word2vec_model(model_file)
#    wtarff.create_arff(arff_file)
    wtarff.create_csv(csv_file)
    


