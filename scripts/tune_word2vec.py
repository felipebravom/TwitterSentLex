# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 15:27:17 2015

@author: admin
"""

import word2vec
import sys

def create_arff(model_file,output_file,n,word_dic):    
    model = word2vec.load(model_file)    
    
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
    
        
    for word_vec in model.vocab:
        line=str()
        word_clean=word_vec.replace('\'','')
        if word_dic.has_key(word_clean):
            emo_word=word_dic[word_clean]
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
            
            for value in model[word_vec]:
                line+=str(value)+','        
        
            line+="\'"+word_clean+'\'\n'
            out.write(line.encode('utf8'))   
    
    out.close()
        


if __name__ == '__main__':
    word_dic={}
    lex_file="/Users/admin/workspace/TwitterSentLex/lexicons/NRC-emotion-lexicon-wordlevel-v0.92.txt"    
    
             
    f=open(lex_file, "rb")
    for line in f.readlines():
        parts=line.strip().split('\t')
        if word_dic.has_key(parts[0]):
            emo_words=word_dic.get(parts[0])
            emo_words[parts[1]]=int(parts[2])
        else:
            emo_words={}
            emo_words[parts[1]]=int(parts[2])
            word_dic[parts[0]]=emo_words
    f.close()    
    
    
    
    input_file="/Users/admin/workspace/TwitterSentLex/tuning_tweets.txt"
    n_ranges=range(100,1001,100)
    w_ranges=range(1,11,1)
    

    for n in n_ranges:
        for w in w_ranges:
            model_name="model_n"+str(n)+"_w"+str(w)
            word2vec.word2vec(input_file, model_name+".bin", size=n, verbose=True, binary=1,window=w)
            create_arff(model_name+".bin",model_name+".arff",n,word_dic)
    
#    n=100
#    w=5
#
#    input_file    
#    
#    