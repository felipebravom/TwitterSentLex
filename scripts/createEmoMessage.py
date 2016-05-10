import csv

def process_file():
    """Process the datasets"""
    file_name="../testTweets/Jan9-2012-tweets-clean.txt" 


  #  header='@relation '+file_name+'\n\n@attribute content string\n@attribute emotion.label {joy,sadness,surprise,fear,anger,disgust}\n\n@data\n'
    
    header='@relation '+file_name+'\n\n@attribute content string\n@attribute emotion.label string \n\n@data\n'
        
    
    out=open(file_name+'.arff',"w")
    out.write(header)



       
    f=open(file_name, "rb")
    for line in f.readlines():
        parts=line.split("::")
        if len(parts)>2:
            
            message=line[20:-7].strip()            
            
            label=parts[len(parts)-1].strip()
            line='\"'+message+'\",\"'+label+'\"\n'
            out.write(line)
            
            
        else:            
            message=parts[0][20:].strip()
            label=parts[1].strip()
            line='\"'+message+'\",\"'+label+'\"\n'
            out.write(line)
    

    f.close()  
    out.close()    
        
       
      
#    rownum=0
#    for row in reader:
#        if rownum!=0:
#            word=row[18]
#            if word not in word_counts:
#                word_counts[word]=1
#            else:
#                word_counts[word]+=1
#            
#            
#        rownum+=1
#        
#    f.close()
#    
#    for tup in word_counts:
#        print tup, word_counts[tup]
        
        
  

    
def create_arff(file_name,word_counts):
    header='@relation '+file_name+'\n\n@attribute sgd.last numeric\n@attribute sgd.means numeric\n@attribute sgd.trunc.means numeric\n@attribute sgd.medians numeric\n'
    header+='@attribute sgd.sds numeric\n@attribute sgd.sign_changes numeric\n@attribute sgd.sign_changes_diff numeric\n@attribute sgd.iqr numeric\n'
    header+='@attribute so.last numeric\n@attribute so.means numeric\n@attribute so.trunc.means numeric\n@attribute so.medians numeric\n@attribute so.sds numeric\n'
    header+='@attribute so.sign_changes numeric\n@attribute so.sign_changes_diff numeric\n@attribute so.iqr numeric\n'
    header+='@attribute pos.label {adjective,adverb,at.mention,common.noun,cord.conj,determiner,emoticon,exist.there,exist.verbal,hashtag,interjection,nom.pos,nominal.verb,numeral,other,preposition,pronoun,proper.noun,proper.noun.poss,punct,twit.disc,url,verb,verb.part}\n'
    header+='@attribute polarity.label {negative,neutral,positive}\n@attribute word_name string\n\n@data\n'
    
    out=open(file_name+'.arff',"w")
    out.write(header)

        
    
    f=open(file_name, "rb")
    reader = csv.reader(f,delimiter='\t')      
      
    rownum=0
    for row in reader:
        if rownum!=0:
            word=row[18]
            if word in word_counts:
                if word_counts[word]==5:
                    line=''
                    for i in range(0,len(row)):
                        if(i<18):                        
                            line+=row[i]+','
                        if(i==18):
                            line+='\''+row[i]+'\'\n'
                    out.write(line)
            
        rownum+=1
        
    f.close() 
    out.close()


        
def update_dict(file_name,word_counts):
    f=open(file_name, "rb")
    reader = csv.reader(f,delimiter='\t')      
      
    rownum=0
    for row in reader:
        if rownum!=0:
            word=row[18]
            if word not in word_counts:
                word_counts[word]=1
            else:
                word_counts[word]+=1
           
            
        rownum+=1
        
    f.close()
    

    


if __name__ == '__main__':
    process_file()

                           