require("wordcloud")


lex<-read.csv("workspace/TwitterSentLex/lexicons/uni-bwn-pos-dp-w2v-BCC-Lex.csv",header=T,sep="\t")
lex$sadness

ang_words<-lex[lex$anger>0.4,c("word","anger")]
wordcloud(ang_words$word,ang_words$anger,max.words=500, random.order=T)
#save in 9*7

ant_words<-lex[lex$anticipation>0.4,c("word","anticipation")]
wordcloud(ant_words$word,ant_words$anticipation,max.words=500,  random.order=T)


disg_words<-lex[lex$disgust>0.4,c("word","disgust")]
wordcloud(disg_words$word,disg_words$disgust,max.words=500,  random.order=T)


fear_words<-lex[lex$fear>0.4,c("word","fear")]
wordcloud(fear_words$word,fear_words$fear,max.words=500,  random.order=T)


joy_words<-lex[lex$joy>0.4,c("word","joy")]
wordcloud(joy_words$word,joy_words$joy,max.words=500,  random.order=T)


sadness_words<-lex[lex$sadness>0.4,c("word","sadness")]
wordcloud(sadness_words$word,sadness_words$sadness,max.words=500,  random.order=T)


surprise_words<-lex[lex$surprise>0.4,c("word","surprise")]
wordcloud(surprise_words$word,surprise_words$surprise,max.words=500,  random.order=T)




polwords<-s140Lex$label!="neutral"
EdLexOW<-s140Lex[polwords,]

EdLexOW<-s140Lex

emoPOS<-EdLexOW$POS=="adjective"|EdLexOW$POS=="common.nouns"|EdLexOW$POS=="verb"|
  EdLexOW$POS=="emoticon"|EdLexOW$POS=="adverb"|EdLexOW$POS=="interjection"|EdLexOW$POS=="hashtag"
EdLexOW<-EdLexOW[emoPOS,]

EdLexOW$positive<-ifelse(EdLexOW$positive==0,0.01,EdLexOW$positive)
EdLexOW$negative<-ifelse(EdLexOW$negative==0,0.01,EdLexOW$negative)

so<-log(EdLexOW$positive,2)-log(EdLexOW$negative,2)
#so<-ifelse(is.nan(so)|is.infinite(so),0,so)

clean.words<-sub("positive-|negative-|neutral-","",EdLexOW$word)
clean.words<-substring(clean.words,3)

d <- data.frame(word = clean.words,freq=so)



wordcloud(d$word,d$freq,min.freq=1.0,max.words=500, ordered.colors=T,
          random.order=T, random.color=F, colors="red")

wordcloud(d$word,-d$freq,min.freq=1.0,max.words=500, ordered.colors=T,
          random.order=T, random.color=T,)



so<-log(predictions$positive,2)-log(predictions$negative,2)
so<-ifelse(is.nan(so)|is.infinite(so),0,so)
names(so)<-clean.words
so<-sort(so, decreasing=T)


