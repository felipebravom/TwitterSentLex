options( java.parameters = "-Xmx4g" )
library( "RWeka")
library("ggplot2")
library("reshape2")
library("xtable")

sign_change<-function(time.series){
  n<-length(time.series)
  changes<-sign(time.series[2:n]*time.series[1:(n-1)])
  count<-0
  for(i in 1:(n-1)){
    if(changes[i]==-1){
      count<-count+1
    }
  }
  return(count/n)
}

sign_change_diff<-function(a){
  time.series<-diff(a)
  n<-length(time.series)
  changes<-sign(time.series[2:n]*time.series[1:(n-1)])
  count<-0
  for(i in 1:(n-1)){
    if(changes[i]==-1){
      count<-count+1
    }
  }
  return(count/n)
}


setwd("TwitterStreamExp/SOSGDseriesData/")

s140.sgd.path<-"s140SGDseries.arff"
s140.so.path<-"s140SOseries.arff"


data.s140.sgd<-read.arff(s140.sgd.path)
data.s140.so<-read.arff(s140.so.path)

data.s140.so<-cbind(data.s140.so,data.s140.sgd$"SGD-bias")



words<-names(data.s140.sgd)

# Calculate the mean and the sd of each word
sgd.last<-as.numeric(data.s140.sgd[dim(data.s140.sgd)[1],])
sgd.means<-sapply(data.s140.sgd,mean)
sgd.trunc.means<-sapply(data.s140.sgd,mean,trim=0.1)
sgd.medians<-sapply(data.s140.sgd,median)

sgd.sds<-sapply(data.s140.sgd,sd)
sgd.sign_changes<-sapply(data.s140.sgd,sign_change)
sgd.sign_changes_diff<-sapply(data.s140.sgd,sign_change_diff)
sgd.iqr<-sapply(data.s140.sgd,IQR)


so.last<-as.numeric(data.s140.so[dim(data.s140.so)[1],])
so.means<-sapply(data.s140.so,mean)
so.trunc.means<-sapply(data.s140.so,mean,trim=0.1)
so.medians<-sapply(data.s140.so,median)

so.sds<-sapply(data.s140.so,sd)
so.sign_changes<-sapply(data.s140.so,sign_change)
so.sign_changes_diff<-sapply(data.s140.so,sign_change_diff)
so.iqr<-sapply(data.s140.so,IQR)


# Get the index of positive and negative Words
poswords<-grepl(pattern="positive",words)
negwords<-grepl(pattern="negative",words)
neuwords<-grepl(pattern="neutral",words)
nonneuwords<-poswords|negwords


polarity.label<-(as.factor(ifelse(
  poswords,"positive",
  ifelse(negwords,"negative",
         ifelse(neuwords,"neutral",NA)))))



neu.label<-(as.factor(ifelse(
  neuwords,"neutral",
  ifelse(nonneuwords,"non_neutral",NA))))


pos.neg.label<-(as.factor(ifelse(
  poswords,"positive",
  ifelse(negwords,"negative",NA))))

# Get the index of group of words
common.nouns<-grepl(pattern="N-",words)
pronouns<-grepl(pattern="O-",words)
nom.pos<-grepl(pattern="S-",words)
proper.nouns<-grepl(pattern="\\^-",words)
proper.nouns.poss<-grepl(pattern="Z-",words)
nominal.verb<-grepl(pattern="L-",words)
proper.noun.verb<-grepl(pattern="M-",words)
verb<-grepl(pattern="V-",words)

adjective<-grepl(pattern="A-",words)
adverb<-grepl(pattern="R-",words)
interjection<-grepl(pattern="\\!-",words)
determiner<-grepl(pattern="D-",words)
# Remove SGD-bias from the vector
determiner[length(words)]<-FALSE

preposition<-grepl(pattern="P-",words)
cord.conj<-grepl(pattern="\\&-",words)
verb.part<-grepl(pattern="T-",words)
exist.there<-grepl(pattern="X-",words)
exist.verbal<-grepl(pattern="Y-",words)
hashtag<-grepl(pattern="#-",words)
at.mention<-grepl(pattern="@-",words)
twit.disc<-grepl(pattern="~-",words)
url<-grepl(pattern="U-",words)
emoticon<-grepl(pattern="E-",words)
# remove positive words
emoticon<-(emoticon& !poswords) 
numeral<-grepl(pattern="\\$-",words)
punct<-grepl(pattern=",-",words)
other<-grepl(pattern="G-",words)
unsure<-grepl(pattern="\\?-",words)


pos.label<-(as.factor(ifelse(common.nouns,"common.nouns",
                             ifelse(pronouns,"pronouns",
                                    ifelse(nom.pos,"nom.pos",
                                           ifelse(proper.nouns,"proper.nouns",
                                                  ifelse(proper.nouns.poss,"proper.nouns.poss",                                    
                                                         ifelse(nominal.verb,"nominal.verb",
                                                                ifelse(proper.noun.verb,"proper.noun.verb",                                    
                                                                       ifelse(verb,"verb",     
                                                                              ifelse(adjective,"adjective",
                                                                                     ifelse(adverb,"adverb",
                                                                                            ifelse(interjection,"interjection",
                                                                                                   ifelse(determiner,"determiner",                                                                                             
                                                                                                          ifelse(preposition,"preposition",
                                                                                                                 ifelse(cord.conj,"cord.conj",
                                                                                                                        ifelse(verb.part,"verb.part",
                                                                                                                               ifelse(exist.there,"exist.there", 
                                                                                                                                      ifelse(exist.verbal,"exist.verbal",       
                                                                                                                                             ifelse(hashtag,"hashtag",
                                                                                                                                                    ifelse(at.mention,"at.mention",
                                                                                                                                                           ifelse(twit.disc,"twit.disc",
                                                                                                                                                                  ifelse(url,"url",    
                                                                                                                                                                         ifelse(emoticon,"emoticon",    
                                                                                                                                                                                ifelse(numeral,"numeral",   
                                                                                                                                                                                       ifelse(punct,"punct",    
                                                                                                                                                                                              ifelse(other,"other", 
                                                                                                                                                                                                     "unsure" )))))))))))))))))))))))))))

polarity.frame<-data.frame(
  sgd.last=as.vector(sgd.last),
  sgd.means=as.vector(sgd.means),
  sgd.trunc.means=as.vector(sgd.trunc.means), 
  sgd.medians=as.vector(sgd.medians),
  sgd.sds=as.vector(sgd.sds),
  sgd.sign_changes=as.vector(sgd.sign_changes),
  sgd.sign_changes_diff=as.vector(sgd.sign_changes_diff), 
  sgd.iqr=as.vector(sgd.iqr),
  so.last=as.vector(so.last),
  so.means=as.vector(so.means),
  so.trunc.means=as.vector(so.trunc.means), 
  so.medians=as.vector(so.medians),
  so.sds=as.vector(so.sds),
  so.sign_changes=as.vector(so.sign_changes),
  so.sign_changes_diff=as.vector(so.sign_changes_diff), 
  so.iqr=as.vector(so.iqr),                
  pos.label,polarity.label)
row.names(polarity.frame)<-words

target.frame<-polarity.frame[is.na(polarity.label),]
write.arff(target.frame,file="s140PolTarget.arff")

write.table(target.frame,"s140ExpWords.csv",quote=F,sep="\t")

polarity.frame<-polarity.frame[!is.na(polarity.label),]

write.arff(polarity.frame,file="s140Polarity.arff")




qplot(polarity.frame$so.last,polarity.frame$sgd.means,
      col=polarity.frame$polarity.label)

ggplot(polarity.frame, aes(x=sgd.means, y=so.means)) + geom_point(size=1.5,aes(color=polarity.label)) +
  xlab("sgd.posneg.mean") +  ylab("so.posneg.mean") 
ggsave(file="../../Dropbox/THESIS/lexExpand/SGDSO.pdf",width=8,height=6)

qplot(polarity.frame$sgd.means,polarity.frame$sgd.means.neutral,
      col=polarity.frame$polarity.label)




word.sample<-polarity.frame[c("positive-A-lovely","neutral-A-yellow","negative-A-upset","negative-V-worried"), ]

lovely<-data.s140.sgd$"positive-A-lovely"
yellow<-data.s140.sgd$"neutral-A-yellow" 
upset<-data.s140.sgd$"negative-A-upset"
worried<-data.s140.sgd$"negative-V-worried"  


sgd=c(lovely, yellow, upset, worried)

word.names.vis<-c(rep("lovely",length(lovely)), rep("yellow",length(lovely)),
                  rep("upset",length(upset)), rep("worried",length(lovely))
)
word.names.vis<-as.factor(word.names.vis)
word.frame<-data.frame(sgd=sgd, period= rep(1:(length(lovely)),4), word=word.names.vis)

ggplot(word.frame,aes(x=period,y=sgd,colour=word,group=word)) + geom_line() + xlab("Time window") +
  ylab("SGD score") + ggtitle("SGD time-series")
ggsave(file="../../Dropbox/THESIS/lexExpand/SGDseries.pdf",width=8,height=4)


lovely<-data.s140.so$"positive-A-lovely"
yellow<-data.s140.so$"neutral-A-yellow" 
upset<-data.s140.so$"negative-A-upset"
worried<-data.s140.so$"negative-V-worried"  

so=c(lovely, yellow, upset, worried)

word.names.vis<-c(rep("lovely",length(lovely)), rep("yellow",length(lovely)),
                  rep("upset",length(upset)), rep("worried",length(lovely))
)
word.names.vis<-as.factor(word.names.vis)
word.frame<-data.frame(so=so, period= rep(1:(length(lovely)),4), word=word.names.vis)

ggplot(word.frame,aes(x=period,y=so,colour=word,group=word)) + geom_line() +  xlab("Time window") +
  ylab("SO score") +  ggtitle("SO time-series")
ggsave(file="../../Dropbox/THESIS/lexExpand/SOseries.pdf",width=8,height=4)

library("ggplot2")
library("reshape2")

