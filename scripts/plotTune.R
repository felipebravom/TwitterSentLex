
tune.para<-read.table(file="Dropbox/THESIS/TAC/tuneResults.csv",header=T,sep='\t')

v <- ggplot(tune.para, aes(n, w, z = F1micro))
v + geom_tile(aes(fill = F1micro))+ scale_fill_gradient(low="black", high="white")+ labs(x="Embedding dimensionality",y ="Window size", title = "F1 micro avg")
v+theme(axis.text=element_text(size=12),
        axis.title=element_text(size=14,face="bold"))


asa.ex<-read.table("Dropbox/THESIS/ECAI2016/tables/SemEvalExHeatMap.csv",header=T,sep='\t')
asa.ex$a<-as.factor(asa.ex$a)
asa.ex$Instances<-as.factor(asa.ex$Instances)
v <- ggplot(asa.ex, aes(a, Instances, z = F1))
v + geom_tile(aes(fill = F1))+ scale_fill_gradient(low="white", high="black", limits=c(0.68, 0.77) )+ labs(x="a",y ="Instances (thousands)", title = "F1 values for ASA (m=T)")

v <- ggplot(asa.ex, aes(a, Instances, z = AUC))
v + geom_tile(aes(fill = AUC))+ scale_fill_gradient(low="white", high="black", limits=c(0.76, 0.86) )+ labs(x="a",y ="Instances (thousands)", title = "AUC values for ASA (m=T)")



asa.non.ex<-read.table("Dropbox/THESIS/ECAI2016/tables/SemEvalNonExHeatMap.csv",header=T,sep='\t')
asa.non.ex$a<-as.factor(asa.non.ex$a)
asa.non.ex$Instances<-as.factor(asa.non.ex$Instances)
v <- ggplot(asa.non.ex, aes(a, Instances, z = F1))
v + geom_tile(aes(fill = F1))+ scale_fill_gradient(low="white", high="black", limits=c(0.68, 0.77) )+ labs(x="a",y ="Instances (thousands)", title = "F1 values for ASA (m=F)")

v <- ggplot(asa.non.ex, aes(a, Instances, z = AUC))
v + geom_tile(aes(fill = AUC))+ scale_fill_gradient(low="white", high="black", limits=c(0.76, 0.86) )+ labs(x="a",y ="Instances (thousands)", title = "AUC values for ASA (m=F)")



