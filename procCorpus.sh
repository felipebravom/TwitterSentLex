java -Xmx4g -cp sentlex.jar:lib/weka.jar:lib/ark-tweet-nlp-0.3.2.jar weka.filters.unsupervised.attribute.WordCentroid -M $3 -I 1 -L -S -i $1 -o $2
