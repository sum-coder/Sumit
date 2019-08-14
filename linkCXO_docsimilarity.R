
##############################################################################################
###############Document Similarity Project####################################################
##############################################################################################

# Project Brief 
# Building a Clustering Model to classify similar Documents
# Step1 - Randomly collect a few sample of job description
# Step2 - Text mine the samples and finding kewords 
# Step3 - Input random CVs and check for document similarity using TF-IDF / Cosine-similarity



## Setting up the working directory..
setwd("D:\\NLP_ML\\corpus\\r-corpus\\pdfs")


## reading pdf document files into r-
pdf.docs <- list.files(path = "D:\\NLP_ML\\corpus\\r-corpus\\pdfs",pattern="pdf$")


## Using libraries
library(pdftools) # read the pdf files
library(tm)       # text mining 
library(quanteda) # text data cleaning
library(wordcloud)# visualizing the corpus

## reading the pdf files, converting into text and assigning the vector
## using lapply()

pdf.docs.text <- lapply(pdf.docs, pdf_text)
pdf.docs.text


## Forming a corpus of text extracted
corp <- Corpus(URISource(pdf.docs.text)) 

## Visualize the texts using wordcloud
wordcloud(corp,random.color = T, use.r.layout =T, min.freq = 3,max.words = 500,
          random.order = F, colors =c(1,2,3)) 


## Cleaning text 
## Creating a term document matrix
pdf.docs.TDM <- TermDocumentMatrix(corp,
                                   control = 
                                     list(removePunctuation = TRUE,
                                          stopwords = TRUE,
                                          tolower = TRUE,
                                          stemming = TRUE,
                                          removeNumbers = TRUE,
                                          bounds = list(global = c(1, Inf)))) #list of words with
                                          # frequency 1 and upto Inf - max.




## Lets check for frequent words, occurring n-times
findFreqTerms(pdf.docs.TDM, 5)


## Scale/normalize/rationalize the data...to avoid outlier effect!
## Converting TDM into a matrix before we scale/normalize the data..
pdf.docs.TDM.df <- as.data.frame(inspect(pdf.docs.TDM))
pdf.docs.TDM.df.scale <- scale(pdf.docs.TDM.df)


## Lets calculate the distance between each rows of vectors
## Euclidean distance - d(i,j) 
## will tell us which words occuring in the matrix is closer to group 
pdf.docs.TDM.df.scale.dist <- dist(pdf.docs.TDM.df.scale,method="euclidean")
pdf.docs.TDM.df.scale.dist


## Cluster dendrogram - cluster and frequency rep.
## Words appearing higher on the cluster are most frequent
## in term of cluster with other words in the node.
Clusdend <- hclust(pdf.docs.TDM.df.scale.dist, method="ward.D")
Clusdend.plot <-plot(Clusdend)


## Lets see how strong association does a word have..
## we check for keywords from a document to test the correlation
findAssocs(pdf.docs.TDM, 'analyst',.80)


############################################################################################################
#########################################Creating n-grams###################################################
############################################################################################################

## Tokenization
## Creating tokens from the text extract
## removing the numbers, punctuations, symbols..
## converting the text into character format
pdf.docs.text.tokens <- tokens(as.character(pdf.docs.text),what='word',
                               remove_numbers=T,remove_punct=T,
                               remove_symbols=T, remove_hyphen=T) 



# Lower casing the tokens
pdf.docs.text.tokens <- tokens_tolower(pdf.docs.text.tokens)

# Removing stopwords 
# quanteda has a built in stopword list in English
pdf.docs.text.tokens <- tokens_select(pdf.docs.text.tokens, stopwords(),
                                  selection = "remove")


# Stemming - collapsing similar/similar looking words into one word
pdf.docs.text.tokens <- tokens_wordstem(pdf.docs.text.tokens, language = "english")

## Tri-grams
## Dimension of the vectors increases above 1700 
pdf.docs.text.tokens.ng <- tokens_ngrams(pdf.docs.text.tokens, 3)
pdf.docs.text.tokens.ng

## Applying TF-IDF function to get the scores on vectors
## Creating document term frequency matrix
pdf.docs.text.tokens.ng.dfm <- dfm(pdf.docs.text.tokens.ng, tolower = FALSE)


# calculating relative term frequency
term.freq <- function(row){
  row/sum(row)
}
# calculating IDF
inv.doc.freq <- function(col){
  corpus.size <- length(col)
  doc.count <- length(which(col>0))
  log10(corpus.size/doc.count)
}

# clculate TF-IDf 
tf.idf <- function(tf, idf){
  tf*idf
}

# normalize all docs.via tf
pdf.docs.text.tokens.ng.dfm.tf <- apply(pdf.docs.text.tokens.ng.dfm,1,term.freq)
dim(pdf.docs.text.tokens.ng.dfm)


# Calculate IDf - 
pdf.docs.text.tokens.ng.dfm.idf <- apply(pdf.docs.text.tokens.ng.dfm,2,inv.doc.freq)
str(pdf.docs.text.tokens.ng.dfm.idf)


# tfidf 
pdf.docs.text.tokens.ng.tfidf <- apply(pdf.docs.text.tokens.ng.dfm.tf,2,tf.idf, idf=pdf.docs.text.tokens.ng.dfm.idf)

# MAKING TEXTS as column vectors
# transpose the matrix..
pdf.docs.text.tokens.ng.tfidf <- t(pdf.docs.text.tokens.ng.tfidf)
View(pdf.docs.text.tokens.ng.tfidf)

#### LSA - Latent Symantic Analysis..
### irlba function to detrmine the most inportant column vectors/helps reduce the vector dimen.
### ASKING THE FUNCTION TO GENERATE 50 MOST IMP.VECTORS
library(irlba)
pdf.docs.text.tokens.ng.tfidf.irlba <- irlba(pdf.docs.text.tokens.ng.tfidf, nv=2, 
                                             maxit = 50) 

## Cosine similarity...lsa()
## Using the lsa() function on the tri-gram matrix above, we can compare the document scores and 
# hence similarity between them.
## install.packages("lsa")
library(lsa) 


