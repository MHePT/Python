import random
import string
import numpy as np

from re import split
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Boolean search
def boolean_search(query, documents):
    vectorizer = CountVectorizer(binary=True)
    docs_vec = vectorizer.fit_transform(documents).toarray()
    query_vec = vectorizer.transform([query]).toarray()
    return docs_vec, query_vec

# Cosine similarity
def cosine_sim(query, documents):
    docs_vec, query_vec = boolean_search(query, documents)
    return cosine_similarity(query_vec, docs_vec)

# Query likelihood model
def query_likelihood(query, documents):
    # Tokenize the documents
    docs_tokens = [doc.split() for doc in documents]
    
    # Calculate the length of each document
    docs_lengths = [len(doc) for doc in docs_tokens]
    
    # Calculate the length of the document collection
    collection_length = sum(docs_lengths)
    
    # Count the occurrences of each word in each document
    docs_word_counts = [Counter(doc) for doc in docs_tokens]
    
    # Count the occurrences of each word in the document collection
    collection_word_counts = Counter(word for doc in docs_tokens for word in doc)
    
    # Tokenize the query
    query_tokens = query.split()
    
    # Calculate the likelihood of the query for each document
    likelihoods = []
    for doc_word_counts, doc_length in zip(docs_word_counts, docs_lengths):
        likelihood = 1
        for word in query_tokens:
            # Calculate the maximum likelihood estimate of the word
            mle = doc_word_counts[word] / doc_length if word in doc_word_counts else 0
            
            # Calculate the maximum likelihood estimate of the word in the collection
            mle_collection = collection_word_counts[word] / collection_length if word in collection_word_counts else 0
            
            # Calculate the smoothed probability of the word
            smoothed_prob = 0.5 * mle + 0.5 * mle_collection
            
            # Multiply the likelihood by the smoothed probability of the word
            likelihood *= smoothed_prob
        
        # Add the log likelihood to the list of likelihoods
        likelihoods.append(np.log(likelihood) if likelihood > 0 else float('-inf'))
    
    return likelihoods


# Evaluation metrics
def precision_recall_fmeasure_rankpower(relevant_docs, retrieved_docs, n):
    
    if len(relevant_docs) > 0 and len(retrieved_docs) > 0:
        # Precision: (relevant docs ∩ retrieved docs) / retrieved docs
        precision = len(relevant_docs) / len(retrieved_docs)

        # Recall: (relevant docs ∩ retrieved docs) / relevant docs
        recall = len(set(relevant_docs) & set(retrieved_docs)) / len(relevant_docs)

        # F-measure: 2 * (precision * recall) / (precision + recall)
        fmeasure = 2 * (precision * recall) / (precision + recall)

        # Rank Power: n / relevant docs ^ 2
        sum = 0
        for i in n:
            sum += i
        rankpower = sum / len(relevant_docs) ** 2

        return precision, recall, fmeasure, rankpower
    return 0


documents = ["Science may set limits to knowledge but shouldn’t limits to imagination"]
documents.append("All of science is nothing more than the refinement of everyday thinking")
documents.append("Science and religion are not at odds science is simply to young to understand")
documents.append("There is no contradiction between true religion and science")


    

query = input("Enter your Query: ")
print("\nCosine Similarity:", cosine_sim(query, documents)[0])
print("Query Likelihood:", query_likelihood(query, documents) ,end="\n\n")

retrieved=[]
relevant_docs = []
n = []
for i in range(4):
    if cosine_sim(query, documents)[0][i] > 0:
        print("Document ",i," : \"",documents[i],'"\n\n')
        retrieved.append(documents[i])
        n.append(i)
        
print("Which Document is relevant? (Enter other Number than displayed to exit)")

for _ in n:
    x = int(input(str(n)+"? "))
    if x in n: 
        relevant_docs.append(documents[x])
    else:
        break
    

precision, recall, fmeasure, rankpower = precision_recall_fmeasure_rankpower(relevant_docs, retrieved, n)
print("\nEvaluation Metrics:","Precision = " + str(precision),"Recall = " + str(recall) , "Fmeasure = " + str(fmeasure) , "RankPower = " + str(rankpower) , sep="\n" )
