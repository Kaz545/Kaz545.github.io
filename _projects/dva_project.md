---
layout: page
title: Hotel Reviews Dashboard
description: Analytics for Online Hotel Reviews.
img: assets/img/hotelreview.png
importance: 1
category: work
related_publications: false
---

The hotel industry frequently relies on conventional performance indicators such as Revenue Per Available Room (RevPAR) and Average Length of Stay (ALOS) to assess operational success. However, these quantitative metrics do not offer insights into the underlying reasons for fluctuations in performance. To address this gap, qualitative feedback from guests plays a critical role in identifying factors contributing to performance changes. Online reviews and customer comments serve as valuable sources of information that can aid hotels in enhancing their services and amenities. Nevertheless, systematically monitoring and analyzing this unstructured data poses significant challenges, particularly for large hotel chains operating across multiple locations.

### Objective

The primary objective of this project is to develop a system that streamlines the analysis of customer feedback, making it more accessible to hotel staff and management while facilitating data-driven decision-making to enhance customer satisfaction. The proposed solution will leverage natural language processing (NLP) techniques to extract key topics from guest reviews, such as identifying negative sentiment associated with cleanliness. The project’s main deliverable will be an interactive dashboard designed to enable analysts to visualize and interpret the data effectively. This dashboard will incorporate various visualization techniques, including time series charts and heat maps, to highlight patterns and trends in customer feedback. Additionally, the dashboard will be customizable, allowing hotel staff to filter data based on specific parameters, such as location, to meet their unique operational needs. A successfully implemented solution will provide hotels with deeper insights into their business performance, ultimately supporting strategic improvements in service quality and operational efficiency.


### Data

For this project we used this <a href="https://www.kaggle.com/datasets/jiashenliu/515k-hotel-reviews-data-in-europe">dataset</a> from Kaggle. 

### Methodology

Although topic segmentation and extraction are well-established techniques, tracking these findings over time remains uncommon due to the ad hoc nature of such analyses. Furthermore, the complexity and time-intensive nature of qualitative data analysis often lead hotels to prioritize their own internal data rather than conducting cross-chain comparisons. By automating and standardizing sentiment analysis and topic modeling into quantifiable metrics, this approach will facilitate comparative analysis across multiple hotel chains. This will enable hotels to benchmark their performance not only against their own historical data but also against industry competitors. The proposed solution will focus on natural language processing (NLP) modeling, data processing, and the development of interactive dashboards. However, the scope of this project does not include the creation of APIs for collecting online reviews.

#### Topic and Sentiment Segmentation

Initially, sentiment analysis was included within the project scope to determine the overall positive or negative sentiment of customer reviews. However, since guests were specifically prompted to provide both positive and negative feedback, their responses were recorded in separate columns. Consequently, the project shifted its focus toward running two distinct topic extraction models for positive and negative reviews.  

At the outset, the Latent Dirichlet Allocation (LDA) method was employed for topic extraction. However, preliminary tests revealed that this approach was computationally expensive and required extensive fine-tuning to generate clearly defined topics. As a result, alternative methodologies were explored. Through further research, the study identified *BERTopic*, a widely used NLP framework, as a more efficient and effective solution for topic modeling.

#### BerTopic

BERTopic offers extensive functionality and flexibility, including various tuning options, to meet the objectives of this project. It provides a topic modeling framework that integrates multiple independent models in a cascading manner. These models can be executed simultaneously or independently, allowing for fine-tuning based on specific requirements. As outlined in the <a href="https://maartengr.github.io/BERTopic/index.html">BERTopic</a>  documentation the primary steps involved in topic determination are as follows:  

1. **Embeddings:** Sentence embeddings are generated using a pre-trained language model. BERTopic supports multiple *sentence-BERT (sBERT)* models available on the Hugging Face platform, specifically within the *sentence-transformers* library. These embeddings can be precomputed and provided as input to the model, enabling fine-tuning and GPU selection, or they can be computed dynamically during model instantiation. The resulting vector space represents all sentences numerically.  

2. **Dimensionality Reduction:** Given the high-dimensional nature of embeddings (~10³ dimensions), clustering directly in this space can be computationally expensive and prone to noise. To address this, a dimensionality reduction technique is applied. While multiple approaches exist, *Uniform Manifold Approximation and Projection (UMAP)* is preferred due to its ability to preserve non-linear relationships within vector spaces while offering improved computational efficiency compared to other methods like t-SNE. Techniques such as *Principal Component Analysis (PCA)* are generally avoided, as they may lead to information loss.  

3. **Clustering:** Once the dimensionality of the dataset is reduced, clustering is performed to group similar reviews. The *Hierarchical Density-Based Spatial Clustering of Applications with Noise (HDBSCAN)* algorithm is utilized because it does not force all data points into clusters, allowing for a more flexible and representative topic structure than traditional clustering techniques.  

4. **Tokenizer:** Within each identified cluster, a classical bag-of-words approach is applied to process text data, where clusters are represented by the reviews assigned to them.  

5. **Weighting Scheme:** A class-based variation of Term Frequency-Inverse Document Frequency (*c-TF-IDF*) is employed, where each cluster is treated as a distinct class. This approach highlights the most significant words per cluster, ensuring that these words effectively represent the corresponding topic.  

6. **Fine-Tuning:** BERTopic provides several post-processing techniques to refine and improve topic representations. These techniques help reduce noise, merge similar topics, and enhance interpretability, leading to a more coherent and informative topic modeling framework.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/BerTopic_flow.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    BERTopic E2E framework.
</div>

The selected pre-trained *sentence-BERT (sBERT)* model for this study was **paraphrase-multilingual-MiniLM-L12-v2**. To optimize computational efficiency and allow for device selection, particularly GPU utilization, we precomputed the embeddings. The remaining models within the BERTopic pipeline were predefined and assigned during model instantiation, enabling explicit hyperparameter specification rather than relying on BERTopic’s default settings.  

In the **dimensionality reduction** step, *Principal Component Analysis (PCA)* was initially applied to improve the initialization of the *Uniform Manifold Approximation and Projection (UMAP)* model, thereby reducing computational overhead.  

For **clustering**, *Hierarchical Density-Based Spatial Clustering of Applications with Noise (HDBSCAN)* was employed. The *n_neighbors* parameter, which determines the degree of locality in topic representations, was adjusted based on computational constraints. A higher value results in more globally coherent topics but significantly increases training time.  

To extract topics, we used *CountVectorizer* with support for unigrams, bigrams, and trigrams, ensuring that extracted topics consisted of at least one word and up to three words.  

Finally, for **fine-tuning**, we incorporated *Maximal Marginal Relevance (MMR)* to enhance topic diversity and reduce redundancy by minimizing inter-topic word repetition. This approach helped ensure a more diverse and interpretable set of extracted topics.

#### Visualizations

For the reporting component, **Tableau** was selected as the visualization tool due to its user-friendly interface and extensive range of visualization options. The development process followed an **agile methodology**, involving multiple iterations of prototyping. The primary end user of the dashboard was identified as a **hotel manager**, ensuring that the design and functionality aligned with their needs.  

Throughout the iterative process, various features were reconsidered or omitted to maintain a practical and efficient solution. For instance, initial ideas such as incorporating **hotel distance as a filter** or **trend clustering visualization** were explored but ultimately excluded due to feasibility constraints. Striking a balance between feature richness and usability was a key consideration in delivering a functional and effective reporting tool. 
The end product resulted in the following two views:

##### Executive View

The view is intended for high-level stakeholders who want to see a broad level understanding of how the customer experience is trending across different hotels and geographies of the organization.

**Filtering**: By Hotel Chain, by Date, by Hotel Name, by Avg. Review Score, by negative/positive reviews, and by review topic.
**Visuals**:
- *Geographic Map*: Interactive map of Europe that will show avg. hotel review score by area.
- *Review Score Trend*: avg. score plotted over time.
- *Reviews segmented by Topic*: distribution of which are the “hottest” topics based on the reviews.
- *Review Comment Word Cloud*: portrays which are the most prevalent keywords customers are using when describing their hotel experience.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/exe_view.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Dashboard Executive View.
</div>

##### Hotel Comparison View

The view provides more granular analysis of the customer sentiment. The user is able to select a specific hotel, topic, and how it relates to the original review.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/dd_view.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Dashboard Deep Dive View.
</div>

### Experiments and Evaluation

Before implementing the model, positive and negative reviews per hotel were separated. Any hotels with less than 100 reviews were excluded to ensure the extracted topics are more objective. After cleaning up the data, stratified sampling was used to obtain one review per day per hotel. This was done to prevent hotels with more reviews from overshadowing those with fewer reviews.

During the data preprocessing stage, an interesting case of the word "nothing" having dual meanings was encountered. A happy customer could use the word "nothing" to mean that nothing is wrong, while an angry customer could use the word "nothing" to mean that nothing works. This experience highlights the challenges of working with text data. Rules were applied based on the review score to account for this type of case. 

Additionally, stop words were removed as they are commonly used and do not add much value to NLP model training. To evaluate the model, the framework called <a href="https://github.com/MIND-Lab/OCTIS">OCTIS</a>  was used. Through this framework, two evaluation metrics were used: **Topic Diversity Score** and **Coherence**.

As the name suggests, the first metric tells us how diverse the topics are. It calculates the distinctiveness of each topic based on the words and concepts each topic contains. A higher score reflects greater diversity in the output of the model. OCTIS uses Jensen-Shannon divergence to calculate the diversity score, which is a popular method in Topic modeling that measures similarity or dissimilarity between two distributions and compares the distributions of words of different topics. For the diversity score for the positive review model, it was *0.91*, while for the negative review model, it was *0.97*, suggesting that the topics generated by each model were pretty unique. In contrast, the negative review model did a better job.

On the other hand, Coherence measures how meaningful and coherent the topics are based on the words assigned to them. A higher number generally indicates that the words assigned to a certain topic are more semantically coherent, and the topic is more interpretable. There are many coherence metrics here, but the 'C_V' method was used, which has the highest correlation with human interpretation. For our models, the positive review model had a coherence of *0.71*, while the negative review model had a coherence of *0.73*, suggesting that the topics generated by the models are pretty meaningful and interpretable.

Some reviews (in both positive and negative review models) where not assigned to a topic given that they did not have enough probability of belonging to a specific topic. These were tagged with a -1 by the models. The reason was that they weren’t informative enough or were not represented by the extracted topic spectrum. This percentage of comments that are mapped to -1 is a good proxy to understand how practical is the solution. To reduce the number of reviews mapped to -1, we explored the number of topics as a hard parameter in the model.

Defining the number of topics is a balance between granularity and practicality. With very granular topic breaks the reporting can become more challenging having 100s of different topic groups. Larger sentences/comments require that additional topic granularity. Provided below is the distribution of comments for when the parameter was set at 30 topics. As one can observe there is a heavy skew to unmapped. We tried testing increasing the number of topics without any successful results. The -1 bucket was not reduced and we only ended up making the tail of the distribution more fragmented. The final solution was set to 30 topics to maintain practically.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/com_dist.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Comment Distribution.
</div>

For reporting purposes, having mappings called topic1, topic 2 in a dashboard does not provide an intuitive solution. A clean version was provided by creating reporting_topic describing the topic in plain English (e.g. good breakfast). This step was manual and required human interpretation of the topics.

### Conclusion

The text analytics solution presented in this study offers a customizable and scalable approach that can be adapted to various industries that handle comment reviews. However, it is important to note that the process of working with text analytics can be complex and requires human validation for better interpretability of results. As the study uncovered, one area of improvement for future research could be addressing the challenge of tying multiple topics into one comment, particularly for longer comments, which could lead to more accurate modeling of unmapped comments. With the advancement of Large Language Models like ChatGPT , there is potential for even more powerful and objective representations of
topic reviews. Overall, this study highlights the potential of text analytics in extracting meaningful insights from customer comments and opens avenues for further research in this area.

**<a href="https://public.tableau.com/app/profile/ricardo.goncalves6915/viz/cse6242_db/ExecutiveView?publish=yes">DASHBOARD LINK</a>** 

## Team Members
- *<a href="https://www.linkedin.com/in/juan-villegas-3a9242107/">JUAN CARLOS VILLEGAS</a>*
- *RICARDO GONCALVES*
- *GABRIEL PEREZ*
- *MEHLIAL KAZMI*