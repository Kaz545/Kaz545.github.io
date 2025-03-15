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

In topic modeling, **diversity** measures the distinctiveness of topics based on the words and concepts they contain. A higher diversity score indicates a greater degree of uniqueness among the extracted topics. **OCTIS** utilizes **Jensen-Shannon divergence** to compute this metric, a widely used approach in topic modeling that assesses the similarity or dissimilarity between distributions by comparing the word distributions across topics. In our analysis, the **diversity score** for the **positive review model** was *0.91*, whereas for the **negative review model**, it was *0.97*. These results suggest that while both models produced distinct topic representations, the negative review model exhibited slightly greater topic uniqueness.  

Conversely, **coherence** evaluates the semantic interpretability of the topics by assessing how meaningful the words assigned to each topic are. A higher coherence score indicates that the topics are more interpretable. Among various coherence metrics, we employed the **C_V method**, which has demonstrated the highest correlation with human topic interpretation. The **coherence score** for the **positive review model** was *0.71*, while the **negative review model** achieved *0.73*, indicating that the generated topics were semantically meaningful and interpretable.  

In both models, some reviews were **not assigned to any topic**, as they lacked a sufficiently high probability of belonging to a specific category. These reviews were assigned a **label of -1**, as they were either **not informative enough** or **did not align well with the extracted topic spectrum**. The proportion of reviews categorized as -1 serves as a useful proxy for evaluating the model’s practical applicability. To minimize the number of unassigned reviews, we explored the **number of topics** as a key hyperparameter.  

Defining the **optimal number of topics** involves a trade-off between **granularity and interpretability**. Excessive topic granularity can make reporting more complex, resulting in hundreds of distinct topic groups, particularly when analyzing longer textual inputs. Below, we present the **distribution of comments** when the model was configured with **30 topics**. The results revealed a strong **skew toward unassigned reviews**, prompting further experimentation with an increased number of topics. However, increasing the number of topics did not successfully reduce the **-1 category**; instead, it further fragmented the topic distribution. As a result, the final model was configured to **30 topics** to ensure a balance between interpretability and practical usability.

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

The text analytics solution developed in this study provides a **customizable and scalable framework** that can be applied across various industries that rely on customer feedback analysis. However, it is essential to acknowledge that **text analytics remains a complex process** that requires **human validation** to enhance the interpretability of results.  

One key area for future research involves **improving the modeling of longer comments**, particularly those containing multiple topics. Addressing this challenge could lead to more **accurate classification** and **reduction of unmapped comments**. Additionally, with the rapid advancements in **Large Language Models (LLMs)**, such as **ChatGPT**, there is an opportunity to leverage more **sophisticated and objective representations** for topic modeling and review analysis.  

Overall, this study demonstrates the **potential of text analytics** in extracting valuable insights from customer feedback while identifying **opportunities for further research** to refine and expand its applicability.

**<a href="https://public.tableau.com/app/profile/ricardo.goncalves6915/viz/cse6242_db/ExecutiveView?publish=yes">DASHBOARD LINK</a>** 

## Team Members
- *<a href="https://www.linkedin.com/in/juan-villegas-3a9242107/">JUAN CARLOS VILLEGAS</a>*
- *RICARDO GONCALVES*
- *GABRIEL PEREZ*
- *MEHLIAL KAZMI*