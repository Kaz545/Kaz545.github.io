---
layout: page
title: ToxicTrek
description: Detecting Social Media toxicity using Word2vec and Ensemble learning.
img: assets/img/tt_page.jpeg
importance: 1
category: work
related_publications: false
---

Social media is an integral part of the contemporary world. Individuals use it to share information, to have conversations or to create content. As one can share, say or comment on anything so along with positive material there is a lot of negativity on social media as well. According to numerous studies negative content, comments and conversations can have an adverse effect on individuals. This is imperative for major social media firms to quickly identify this type of content and remove it so that social media is a safe space for everyone. Hence, here we will try to identify whether a certain comment in English Wikipedia talks data-set is toxic (negative contribution to conversation) or not.

Our Methodology include firstly to convert the raw comments data into numerical representations (embeddings) and then use these as inputs to classifiers such as Logistic Regression, KNN, Naïve Bayes and ensemble methods such as RandomForest and AdaBoost to classify that whether a certain comment is toxic or not.

To understand the data better we plotted the WordCloud for toxic and non-toxic comments and we can clearly see that in toxic comments there are a lot of expletive and indecent words while in non-toxic its the opposite. We can find these plots below.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/non_toxic_comments.png" title="Non Toxic Comments" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/toxic_comments.png" title="Toxic Comments" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Left Image is wordcloud for non-toxic comments while on right we have wordcloud for toxic comments.
</div>

Furthermore, we converted our raw data into embeddings using word2vec. A language model details about which can be found here. After converting the data into embeddings we plotted the data after apply PCA and we can see that toxic and non toxic comments are well separated in this space. Hence this output from the word2vec model can be used as an informative features to our classification models.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/pca_plot.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    PCA results.
</div>

Note that we used PCA just to visualize the data and passed the whole output of 300 dimensional embeddings from word2vec to our models. Results for the models can be found below and AdaBoost seemed to have the best performance across all the metrics. Since we had imbalanced dataset we also utilized SMOTE to cater for that and then ran our models on balanced dataset. Models performed relatively better on the balanced dataset.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/tt_results.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Results.
</div>



