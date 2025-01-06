---
layout: page
title: SRM 
description: Semantic-Based Recommendation Model.
img: assets/img/srm_page.jpeg
importance: 1
category: work
related_publications: false
---

In the realm of e-commerce, the ability to provide personalized recommendations based on user reviews is a key to enhancing user experience. This project aims to develop a semantic-driven recommendation system by performing topic modeling on Amazon Fine Food Reviews. The traditional recommendation systems often overlook the rich semantic information available in user reviews. This project aims to leverage this information to not only provide more personalized recommendations but also understand why certain recommendations are made. The project combines Natural Language Processing (NLP) and recommendation systems, two significant areas in machine learning, to solve a real-world problem.

**Data**

Data used for this project has been taken from Kaggle from <a href="https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews/"> here. </a>

**Methodology**

*Topic Modelling*

As with most recommendation models the first step usually includes matrix factorization to create two low rank matrices, one for users and one for product from user - product interaction matrix. We had around 256,059 users and 74,258 products and their interaction matrix was of size $$ 1.9 \times 10^{10} $$ . it was very difficult for us to work with this matrix due to its size, it was computationally intensive. We utilized topic modeling to cluster our users and products so that we can reduce the size of this interaction matrix and efficiently work with it.
Since we were clustering users and products we firstly created a history of users and products and find the all the reviews against each. We did this so that we can correctly identify a user and a product in semantic space based on all the interactions that included them.
Using this history we performed topic modeling to cluster the user and products. Steps included to convert the textual data into an embedding space, then performing dimensionality reduction and clustering and lastly fine tuning and evaluating the topics extract.


<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/product_topics.png" title="Product Topics" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/user_topics.png" title="User Topics" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Left Image showcase main themes/topics for porducts while the one on the right does it for the users.
</div>

*Neural Collaborative Filtering*

Topic modeling has enabled us to capture semantic representations of products and users based on reviews, as well as to assess the similarity between them. The Neural Collaborative Filtering (NCF) model is crafted to understand the function that represents user-item interactions, such as similarity, which can then be applied to forecast the probability of a user engaging with an item, for instance, a user providing a rating for a product. This model processes the unique identifiers of a user and an item to generate a predictive score for their potential interaction.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/model_architecture_srm.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Model Architecture.
</div>

Below schematic summarises the methodology:

**End-to-End Flow**

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/e2e_srm.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    E2E Flow.
</div>



