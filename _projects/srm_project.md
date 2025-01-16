---
layout: page
title: SRM 
description: Semantic-Based Recommendation Model.
img: assets/img/srm_page.jpeg
importance: 1
category: work
related_publications: false
---
**NOTE :** Colloborated with <a href="https://www.linkedin.com/in/gshilpa3/"> Shilpa G </a> on this project. 

In the realm of e-commerce, the ability to provide personalized recommendations based on user reviews is a key to enhancing user experience. This project aims to develop a semantic-driven recommendation system by performing topic modeling on Amazon Fine Food Reviews. The traditional recommendation systems often overlook the rich semantic information available in user reviews. This project aims to leverage this information to not only provide more personalized recommendations but also understand why certain recommendations are made. The project combines Natural Language Processing (NLP) and recommendation systems, two significant areas in machine learning, to solve a real-world problem.

### Data

We used amazon fine food reviews data-set available on Kaggle <a href="https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews/"> here. </a>. It had total of 568,454 reviews from around
256,059 users on 74,258 products. It included unique iden-
tifier for users, products and ratings and plain text review.

### Methodology 

### Data Preprocessing

Since our approach included using the textual data to extract semantic features that would be essentially inputs to our deep learning framework for recommendations. There were multiple data pre-processing steps that are as follows:
 - We cleaned our reviews data by removing all non alphabetic characters and stop words as they would not add any value to the semantic representation of the review.
 - Since we were doing topic modeling hence we made sure that sentences with just one words are nouns this ensured more meaningful topics.
 - We performed stemming for text normalization and reduced all the words to their root as this would help us consolidate the words with same base and enhance the topic modeling.
After cleaning our text data we analyzed the it using word cloud to understand it better and to ensure that we have the data in the format we want it to be.

#### Topic Modelling 

As with most recommendation models the first step usually includes matrix factorization to create two low rank matrices, one for users and one for product from user - product interaction matrix. We had around 256,059 users and 74,258 products and their interaction matrix was of size $$ 1.9 \times 10^{10} $$ . it was very difficult for us to work with this matrix due to its size, it was computationally intensive. We utilized topic modeling to cluster our users and products so that we can reduce the size of this interaction matrix and efficiently work with it.
Since we were clustering users and products we firstly created a history of users and products and find the all the reviews against each. We did this so that we can correctly identify a user and a product in semantic space based on all the interactions that included them.
Using this history we performed topic modeling to cluster the user and products. Steps included to convert the textual data into an embedding space, then performing dimensionality reduction and clustering and lastly fine tuning and evaluating the topics extract. Below schematic summarises the methodology:

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/e2e_srm.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    E2E Flow.
</div>

#### Topic Extraction 

After extracting the topics we need some representation of them we can do this using count vectorizer that would essentially find the count of every token in a cluster and then using this bag of words representation we used class/topic based TF-IDF to find the important terms in a particular
topic. We generalize TF-IDF procedure to clusters of documents. First, we treat all documents in a cluster as a single document by simply concatenating the documents. Then, TF-IDF is adjusted to account for this representation by translating documents to clusters:

$$  W_{t,c} = t_{f,c}.\log(1 + \frac{A}{tf_{t}}) $$

Where the term frequency models the frequency of term $$t$$ in a class $$c$$ or in this instance. Here, the class $$c$$ is the collection of documents concatenated into a single document for each cluster. Then, the inverse document frequency is replaced by the inverse class frequency to measure how much information a term provides to a class. It is calculated by taking the logarithm of the average number of words per class $$A$$ divided by the frequency of term $$t$$ across all classes. To output only positive values, we add one to the division within the logarithm.
We can find the main topics for each user and product in below schematics. 

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

#### Neural Collaborative Filtering

Topic modeling has enabled us to capture semantic representations of products and users based on reviews, as well as to assess the similarity between them. The Neural Collaborative Filtering (NCF) model is crafted to understand the function that represents user-item interactions, such as similarity, which can then be applied to forecast the probability of a user engaging with an item, for instance, a user providing a rating for a product. This model processes the unique identifiers of a user and an item to generate a predictive score for their potential interaction.

The proposed NCF model is designed to capture the latent interactions between users and items. The model accepts as input the identifiers of a user and an item, and outputs a prediction representing the likelihood of interaction between the user and the item. The architecture of the model is as follows:

- Embedding Layers: The user and item identifiers are independently passed through distinct embedding layers. These layers transform the identifiers into dense vector representations, effectively capturing the latent factors associated with each user and item. The embedding layers are initialized with pre-existing embeddings and are held constant during training.
- Element-wise Multiplication: The resulting user and item embeddings are subjected to an element-wise multiplication operation, yielding an interaction vector. This operation can be mathematically represented as:
$$ \text{mul} = \text{user\_embedding} \odot \text{item\_embedding} $$
- Concatenation: The user embeddings, item embeddings, and interaction vector are concatenated together to form a unified vector.
- Fully Connected Layers: The concatenated vector is subsequently passed through a series of fully connected layers, each equipped with a rectified linear unit (ReLU) activation function. Each layer can be mathematically represented as: $$ dense= ReLU (W ∗ previous layer + b) $$, where $$ W $$ and $$ b $$ denote the weights and biases of the layer, respectively.
- Output Layer: The final fully connected layer is linked to a single output neuron without an activation function. This neuron produces the predicted interaction, which can be mathematically represented as: $$ prediction = W * last\_dense\_layer + b_i$$
- Loss Function: The model employs the mean squared error loss function, which is appropriate for regression tasks. The loss function can be mathematically represented as: $$ loss = mean((prediction - true\_interaction)^2)$$. 
We can find the NCF model architecture below:

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/model_architecture_srm.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Model Architecture.
</div>

### Experiments

Before evaluating and presenting results for our NCF model we evaluated our topic model using the framework called OCTIS. In addition to visually
inspecting the topics we also used diversity metric to evaluate the topics separately for users and for products. The diversity score is the metric that measures how diverse the **topk** words of a topic are to each other. We chose $$k=10$$ and diversity score for users topics was **0.99** while for products it was **0.97** both suggests that both models were good while users model being slightly better. Once we were confident with our topic models we went on with NCF model. 

Moreover, in the course of our experiments, we trained the Neural Collaborative Filtering model under varying degrees of complexity. The most complex variant of the model incorporated four dense layers, weight decay, and dropout regularization succeeding each layer. On the other hand, the least complex model was composed of merely two dense layers and did not include any form of regularization.

Interestingly, a model of intermediate complexity as shown above, comprising three dense layers, yielded the most promising results. The optimal hyperparameters for this model were determined to be a learning rate of **0.001**, a batch size of **1024**, and a total of **5** training epochs.

The performance of this model was evaluated based on the training,validation, and test losses. The model achieved a training loss of **1.2275e-06**, indicative of its ability to learn effectively from the training data. The validation loss was recorded as **2.1576e-08**, suggesting that the model was able to generalize well to unseen data. Finally, the model demonstrated a test loss of **2.1152935e-08**, further attesting to its robust predictive performance on entirely new data. 

In our analysis, we observed that users who posted reviews about soups were clustered into a single topic. A selection of these user reviews is presented in table on the left below. Based on their interest in soups, these users were recommended products related to this category, as depicted in table on the right below . Notably, these recommendations included items such as soya sauce and other Japanese-related products. This aligns with the understanding that soup is a significant component of Japanese cuisine. Thus, the recommendation system effectively identified and catered to the users' apparent interest in soup-based dishes, particularly those of Japanese origin.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/users_soup.png" title="Soup Users" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/products_recom.png" title="Recommended Products" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Left image shows users who ordered soup and on the left are the products that our model recommended for them.
</div>

### Conclusion 

To conclude, this project has demonstrated the effectiveness of utilizing Neural Collaborative Filtering (NCF) for product recommendation, as an alternative to traditional matrix factorization methods. A key innovation of our approach lies in the incorporation of semantic representations of users and products, which were employed as pre-trained embeddings in the model. By leveraging these semantic representations, our model was able to capture more nuanced user-product interactions, going beyond mere transactional data to encompass the underlying characteristics and preferences of users and products. This resulted in a more sophisticated and potentially more accurate recommendation system.

Looking ahead, there are several avenues for further enhancing the performance and utility of our model. One potential improvement could involve refining the semantic representations of users and products, perhaps by incorporating additional sources of data or employing more advanced natural language processing techniques. Additionally, the model's performance could potentially be boosted by exploring more complex architectures or more sophisticated training strategies. For instance, techniques such as learning rate scheduling, advanced regularization methods, or ensemble methods could be investigated.




