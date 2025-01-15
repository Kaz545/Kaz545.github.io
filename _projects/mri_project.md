---
layout: page
title: BrainMRI-GPT 
description: Neuro-imaging analysis using Multimodal models.
img: assets/img/tumor.jpeg
importance: 1
category: work
related_publications: false
---

Brain tumors significantly impact global mortality rates, making their detection and management a critical concern. Accurate segmentation of brain tumors from neuroimaging data plays a vital role in enhancing disease diagnosis, guiding treatment strategies, monitoring progression, and supporting clinical research. Effective segmentation is essential to identify both the tumor’s location and its size.

We propose a novel approach to brain tumor detection that harnesses CLIP’s ability to align text descriptions with visual features. By fine-tuning this capability, we aim to identify and segment brain tumors with high accuracy, enabling clinicians to localize the tumor and assess its extent. To further enhance the usability of this system, we incorporate a Large Language Model (LLM) to facilitate interactive dialogues. This addition allows for natural language explanations of the model’s predictions, making the system more interpretable and suitable for clinical decision-making. Our approach not only seeks to advance diagnostic accuracy but also strives to bridge the gap between AI predictions and human understanding, fostering trust and collaboration in medical contexts.

### Data 

The visual data was taken from <a href="https://www.kaggle.com/datasets/ashkhagan/figshare-brain-tumor-dataset"> here </a>. This brain tumor dataset contains 3064 T1-weighted contrast-inhanced images from 233 patients with three kinds of brain tumor: meningioma (708 slices), glioma (1426 slices), and pituitary tumor (930 slices). As this dataset didnt contain the negative samples and we needed them to train our model so we took the negative samples, i.e samples with no brain tumor from here <a href="https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset"> here </a>. After augmentation the total number of samples in our dataset were 4659. We had two columns one corresponds to the MRI scan image and the second column was the binary tumor mask. For us the first column, the image was the input and the mask was something that we were trying to predict hence that was our target.

### Methodology

The below image illustrates a high-level overview of our approach. We utilize both normal and abnormal textual descriptions, which are processed through the CLIP text encoder to obtain corresponding text embeddings. Simultaneously, normal and abnormal MRI images are passed through the CLIP visual encoder. From the visual encoder, we extract outputs from the 6th, 12th, 18th, and 24th hidden layers, representing intermediate patch-level features.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/e2e_flow_mri.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    E2E Flow.
</div>

Since these visual features exist in a different embedding space than the encoded text, we align them using a trainable decoder. This decoder maps the visual features into the joint embedding space, ensuring compatibility with
the text embeddings. The alignment is performed such that the dot product (or similarity measure) between the transformed visual features and text embeddings highlights the most relevant patches, producing precise tumor localization results. These localization maps, combined with carefully engineered textual prompts, are then passed to a large language model (LLM). This approach enhances the LLM’s understanding, enabling it to provide more accurate and contextually relevant responses about the tumor’s characteristics and location.

In this study, we leverage the CLIP framework with the ViT-B/14 (Vision Transformer Base) model variant. Both the visual and textual encoders of CLIP are utilized: the visual encoder processes MRI image embeddings, while the textual encoder generates embeddings from associated textual descriptions. These multimodal embeddings serve as the foundation for tumor detection and localization.

While image data for the visual encoder was readily available, corresponding textual data for the text encoder was absent. To address this, we systematically generated 20 text prompts for each category (normal and abnormal images). These prompts, identical across all images in their respective categories, provided descriptive textual input to
the CLIP textual encoder.

Moreover we utilize the prompt ensemble technique where we averaged the the text features extracted by the text encoder as the final text features for both normal and abnormal texts $$ Ft ∈ R2×C $$, where $$ C $$ denotes the number of channels of the feature. Let $$ Fc ∈ R1×C $$ represent the image features derived from the visual encoder for classification. The relative probabilities $$ s $$ of the MRI being classified as either normal or abnormal can then be expressed as:

$$ s = \text{softmax}(F_c F_t^T) $$

We use the probability corresponding to the abnormal class as the anomaly score for the image. 

#### Decoder

The CLIP model, originally designed for classification tasks, maps only the final image features to the joint embedding space, enabling direct comparison with text features. However, intermediate image features from earlier layers, which are crucial for fine-grained localization, are not inherently aligned with the joint embedding space. To address this limitation, we introduce a decoder that maps these intermediate image features into the joint embedding space, enabling meaningful comparisons with text embeddings. Using the transformer-based architecture ViT, we empirically divide the layers into four stages—specifically, the 6th, 12th, 18th, and 24th hidden layers. For each stage, the decoder processes the corresponding output features.
The decoder is composed of a linear layer followed by a Leaky ReLU activation function, which ensures smooth handling of negative values. A final linear layer completes the mapping process, producing features aligned with the joint embedding space.

This architecture as shown below effectively bridges the gap between intermediate visual features and textual embeddings, facilitating accurate patch-level similarity comparisons and improving localization performance. Furthermore, We employ the linear combination of Focal loss and Dice loss to train the decoder.

$$ LOSS = DICE_LOSS + FOCAL_LOSS $$

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/decoder_mri.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Decoder Structure.
</div>

#### LvLM

After obtaining the tumor localization results, we passed them along with the corresponding textual prompts to a large language model (LLM) to extract detailed information from the MRI. We utilized Meta’s Llama 3.2 11B
Vision Instruct Model, a powerful multimodal LLM optimized for visual tasks The Llama 3.2-Vision collection consists of pre-trained and instruction-tuned models that excel in image reasoning, visual recognition, captioning, and answering general questions about images.

### Experiments

Training was done over 3 epochs, we utilized the Adam optimizer with specific hyperparameter settings: a learning rate of 0.01 and momentum parameters of 0.5 and 0.999. These values were selected to balance stability and convergence speed, especially given the inherent variability in the training data. The batch size was set to 16, ensuring efficient training while maintaining computational feasibility.
On the large language model (LLM) side, we explored various prompting paradigms to evaluate their efficacy in different scenarios. These included zero-shot prompting, where we just passed the raw image and asked for the tumor location; instruction-based prompting, where we passed the localization results of the raw images and asked for the tumor location explicitly stating that red region indicates the tumor; and chain-of-thought (CoT) prompting, which encouraged step-by-step reasoning to solve more complex problems, here again we passed the localization results and
just asked for the tumor location and details. These prompting strategies were assessed to determine their impact on the model’s ability to interpret and generate meaningful outputs.

To evaluate our model’s localization performance, we relied on pixel-level metrics that are well-suited for assessing the precision and reliability of segmentation tasks. Specifically, the pixel-level AUROC score achieved an impressive 99.1, indicating the model’s ability to distinguish between tumor and non-tumor regions with high accuracy. Additionally, the accuracy score of 99.02 demonstrates that the majority of pixel classifications align with the ground truth, reflecting the robustness of the model in identifying tumor locations. However, the precision score of 73.22 highlights
a slightly lower proportion of correctly identified tumor pixels among all predicted positives, suggesting some room for improvement in minimizing false positives.

### Results

The below table presents the tumor localization results achieved by our approach. It clearly demonstrates the method’s effectiveness in accurately identifying and segmenting tumor regions. As evidenced by the visualized outcomes, the approach reliably highlights tumor locations with minimal deviations, underscoring its potential for precise tumor segmentation in neuro-imaging applications.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/results_mri.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Qualitative Results.
</div>


Moreover, the table below presents the responses generated by the LvLM using various prompting techniques. The zero-shot approach yielded suboptimal results, as the LvLM struggled to accurately identify the tumor location and provided incorrect details and answers. However, incorporating localization results in the prompt alongside instruct and CoT strategies significantly improved the LvLM’s performance. These methods enabled the model to
correctly identify the tumor’s location and produce accurate responses.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/lvlm_results.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    LvLM Response.
</div>

