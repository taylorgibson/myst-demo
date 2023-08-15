---
title: Doxorubicin Efficacy Prediction for Glioblastomas Using Deep Learning and Differential Equations
date: 2023-08-10

authors:
  - name: Arnav Garg
    email: arnavkumargarg@gmail.com
    affiliations:
      - North Carolina School of Science and Mathematics, Durham, North Carolina
    roles: 
      - Conceptualization
      - Data Curation
      - Formal Analysis
      - Funding Acquisition
      - Writing – Original Draft
      - Writing – review & editing
  - name: Maruthi Vemula
    email: vemula.maruthim@gmail.com
    corresponding: true
    affiliations:
      - North Carolina School of Science and Mathematics, Durham, North Carolina
    roles:
      - Software
      - Supervision
      - Validation
      - Visualization
      - Writing – Original Draft
      - Writing – review & editing
  - name: Pranav Narala
    email: pranavnarala@gmail.com
    affiliations:
      - North Carolina School of Science and Mathematics, Durham, North Carolina
    roles:
      - Investigation
      - Methodology
      - Project Administration
      - Resources
      - Writing – Original Draft
      - Writing – review & editing

abbreviations:
  CNN: Convolutional Neural Network
  NIfTI: Neuroimaging Informatics Technology Initiative
  ODE: Oridnary Differential Equation
  MRI: Magnetic Resonance Imaging

exports:
  - format: pdf
    template: arxiv_two_column
    output: exports/paper-version02-arxiv.pdf
  - format: pdf
    template: arxiv_nips
    output: exports/paper-version02-arxiv-nips.pdf
  - format: docx
---

+++ {"part": "abstract"}
This paper presents a novel approach for predicting the efficacy of Doxorubicin treatment for glioblastoma. Glioblastomas’ rapid growth places them among the most aggressive cancers, killing thousands of Americans every year {cite:p}`https://doi.org/10.1093/neuonc/now207`. The rapid progression of glioblastoma coupled with the high cost of cranial imaging makes clinical decision-making uniquely challenging. Doxorubicin is a commonly used chemotherapy drug to treat glioblastomas. However, predicting the treatment's efficacy remains challenging and time-consuming. Inaccurate predictions can lead to ineffective treatments, severe side effects, and even death. To address this issue, a framework was developed that amalgamates deep learning and differential equations to accurately predict tumor volume growth over time. Specifically, a 2D U-net convolutional neural network (CNN) was employed to segment MRI brain tumor regions and obtain initial volumes. The Gompertz differential equation was then utilized to model the predicted tumor volume growth over time, achieving a mean absolute percent error of 4.98%. The Gompertz model was modified to incorporate the cytotoxic effect of Doxorubicin treatment. The methodology predicted the final tumor volume of the tumor after being treated with Doxorubicin over multiple 21-day cycles, enabling us to predict the efficacy of treatment and identify patients who may benefit most from this therapy. A user-friendly web application was developed to allow users to input NIfTI files of MRI scans and receive as output a time-course prediction of tumor volume with and without chemotherapy treatment. This approach provides a prediction of Doxorubicin treatment efficacy and can improve patient outcomes and treatment plans.
+++ 

## Introduction

```{figure} #horsepower
:name: demo-notebook-link
:alt: This is a demo of a notebook link
:align: center

Unrelated to this article, but just showing you that this is possible.
```

### Background
Although rare, the 100 different subtypes of gliomas account for a disproportionately high number of cancer-related deaths, emphasizing the high stakes involved with the disease (Miller et al., 2021). These tumors are separated into four main grades, first to fourth, with each grade representing an increasingly faster-growing cancer. Glioblastoma Multiforme (glioblastoma) is a grade four glioma characterized by variances in tumor appearance, ranging from soft and yellow due to tissue necrosis, or showing marked cystic degeneration or hemorrhage {cite:p}`noauthor_notitle_2017`. Although the exact underlying cause of glioblastoma is unknown, it can occur in people with genetic syndromes such as neurofibromatosis type 1 {cite:p}`noauthor_glioblastoma_nodate`. glioblastoma is a rare tumor with a global incidence of less than 10 patients per 100,000 people; the average survival rate of glioblastoma is around 14-15 months after diagnosis.

Along with the unknown cause of the disease and the lack of successful treatment, glioblastoma has other significant clinical challenges. The rapid growth of the tumor leads to a minute period of time between diagnosis and treatment decisions. Because of this, patients often only take 1-3 scans. The lack of MRI scans exacerbates the uncertainty of treatment decisions, as doctors are unsure of what the growth may look like. The rapid growth leads to an average time of 13 days between diagnosis and treatment decisions {cite:p}`https://doi.org/10.1093/noajnl/vdab053` There are two main options for treatment: surgery and chemotherapy. This places families of patients in a difficult position as they decide to spend hundreds of thousands of dollars on an uncertain treatment or let their patients live their last few months undergoing palliative treatment. Even after deciding on treatment, choosing between surgery or chemotherapy comes with their respective drawbacks. When performing the surgery, doctors decide between removing large amounts of tissue and sacrificing brain tissue, or cutting the exact amount and risking recurrence from hidden tumor cells. Any treatment option for patients with glioblastoma is shrouded in a dense fog of uncertainty about its efficacy. 

### Research Goal

Accurate and precise prediction of brain tumor progression is crucial to prognosis, as both the growth rate and location of the tumor are integral to its impact on the central nervous system. The viability of tumor resection as a surgical intervention is limited by imaging accuracy and the ability to forecast the invasion of tumors into regions of healthy tissue. Based on the health of the patient and the progression of the disease, palliative care may be preferred to aggressive treatment. With more accurate measurement and prediction of tumor volumes, patients and clinicians will be able to make more informed decisions about the goals of care.

Our work here provides a scan-to-forecast pipeline that will serve as a tool for glioblastoma clinical decision-making. Using a 2D Splicing Convolutional Neural Network (CNN) and Ordinary Differential Equation (ODE) to accurately predict glioblastoma progression bridges the gap between current prediction processing. Along with predicting tumor growth over time in the absence of treatment, the model also predicts tumor volume in the presence of Doxorubicin treatment over 21-day intervals. We hypothesize that this 3 step framework will enable us to efficiently perform brain tumor prognosis to assist with medical decision-making. These predictions may assist in correcting treatment plans for patients with gliomas.This knowledge may also be useful in determining the best types of care, and whether or not the treatment would be effective or worthwhile.

## Methods

### Dataset
The data used for this project was retrieved from the Brain Tumor Segmentation Challenge 2021 hosted by The Center for Biomedical Image Computing and Analytics at the University of Pennsylvania. All of the files are in NIfTI file format in order to provide the highest quality of 3D scans from MRI and CT scans. They describe native (T1), post-contrast T1-weighted (T1Gd), T2-weighted (T2), and T2 Fluid Attenuated Inversion Recovery (T2-Flair) scans {cite:p}`noauthor_miccai_nodate`. The dataset contained 1666 training data folders, each containing a NIfTI file of Flair, T1, and T2 MRI scans and the corresponding segmentation labels done by medical professionals for those scans.

### Data Processing
To prepare training and testing data from the NIfTI files of the various MRI scans, the Python Nibabel module was used to load the files as voxels, represented by a 3D array. This data was processed for use by a 2D segmentation model applied over slices of the 3D array rather than a 3D segmentation model due to the limited computational resources available during this project. This resulted in 155 training examples per voxel. The 2D splices were resized to 128 by 128 pixels using the OpenCV module because the segmentation model can only take fixed input sizes. The splices containing no tumor regions were used as a control for the dataset. Additionally, this same process of taking 2D slices and resizing was done to the segmentation label voxel, which represents what the model’s output should be. The segmentation label contains an integer value for each pixel, representing an integer encoding of which class the image belongs to. 

Due to a 25.5GB of system RAM limit on the program environment, Google Colaboratory, 100 NifTi files were used. Of these, 20 NifTi files were reserved for testing data. This data is used for model validation in order to avoid over-fitting.

### 2D U-Net Segmentation
Semantic segmentation is a computer vision technique in which pixel-level maps are created to get detailed separations of different objects in an image. In this use case, segmentation will be applied to separate out 4 different regions of a tumor, as described in the previous section. The 2D U-Net CNN model applied in this research utilizes an encoder-decoder architecture and was implemented using the Tensorflow and Keras modules. The encoder structure of the model is used to extract important patterns and shapes from the image while removing unnecessary details through downsampling. This is done through a series of convolutional and max pooling layers. This reduces the dimensionality of the data to only maintain important information with a feature map. The decoder structure is then used to convert the feature map to output segmentation maps to expand out the important patterns in the encoded feature map through upsampling. This is done with a similar structure to the encoder, except convolutional 2D transpose layers are used to perform upsampling rather than max pooling layers. There are the same number of encoder and decoder blocks, and the output of corresponding encoder blocks is concatenated onto the output of the previous decoder on that layer.

Once the model architecture was defined, the model was compiled with the Adam optimizer, which is the algorithm used to change the values of model parameters to improve the model’s accuracy. It does this by performing gradient descent on the loss function, which is a mathematical function that describes the inaccuracy of a model’s prediction. The loss function used was sparse-categorical-cross-entropy since the labeled data was integer encoded, meanwhile, the model’s outputs were one-hot encoded. Once the segmentation model was finished training, a function was created to calculate the predicted and labeled volumes of the tumor present in the MRI scan so that the Gompertz Differential Equation Model could be used, which is described in the next section. To calculate the volume, the number of pixels classified as the enhancing tumor class was stored in a variable and multiplied by a scaling constant to convert it to cubic millimeters.

### Fitting Gompertz Differential Equation Model to Experimental Data
The Gompertz model was used to model the growth of the tumor based on the initial volumes from the segmentation results. This ODE model was chosen in particular because, in previously conducted studies, it has been shown to successfully model breast and lung cancer growth {cite:p}`wang_student_nodate`. The model supposes quasi-logistic growth, wherein an initial per-capita cell division rate of a begins to slow as the tumor approaches a maximal volume of B. The model from {cite:p}`taib_mathematical_2022` was used and is shown below.

:::{note} Base Gompertz Differential Equation Model
```{math}
:label: gompertz-base
\frac{dV}{dt}=a\big(\ln(B) - \ln(V)\big) \cdot V
```
$V$ - tumor volume (mL), $t$ - time (days), $a, B$ - growth constants
:::

In order to determine the parameters $a$ and $B$, the model was fit to in vivo experimental growth data collected from {cite}`taib_mathematical_2022`, which measured exponential growth rates for small, medium, and large tumors (defined as small tumors < 3.88mL, medium tumors >= 3.88mL and < 36.88mL, large tumors >= 36.88mL in the study). An exponential model based on the average tumor growth for each size was used to calculate the final tumor volumes after 13 days {cite:p}`https://doi.org/10.1093/noajnl/vdab053`, the average time from diagnosis to starting treatment. The relative error of the Gompertz model compared to these exponential in vivo data predictions was calculated to be 4.98%.

### Adding Chemotherapy Term for Gompertz Model
The Gompertz model was able to fit well to the exponential growth rates. However, the standard model does not account for chemotherapy treatment administered to patients, which is essential to take into consideration when modeling tumor growth. In order to do this a chemotherapy-mediated cell death term is subtracted from the standard model to account for decreasing tumor size {cite:p}`https://doi.org/10.1186/s12885-016-2164-x`. The per-capita cancer cell death rate is shown to decrease exponentially from the time of Doxorubicin administration. This is shown below.

:::{note} Modified Gompertz Differential Equation Model
```{math}
:label: gompertz-modified
\frac{dV}{dt} = a\big(\ln(B) - \ln(V))\cdot V - \gamma e^{-r(t \% 21)} V 
```

$V$ - current tumor volume (mL), $t$ - time (days), $a, B$ - growth constants, $r = \ln(2) / t_{\text{half}}$, $\gamma$ - proportionality constant for cell death rate (1/days)
:::

Doxorubicin is typically administered via a single dose of 60-75 mg/m2 every 21-28 days {cite:p}`khan_doxorubicin:_nodate`. The half-life of the drug is 20-30 hours (approximated here as 25 hours) {cite:p}`https://doi.org/10.2165/00003088-200342050-00002`. The percent symbol represents the modulus division of the current time. This is done to get the time since administration for 21-day cycles. The value of the decay rate r was calculated to reproduce the half life approximation of 25 hours. The parameters a and B are taken to have the values fit to the model without intervention in the previous section.

The peak cell death rate  is estimated using observations from Eramo et al, wherein in vivo tumor stem cell death rates were sampled over the initial 48-hour period after Doxorubicin drug administration. It was measured that tumor volume decreases by 20% after the first 2 days. We have selected a value of  to match this observation. The final parameter table for the fitted model is shown below.

```{list-table} Gompertz Model Parameter Table: This table indicates the different parameters, values, units, and sources for each term in the two Gompertz differential equations.
:header-rows:1
:name: gompertz-table
* - Parameter
  - Value
  - Units
  - Source
* - $V$
  - Tumor Volume
  - mL
  - None
* - $t$
  - Time
  - days
  - None
* -  $t_{\text{half}}$
  - 1.04
  - days
  - {cite}`https://doi.org/10.2165/00003088-200342050-00002` (20-30 hours approximated as 25)
* - $a$
  - 0.0125
  - days$^{-1}$
  - Fitted to {cite}`taib_mathematical_2022` Growth Data
* - $B$
  - 25000 
  - mL
  - Fitted to {cite}`taib_mathematical_2022` Growth Data
* - $\gamma$ 
  - 0.252
  - days$^{-1}$ 
  - Fit to simulate 20% decrease from Eramo et al.
* - $r$
  - 0.666 
  - days$^{-1}$ 
  - Calculated from $t_{\text{half}}$
```

## Results

Our study produces a comprehensive three-step framework that forecasts glioblastoma tumor volume using MRI scans as input. The first aspect of the framework is a 2D U-Net segmentation model trained to separate various biological tumor regions. Then, the initial volume of the tumor is calculated as input to a Gompertz differential equation model which predicts the growth of a tumor with respect to an initial volume. Finally, the model is re-evaluated with the inclusion of chemotherapy via Doxorubicin administration in 21-day cycles.

```{figure} ./images/image14.png
:name: overall-framework
:alt: Figure describing amalgamation of differential equation and deep learning components, as well as output flow for each section.
:align: center

Overall Framework: Figure describing amalgamation of differential equation and deep learning components, as well as output flow for each section.
```

Our system utilizes the segmentation data to determine the size of the tumor from a conversion factor that converts the number of pixels to the volume in millimeters. The tumoral volume computed by the CNN is then passed in as the initial condition of a differential equation model, which forecasts a time course of tumoral volume in both the presence and absence of Doxorubicin treatment. The model accounts for the intermittent administration of Doxorubicin in 21-day cycles by allowing the cytotoxic effect to decay exponentially between treatments.

Finally, a website was built that culminates this framework in a seamlessly integrated fashion. When loaded into the website, medical professionals can upload Flair, T1, and T2 MRI scans of the patient. Developed using HTML, CSS, JS, and Flask, this online platform ensures accessibility and ease for medical professionals worldwide. 

```{list-table} Model Segmentation Predictions and Corresponding Labels for Various Flair, T1, and T2 scans: This table indicates the highlighted regions from the model’s predictions and the dataset labels for specific regions. It contains flair, T1, and T2 scans of a sagittal MRI scan with a tumor in the right cerebral cortex.
:header-rows:1
:name: model-seg-table
* - Segmentation Overlay on Scan
  - Flair
  - T1
  - T2
* - Predicted
  - ![alt](./images/image11.png 'title') 
  - ![alt](./images/image16.png 'title') 
  - ![alt](./images/image17.png 'title') 
* - Actual
  - ![alt](./images/image19.png 'title') 
  - ![alt](./images/image18.png 'title') 
  - ![alt](./images/image15.png 'title') 

```

[](#model-seg-table) contains a table that shows images of the model segmentation map predictions and their corresponding label maps for visualization of prediction results. Flair, T1, and T2 scans are shown. The dark blue regions correspond to brain MRI. The aqua regions are necrotic tumor regions. The yellow regions are peritumoral edematous/invaded tissue. The red regions are the enhancing tumor regions. This table is produced after the 2D-Unet CNN has completed running. 

```{figure} ./images/image12.png
:name: projected-growth-rate
:alt: Projected Growth Rate of Tumor over 40 Days with and without Chemotherapy: Predicted growth of the tumor with and without Doxorubicin for an initial tumor volume of 1.017 mL. The drug was administered after 13 days due to the time between diagnosis and treatment.
:align: center

Projected Growth Rate of Tumor over 40 Days with and without Chemotherapy: Predicted growth of the tumor with and without Doxorubicin for an initial tumor volume of 1.017 mL. The drug was administered after 13 days due to the time between diagnosis and treatment. {cite:p}`10.1093/noajnl/vdab053`.
```

[](#projected-growth-rate) displays the primary outcome of this prognosis framework which is a graph showing the modeled growth of a patient’s tumor with and without chemotherapy simulation produced from our Gompertz ODE model. As seen above, the administration of Doxorubicin in 21-day cycles creates a sharp decrease in tumor size during the initial 2-3 day period. However, the tumor soon returns to its normal growth after the drug exponentially decays from the bloodstream. This leads to a net overall decrease in the tumor size of approximately 2 mL. This figure shows how the administration of Doxorubicin would delay the growth of a tumor based on patient-specific conditions.

## Discussion

Computational modeling of diseases has had a profound impact on society as doctors and radiologists are increasingly making use of machine learning and AI to analyze data. Of these models, classification is a common sector of computational medicine that is used for brain tumors. However, tumor progression is under-researched {cite:p}`https://doi.org/10.1124/pr.117.014944`. The rapid growth and low survival rate of Glioblastomas make our research problem even more important to explore. The goal of this research was to test our hypothesis of how computational modeling techniques could be combined to create a tumor prognosis method.

A common issue with developing Glioblastoma prognosis frameworks is the lack of time series MRI data. Due to the high cost of taking MRI scans and the short time from patient diagnosis to death, the time series datasets that do exist may only contain very few amounts of scans, which is not enough to train complex machine learning models to identify trends effectively and make accurate predictions. Although some prognosis models exist, our work approached this problem by amalgamating various deep learning and ODE models, highlighting the novelty of this research. By splitting the prognosis tasks into 3 subtasks for which there are datasets specifically, the framework is able to combat the issue of missing time series MRI scans. This framework consisted of three main methods: A 2D U-Net to segment various biological tumor regions, a Gompertz differential equation model for tumor growth, and a chemotherapy simulation component. Along with the novelty of our ensemble model, the inclusion of the cytotoxic effect of intermittently administered chemotherapy in the Gompertz model is unique to our research. 

There is no existing cure for Glioblastomas and only treatments that delay the growth of tumors. Additionally, the high costs of these treatments and varying socioeconomic factors for patients make it difficult for them and oncologists to determine future steps for palliative care. The 3 main aspects of decision-making in oncology are contextual factors, decision-maker-related criteria, and decision-specific criteria. Contextual factors mainly consist of a patient’s socioeconomic factors, government factors, and insurance coverage. Decision-maker-related criteria refer primarily to the behavioral science and cognitive bias aspect of decision-making. The final component, decision-specific criteria, includes factors such as age, tumor stage, emotional stress, treatment toxicity, time pressure, treatment intent, etc {cite:p}`https://doi.org/10.1159/000492272`.

With all of these complex, interconnected factors present in informing clinical decision-making, oncologists are faced with an extremely difficult task. Our framework aims to provide insights into the decision-specific criteria of decision-making. That is, through providing time-course tumoral volume predictions with and without the administration of chemotherapy, patients and oncologists can better understand the treatment’s impact on tumor growth. Using the modeled tumor growth and their background knowledge of symptoms caused by tumor growth in various neural regions, oncologists can determine how treatment may influence the performance status of patients. They can better understand how treatment will prolong tumor growth, potentially delaying the time until worse symptoms develop. Taking this information, they can combine it with treatment intent, time pressure, treatment toxicity, tumor stage, and emotional stress to gain insights into treatment factors. They can combine these insights with decision-maker-related criteria and contextual factors to make more informed decisions for proceeding with treatment. To make our solution accessible for medical professionals to use for this purpose, we implemented it on a web app where users can upload MRI scan files of a specific patient, and a personalized prognosis report is generated.

In future iterations of this work, we hope to improve the predictive power of the pipeline with regard to patient outcomes in two key ways. The first is to gain access to more computational resources and train a 3D U-Net segmentation model. Using a 3D U-Net, as opposed to a 2D U-Net, will allow the MRI scans to be processed all at once allowing for greater spatial context for image segmentation and classification. Secondly is to additionally train the CNN to determine what neural regions are at risk for tumoral invasion, which will enable medical professionals to more precisely forecast overall morbidity in the patient with respect to various treatment protocols.

## Conclusion

### Competing Interests
The authors have no competing interests to declare.

+++ {"part": "data-availability"}

Open access data may be found here: https://www.med.upenn.edu/cbica/brats2020/data.html 

+++

+++ {"part": "acknowledgements"}

We would like to thank Mr. Keethan Kleiner and Dr. Michael Lavigne for their unwavering support as our mentor, Dr. Heather Mallory and Linden James for their priceless feedback, and Dr. Joe LoBuglio for his support in providing resources for this project through the Ryden AI program at the North Carolina School of Science and Mathematics.

+++