### Krishna Patel, Justin Deal, Anthony Marshall, Ben Pooser, Trevor Stanfield

# MOTIVATION OF PROJECT

Breast cancer remains the second most common cancer diagnosis across the world while also disproportionately affecting women, and many types can show relatively few signs and symptoms. It is estimated that 1 in 8 women in the United States are at risk of developing breast cancer. In 2020, it is predicted that 42,000 women in the United States will die from this disease [5]. The inconspicuous nature of this disease along with its prevalence are powerful testaments to its deadliness, and in an effort to lessen its threat, we have decided to concentrate on how we could best identify the disease for faster and more effective treatment.

Invasive ductal carcinoma (IDC) is the most common type of breast cancer, representing over 80% of all cases [1]. In order to identify the type of cancer and its invasiveness, pathologists scan large sections of benign regions of tissue in order to identify malignant areas; however, this process can be time consuming and difficult [2]. With this issue in mind, we have decided to focus our efforts on extracting the characteristic features of IDC from a dataset of breast histology images using various machine learning techniques. Once we extract these features, we will use binary classification to determine if IDC is present, and then compare our methods to find the model with the highest accuracy that will also generalize best to real world applications.  

# DATASET AND PREPROCESSING

## **Dataset**
Our dataset is a Kaggle dataset provided by Paul Mooney. It consists of 277,524 breast histology images which originated from 162 whole mount Breast Cancer slide images [3].

Images in this dataset are of size 50x50x3. Each file name follows this format, which reveals if the image is cancerous images or not: u_xX_yY_classC.png => example 10253_idx5_x1351_y1101_class0.png.


- u : the patient ID (i.e. 10253_idx5)
- X : the x-coordinate of potential IDC (i.e. x1351)
- Y : the y-coordinate of potential IDC (i.e. y1101)
- C : the class where 0 is non-IDC and 1 is IDC [3]

### Example of the images:

![Examples of Dataset](./images/dataSetExample.png)

## **Data Preprocessing**

We processed the data, and based on their class (0 for non-cancerous and 1 for cancerous), we were able to split our dataset. From this, we learned that these images contained 277524 total patches that we could analyze.

- `Total Image Patches:  277524`
- `Total Positive Patches:  78786`
- `Total Negative Patches:  198738`

# Methods

## **Convolutional Neural Network**

### Approach
We chose to use a convolutional neural network (CNN) for detecting IDC because of its ability to identify the defining spatial properties of images and because of its pervasiveness throughout fields involving image classification such as computer vision.  Using a CNN, we were able to achieve a recall of x percent and an accuracy of y percent. Before training our CNN on our data, there were several preprocessing steps that needed to occur.

One of the initial observations of the data was that the number of positive samples was significantly lower than the number of negative samples.

<!-- ![Graph 1. Ratio of Negative to Positive Samples](images/Graph_RatioP2N_CNN.png) -->
<img src="images/white.jpg" width = "10"/>
<div>
<img src="images/white.jpg" width = "200"/>
<img src="images/Graph_RatioP2N_CNN.png" width = "400"/>
</div>

<img src="images/white.jpg" width = "10"/><img src="images/white.jpg" width = "10"/><img src="images/white.jpg" width = "10"/><img src="images/white.jpg" width = "10"/><img src="images/white.jpg" width = "20"/><img src="images/white.jpg" width = "20"/><img src="images/white.jpg" width = "20"/><img src="images/white.jpg" width = "20"/><img src="images/white.jpg" width = "20"/><img src="images/white.jpg" width = "20"/><img src="images/white.jpg" width = "20"/><img src="images/white.jpg" width = "20"/><img src="images/white.jpg" width = "20"/><img src="images/white.jpg" width = "20"/><img src="images/white.jpg" width = "10"/>_Graph 1. Ratio of Negative to Positive Samples_



<img src="images/white.jpg" width = "10"/>

### How we improved our approach

The imbalanced class distributions will lead to our model generalizing poorly. This problem is exacerbated if we have unrepresentative test/train splits. In order to combat the unbalanced class problem and increase the generalizability of our model, we used stratified k fold validation over 80% of the data. The other 20% was saved for testing. By using stratified k-fold validation, we ensured that each validation fold had an even proportion of positive and negative samples.

<!-- ![Graph 2. Ratio of Negative to Positive Samples Manifold](images/Graph_RatioP2NManifold_CNN.png) -->

<img src="images/white.jpg" width = "10"/>

<div>
<img src="images/white.jpg" width = "200"/>
<img src="images/Graph_RatioP2NManifold_CNN.png" width = "400"/>
</div>

<img src="images/white.jpg" width = "10"/><img src="images/white.jpg" width = "10"/><img src="images/white.jpg" width = "10"/><img src="images/white.jpg" width = "10"/><img src="images/white.jpg" width = "20"/><img src="images/white.jpg" width = "20"/><img src="images/white.jpg" width = "20"/><img src="images/white.jpg" width = "20"/><img src="images/white.jpg" width = "20"/><img src="images/white.jpg" width = "20"/><img src="images/white.jpg" width = "10"/><img src="images/white.jpg" width = "10"/><img src="images/white.jpg" width = "10"/><img src="images/white.jpg" width = "20"/><img src="images/white.jpg" width = "10"/>_Graph 2. Ratio of Negative to Positive Samples Five Folds_


<img src="images/white.jpg" width = "10"/>

The model was also improved by changing the color space of the input image. The RGB colorspace is a non-uniform colorspace, and it does not separate color from luminance like HSV or LUV do. The images are normalized and converted to the LUV colorspace as a result. LUV is perceptually linear and is therefore better for detecting color differences than RGB.


<!-- ![Image 1. Color Space Modification: HSV](images/ColorSpace_HSV.png) ![Image 2. Color Space Modification: LUV](images/ColorSpace_LUV.png) -->

<img src="images/white.jpg" width = "10"/>

<div>
<img src="images/white.jpg" width = "110"/>
<img src="images/hsv.png" />
<img src="images/white.jpg" width = "100"/>
<img src="images/ColorSpace_LUV.png" />
</div>


<img src="images/white.jpg" width = "30"/><img src="images/white.jpg" width = "20"/><img src="images/white.jpg" width = "20"/><img src="images/white.jpg" width = "20"/><img src="images/white.jpg" width = "20"/>_Image 1. Color Space Modification: HSV_
<img src="images/white.jpg" width = "20"/>
<img src="images/white.jpg" width = "20"/>
<img src="images/white.jpg" width = "10"/>
_Image 2. Color Space Modification: LUV_


<img src="images/white.jpg" width = "20"/>

  After the data wrangling and preprocessing, the images were fed into a CNN with 7 convolution layers, 3 max pooling layers, a hidden dense layer, and an output layer. We also incorporated dropout layers into our model to help prevent overfitting the data.

<img src="images/white.jpg" width = "10"/>


<img src="images/CNNlayers.png"/>

<img src="images/white.jpg" width = "10"/><img src="images/white.jpg" width = "10"/><img src="images/white.jpg" width = "10"/><img src="images/white.jpg" width = "10"/><img src="images/white.jpg" width = "20"/><img src="images/white.jpg" width = "20"/><img src="images/white.jpg" width = "20"/><img src="images/white.jpg" width = "20"/><img src="images/white.jpg" width = "20"/><img src="images/white.jpg" width = "20"/><img src="images/white.jpg" width = "10"/><img src="images/white.jpg" width = "10"/><img src="images/white.jpg" width = "10"/><img src="images/white.jpg" width = "20"/><img src="images/white.jpg" width = "20"/><img src="images/white.jpg" width = "20"/><img src="images/white.jpg" width = "20"/><img src="images/white.jpg" width = "10"/>_Graphic 1. Visualization of Graphics Layers_

<img src="images/white.jpg" width = "10"/>

The resulting learning curve shows the average loss and accuracy of the 5 validation folds plotted against the training data over 30 epochs. The recall chart was omitted because both the training and validation recall followed a strong logarithmic curve.

<div>
<img src="images/white.jpg" width = "30"/>
<img src="images/acc_curve.png" />
<img src="images/white.jpg" width = "30"/>
<img src="images/losscurve.png" />
</div>


From these charts, we can see that both the validation loss and validation accruacy peak at the 16th epoch. The model is then trained on the full 80 percent training split over 16 epochs. 

-- confusion matrix

![](images/IsToDo.png)

--blurb describing next --

![](images/IsToDo.png)

--blurb

![](images/IsToDo.png)

--heat map

![](images/IsToDo.png)

--Results--

![](images/IsToDo.png)

# Supervised Learning Models
###### Linear Regression, k-Nearest Neighbors, Support Vector Machine, and Neural Network

In addition to the convolutional neural net, we ran other supervised learning models such as linear regression, k-nearest neighbors, support vector machine, and neural network. We utilized sklearn’s Kfold and split the dataset into 10 training tests. Then we used sklearn’s LogisticRegression, RandomForestClassifier, KNeighborsClassifier, Support Vector Machine SVC, DecisionTreeClassifier and compared their mean accuracies. We ran these methods twice: first on the data set and then on a data set that we had preprocessed with PCA.

## Data Preprocessing
With such a large data set size, we wanted to see what would happen to our results and processing time if we ran PCA on the data. By using PCA, we would be able to reduce the number of features in our dataset while maintaining as much information as possible. We used a 95% variance. This data can be visualized in the scree plot below.

<img src="images/white.jpg" width = "10"/>

<div>
<img src="images/white.jpg" width = "200"/>
<img src="images/scree.png" width = "400"/>
</div>

<img src="images/white.jpg" width = "10"/><img src="images/white.jpg" width = "10"/><img src="images/white.jpg" width = "10"/><img src="images/white.jpg" width = "10"/><img src="images/white.jpg" width = "20"/><img src="images/white.jpg" width = "20"/><img src="images/white.jpg" width = "20"/><img src="images/white.jpg" width = "20"/><img src="images/white.jpg" width = "20"/><img src="images/white.jpg" width = "20"/><img src="images/white.jpg" width = "20"/><img src="images/white.jpg" width = "20"/><img src="images/white.jpg" width = "20"/><img src="images/white.jpg" width = "20"/><img src="images/white.jpg" width = "20"/><img src="images/white.jpg" width = "10"/><img src="images/white.jpg" width = "10"/><img src="images/white.jpg" width = "10"/><img src="images/white.jpg" width = "20"/><img src="images/white.jpg" width = "10"/>_Plot 1. Post-PCA Scree Plot_


<img src="images/white.jpg" width = "10"/>

## Results
In the dataset without PCA, on average, K-nearest neighbors, Random Forest, and Support Vector Machine had very similar average accuracies across the 10 folds. However, SVM had the highest spread of accuracies with the best accuray being above 80%. the dataset with PCA had the highest average accuracy with Logistic Regression as the second best by 1%. When running these tests, SVM also took the longest to run compared to the other models. We plotted the average accuracies, along with their standard deviations in the box plots below. The written data represents average accuracy with their standard deviations in parenthesis.

### `No PCA`
- `LR: 0.676577 (0.022594)`
- `RF: 0.742342 (0.030736)`
- `KNN: 0.736937 (0.037966)`
- `SVM: 0.731532 (0.047807)`
- `DTC: 0.652252 (0.062572)`

<img src="images/white.jpg" width = "10"/>

<div>
<img src="images/white.jpg" width = "10"/>
<img src="images/no-pca.png" width = "400"/>
</div>

<img src="images/white.jpg" width = "10"/><img src="images/white.jpg" width = "10"/><img src="images/white.jpg" width = "10"/><img src="images/white.jpg" width = "10"/><img src="images/white.jpg" width = "10"/> _Plot 2. Comparison of 5 Methods without PCA

<img src="images/white.jpg" width = "10"/>

### `PCA with 95% Variance`
- `LR: 0.707207 (0.050200)`
- `RF: 0.639640 (0.049672)`
- `KNN: 0.664865 (0.050450)`
- `SVM: 0.711712 (0.059759)`
- `DTC: 0.618919 (0.020954)`

<img src="images/white.jpg" width = "10"/>

<div>
<img src="images/white.jpg" width = "10"/>
<img src="images/no-pca.png" width = "400"/>
</div>

<img src="images/white.jpg" width = "10"/><img src="images/white.jpg" width = "10"/><img src="images/white.jpg" width = "10"/><img src="images/white.jpg" width = "10"/><img src="images/white.jpg" width = "10"/> _Plot 3. Comparison of 5 Methods with PCA

## Discussion
From these results, we expected methods run on the dataset with PCA to have a higher accuracy level because it would be a more specific dataset with lesser noise. However, as shown in our results, the methods run on the dataset without PCA had better accuracies in general, and there are many reasons as to why something like this might occur. One reason might be that PCA does not take into consideration the prediction target. This would in fact decrease accuracy because while PCA is reducing noise in the data set, it might also remove regions critical to determining the actual prediction target. Also, since PCA is linear, it might hurt its accuracy when applied to images as vectors instead of reducing them to 2 images, rotation, and scale.

# Conclusion

![](images/IsToDo.png)

# Contributions
- Krishna Patel - Supervised Learning Methods, Github pages setup
- Justin Deal - Convolutional Neural Network
- Anthony Marshall - Convolutional Neural Network
- Ben Pooser - Supervised Learning Methods
- Trevor Stanfield - Github pages maintenance

# References
[1] Brown, Ken. “Invasive Ductal Carcinoma (IDC) Breast Cancer: Johns Hopkins Breast Center.” Johns Hopkins Medicine, 3 Nov. 2017, www.hopkinsmedicine.org/breast_center/breast_cancers_other_conditions/invasive_ductal_carcinoma.html.

[2] Cruz-Roa, Angel, et al. “Automatic Detection of Invasive Ductal Carcinoma in Whole Slide Images with Convolutional Neural Networks.” Medical Imaging 2014: Digital Pathology, 2014, doi:10.1117/12.2043872.

[3] Mooney, Paul. :Breast Histopathology Images.” Kaggle, 2018. https://www.kaggle.com/paultimothymooney/breast-histopathology-images

[4] “U.S. Breast Cancer Statistics.” Breastcancer.org, 27 Jan. 2020, www.breastcancer.org/symptoms/understand_bc/statistics.

[5] “Use Case 6: Invasive Ductal Carcinoma (IDC) Segmentation.” Andrew Janowczyk, 5 Jan. 2018, www.andrewjanowczyk.com/use-case-6-invasive-ductal-carcinoma-idc-segmentation/.



<!-- ## Welcome to GitHub Pages

You can use the [editor on GitHub](https://github.com/patelkb99/4641-team31.github.io/edit/master/README.md) to maintain and preview the content for your website in Markdown files.

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.

### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/patelkb99/4641-team31.github.io/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://help.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out. -->
