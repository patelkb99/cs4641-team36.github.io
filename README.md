### Krishna Patel, Justin Deal, Anthony Marshall, Ben Pooser, Trevor Stanfield

# MOTIVATION OF PROJECT

Many different types of breast cancer show relatively few signs and symptoms, and the inconspicuous nature of this disease only exacerbates the danger. 1 in 8 women in the United States will develop breast cancer in her life. In 2020, it is predicted that about 42,000 women in the United States are expected to die from breast cancer [5]. 

Invasive ductal carcinoma (IDC) is the most common form of breast cancer, representing over 80% of all cases [1]. In order to identify the type of cancer and its invasiveness, pathologists scan large sections of benign regions in order to identify malignant areas; this process can be time consuming and difficult [2]. Therefore, for our project, we will extract features from a dataset of breast histology images. After performing feature extraction we will use binary classification to determine the presence of IDC [4]. 

# DATASET AND PREPROCESSING

## Dataset
Our dataset is a Kaggle dataset provided by Andrew Janowczyk. It consists of 5547 breast histology images [4].

Images in this dataset are of size 50x50x3. Each file name follows this format, which reveals if the image is cancerous images or not: u_xX_yY_classC.png => example 10253_idx5_x1351_y1101_class0.png.

- u : the patient ID (i.e. 10253_idx5)
- X : the x-coordinate of potential IDC (i.e. x1351)
- Y : the y-coordinate of potential IDC (i.e. y1101)
- C : the class where 0 is non-IDC and 1 is IDC [6]

### Example of the images:

![Examples of Dataset](/images/datasetExample.png)
Format: ![Alt Text](url)

## Data Preprocessing

Add information on data preprocessing

## Convolutional Neural Network

# Methods
## K-Nearest Neighbors
## Neural Networks
## Support Vector Machines

# Conclusion

# References
[1] Brown, Ken. “Invasive Ductal Carcinoma (IDC) Breast Cancer: Johns Hopkins Breast Center.” Johns Hopkins Medicine, 3 Nov. 2017, www.hopkinsmedicine.org/breast_center/breast_cancers_other_conditions/invasive_ductal_carcinoma.html.

[2] Cruz-Roa, Angel, et al. “Automatic Detection of Invasive Ductal Carcinoma in Whole Slide Images with Convolutional Neural Networks.” Medical Imaging 2014: Digital Pathology, 2014, doi:10.1117/12.2043872.

[3] Janowczyk, Andrew, and Anant Madabhushi. “Deep learning for digital pathology image analysis: A comprehensive tutorial with selected use cases.” Journal of pathology informatics vol. 7 29. 2016, http://doi.org/10.4103/2153-3539.186902.

[4] Janowczyk , Andrew. “Breast Histology Images.” Kaggle, 2017, www.kaggle.com/simjeg/lymphoma-subtype-classification-fl-vs-cll.

[5] “U.S. Breast Cancer Statistics.” Breastcancer.org, 27 Jan. 2020, www.breastcancer.org/symptoms/understand_bc/statistics.

[6] “Use Case 6: Invasive Ductal Carcinoma (IDC) Segmentation.” Andrew Janowczyk, 5 Jan. 2018, www.andrewjanowczyk.com/use-case-6-invasive-ductal-carcinoma-idc-segmentation/.


## Welcome to GitHub Pages

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

Having trouble with Pages? Check out our [documentation](https://help.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.
