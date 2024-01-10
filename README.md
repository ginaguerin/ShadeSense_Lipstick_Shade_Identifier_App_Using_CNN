# <center> <b> <span style="color: hotpink;">  ShadeSense </center> #
<center>

![](https://github.com/ginaguerin/ShadeSense_Lipstick_Shade_Identifier_App/blob/master/logos/logo3.2.jpeg?raw=true)
</center>

## <u> Concept: </u> ##

- ShadeSense is an innovative application designed to instantly identify the specific shade of lipstick someone is wearing in real-time as well as through uploading an image. The ultimate goal of this application is to create an extensive library capable of recognizing a wide range of lipstick shades and brands. Importantly, ShadeSense aims to be inclusive, ensuring accurate detection across all lipstick colors, irrespective of gender or skin tone.

## <u> Scope: </u> ##

- In its initial phase, ShadeSense will begin with a curated selection from the renowned makeup brand, MAC Cosmetics. This curated collection consists of five distinct lipstick shades, serving as a starting point for the app's development. As ShadeSense evolves, it will expand its library to encompass a diverse array of lipstick shades from various brands, further enhancing its capabilities.

## Dataset Overview: ##

- **Images:** The dataset comprises original resized images in JPG format. These images are the input data for training the lipstick shade identification model.

- **Annotations:** Each image is associated with an annotated XML file. These XML files likely contain information about the location and class labels of the lipstick shades within the corresponding image.

- **Classes:** There are 7 distinct classes corresponding to different lipstick shades. The goal is to train the model to recognize and classify these lipstick shades automatically.

- **Supervised Learning:** The dataset is suitable for supervised learning, where the model learns from the paired examples of images and their corresponding annotations to generalize and make predictions on new, unseen data.

- **Training Objective:** The objective of the model is to analyze the images and identify the correct lipstick shade class based on the provided annotations.

- **Neural Network Architecture:** The CNN architecture outlined in the baseline model is designed to process the image data and make predictions for the 7 lipstick shade classes.

## <u> <span style="color: red;"> Limitations: </span> </u> ##


1. **Limited Dataset Size:**
   - With only 220 images for the six lipstick shades, including images of no lipstick, the dataset might be relatively small for training a highly accurate and robust neural network.
2. **Model Generalization:**
   - The model's ability to generalize to different lipstick shades, brands, and skin tones might be limited initially. Training on a diverse dataset can help improve generalization.

3. **Brand and Shade Specificity:**
   - The initial focus on MAC Cosmetics and six specific shades may limit the app's applicability to users with different preferences or those using other brands. Expanding the dataset to include various brands and shades can address this limitation over time.

4. **Real-Time Processing Constraints:**
   - Real-time processing on live camera feeds can be computationally intensive, leading to potential performance limitations, especially on devices with lower processing power. Optimizing the model and deploying it on platforms that support efficient real-time processing.

5. **Lighting and Environmental Conditions:**
   - The accuracy of the lipstick detection may be influenced by lighting conditions and the environment. Variations in lighting may impact color perception, potentially affecting the model's performance.

6. **User Privacy Concerns:**
   - The app involves capturing and processing images in real-time, raising privacy concerns. Making sure to clearly communicate to users how their data will be handled, stored, and if any images are stored temporarily for processing.

7. **Device Compatibility:**
   - The app's real-time detection capabilities may be influenced by the camera quality and specifications of the user's device. Ensure compatibility across a range of devices and optimize the app's performance accordingly.

8. **Legal and Ethical Considerations:**
   - Ensure compliance with legal and ethical standards, especially when dealing with image data. Being aware of privacy laws, data protection regulations, and obtain consent when necessary.

9. **Feedback and Iterative Development:**
   - Given the initial focus on a smaller dataset and makeup brand, user feedback will be crucial for identifying limitations and areas for improvement. Planning for iterative development to enhance the app based on user experiences and preferences.

## Baseline Model ##

 The baseline model is meticulously crafted for the lipstick shade identification task using a dataset comprising original resized JPG images and their corresponding annotated XML files. Here are the key aspects of the baseline model, with a focus on monitoring the F-1 score due to our small dataset, imbalanced classes, and concerns for misclassifying categories:

- **Model Architecture:** A Convolutional Neural Network (CNN) is tailored for image processing, featuring three convolutional layers (32, 64, and 128 filters), subsequent max-pooling layers, a flatten layer, and two dense layers. The model is specifically designed to process images with dimensions (512, 512, 3).

- **Dataset Structure:** Our dataset consists of paired examples, linking original resized images to annotated XML files. Each image is enriched with details about the location and class labels of various lipstick shades.

- **Supervised Learning:** The model adopts a supervised learning approach, learning to make predictions based on annotated images, with the objective of automatically categorizing lipstick shades into 7 distinct classes.

- **Training Objective:** The primary aim is to train the model for accurate recognition and classification of lipstick shades. Ground truth information from annotated XML files serves as the foundation for model training.

- **Model Compilation:** The model is compiled using the Adam optimizer, sparse categorical crossentropy loss function, and includes F-1 score as an additional evaluation metric. This modification aligns with our focus on addressing the challenges posed by our small dataset and imbalanced classes.

- **Training Process:** The model undergoes training for 10 epochs using the paired dataset, optimizing parameters to minimize crossentropy loss and maximize the F-1 score. The validation dataset plays a crucial role in evaluating the model's generalization to unseen data.

## Summary of Enhanced Model Performance ##

### Model Architecture Enhancements: ###
The enhanced model introduced several architectural modifications, including dropout layers and L2 regularization, to address overfitting concerns. However, the model's performance in terms of accuracy and key metrics did not show significant improvement.

### Training and Evaluation Results: ###
The training and validation results over 10 epochs for the enhanced model are outlined below:

- **Epoch 1/10:**
  - Training Loss: 3419.66, Training Accuracy: 14.29%
  - Validation Loss: 32.64, Validation Accuracy: 15.91%

- **Epoch 10/10:**
  - Training Loss: 20.5985, Training Accuracy: 90.29%
  - Validation Loss: 21.3792, Validation Accuracy: 68.18%

### Performance Metrics: ###
The classification report provides insights into the model's capability to classify lipstick shade categories. Key observations include:

- **Classification Report:**
  - "Crème D'Nude" and "None" show high precision, recall, and F-1 scores, indicating strong performance. 
  - "Honey Love" and "Whirl" face challenges, with lower F-1 scores, suggesting potential areas for improvement.
  - Overall accuracy is 68.18%, indicating some improvements in correctly classifying lipstick shades.


## Summary of Final Model with Lowered Dropout Rate ##

### Model Architecture and Training Dynamics: ###
The model, incorporating VGG16 with a lowered dropout rate of 0.38, demonstrates improved stability and generalization during an 8-epoch training regimen:

## Interpretation of Model Results ##

- **Epoch 1/8:**
  - Training Loss: 9.11, Training Accuracy: 18.86%
  - Validation Loss: 3.43, Validation Accuracy: 38.64%

- **Epoch 8/8:**
  - Training Loss: 0.43, Training Accuracy: 81.82%
  - Validation Loss: 0.43, Validation Accuracy: 81.82%

### Test Set Performance: ###
The model demonstrates promising results on the test set, emphasizing robustness and generalization.

- **Test Loss:** 0.43, **Test Accuracy:** 81.82%

### Classification Report: ###
The classification report provides a detailed breakdown of the model's performance across makeup product categories.

- **Key Observations:**
  - Precision, recall, and F1-score metrics exhibit improvements, especially for classes like "Honey Love," "Ruby Woo," and "Whirl."
  - The overall accuracy is 81.82%, highlighting the model's capability to correctly classify makeup products.

### Conclusion: ###
The new model results reflect substantial enhancements, emphasizing improved accuracy and class-specific metrics. This suggests that the model has successfully learned intricate patterns within the data, showcasing its potential for accurate lipstick shade.


## Conclusion and Future Steps ##

The culmination of our model tuning efforts has resulted in a highly promising Convolutional Neural Network (CNN) based on VGG16. Leveraging strategies such as L2 regularization, dropout layers, and a custom learning rate, our model exhibits exceptional performance with a test accuracy of 81.82%. The classification report further emphasizes improved precision, recall, and F1-score metrics across various lipstick shades.

Moving forward, we plan to integrate this finalized model into our ShadeSense Lipstick Shade Identifier App. Real-world usage will provide valuable insights into the model's practical performance, allowing us to analyze its strengths and identify areas for further refinement. Additionally, we remain committed to continuous improvement by fine-tuning the model, expanding our dataset, and exploring opportunities to enhance its capabilities.

Excitement surrounds the prospect of observing the model's progression in real-world scenarios and its adaptability to a diverse range of user-generated inputs. This iterative approach ensures that our app evolves with user interactions, delivering accurate and reliable results for lipstick shade identification. We look forward to the app's continued development, with a keen eye on user feedback and data augmentation to propel its performance to new heights.


![](https://github.com/ginaguerin/ShadeSense_Lipstick_Shade_Identifier_App/blob/master/logos/4girls.jpeg?raw=true)

# <b> <span style="color: hotpink;"> Repository Directory #

```bash
├── Slides                  <- Project Powerpoint Presentation 
├── Images                  <- Original & Annotatted images used in this project
├── Logos                   <- Logos created using DALL·E 3
├── .gitignore              <- Checkpoints & Models
├── Master Notebook         <- Notebook containing model tuning and analysis
├── README.md               <- Contains README file
└── ShadeSense.py           <- Creation of CNN application

```