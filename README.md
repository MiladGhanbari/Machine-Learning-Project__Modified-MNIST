# Modified-MNIST
In this project, the problem of finding the largest digit in an image is tackled using several machine learning approaches. The provided dataset is a modified version of the MNIST dataset. There are multiple digits in each image, and these digits have been randomly rotated, resized, and stuck onto a randomly generated background. In the first attempt, the images were preprocessed in order to extract the largest digit from the image and to transform the modified MNIST problem to the typical MNIST problem. This report discusses and compares various approaches of image classification that fall within the machine learning paradigm such as Logistic Regression (LR), Support Vector Machines (SVM), Convolutional Neural Networks (CNNs), and Ensemble Method. The optimum values for each of the hyperparameters related to the classifiers were obtained by using a grid search approach. Furthermore, some very deep models such as VggNet, ResNet, and Xception were experimented on the provided data. K-fold cross-validation method was performed on the training data to achieve a reliable and robust model. In addition, dropout and data augmentation methods were utilized to prevent deep learning approaches from overfitting. Finally, through performing extensive experiments on the mentioned classifiers, the best model resulted in the highest performance was the ensemble of 10 Xception models with 98.433 % accuracy on the test data.


# Instructions:
Below is the instructions in order to obtain the result shown in the report. 

* Linear Regression:
First put train and test datasets in the same directory indicated as "Data_Path". Then run "Linear Regression.py" code to see the results.

* SVM:
First put train and test datasets in the same directory indicated as "Data_Path". Then run  "SVM.py" code to see the results.


* CNN:
First put train and test datasets in the same directory indicated as "Data_Path". Then run "CNN_ensemble_15.py" code to see the results.


* Xception:

	a. Put the train dataset, test dataset and labels in the directory where .py files exist.
	
	b. Run training_Xception.py for training the model (or run the training_Xception_ensemble_20.py for the 20 ensemble method)
	
	c. Run Xception_result.py for predicting the test dataset (or run the Xception_result_ensemble_20.py for the 20 ensemble method) 
