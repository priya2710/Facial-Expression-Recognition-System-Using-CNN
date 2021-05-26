# Facial-Expression-Recognition-System-Using-CNN

![5](https://user-images.githubusercontent.com/52822987/119609680-9e0a5500-be15-11eb-9d30-a4ab7696cf03.JPG)

#About

Facial expressions are the visible manifestation of the affective state,cognitive activity, intention, personality and psychopathology of a person and plays a communicative role in interpersonal relations. Automatic recognition of facial expressions can be an important component of natural human-machine interfaces; it may also be used in behavioural science and in clinical practice. An automatic Facial Expression Recognition system needs to perform detection and location of faces in a cluttered scene, facialfeature extraction, and facial expression classification. Facial expression recognition system is implemented using Convolution Neural Network (CNN). CNN model of the project is based on LeNet Architecture. Kaggle facial expression dataset with seven facial expression labels as happy, sad, surprise, fear, anger, disgust, and neutral is used in this project.

#Summary

1. Confmatrix.py : Generates confusion matrix  using two files truey.npy and predy.npy
2. Detect Emoticon (In recorded video): Take video file as input (here detect_video_emoticon.mp4 is used) and generate another video file with recognized expression.
3. fer2013.csv : Spreadsheet containing the database , used for training, testing and validating.
4. haarcascade_frontal_face.xml : predeveloped XML file, used here to detect front faces from images.
5. Image_from_camera.py : Take picture from web camera and generate prediction.
6. model_weights.h5 : Contains weights of trained neural network.
7. model.json : Contains all the information about our model in JavaScript Object Notation format.
8. model(train).py : Used for training the model using database, generates modXtest.npy and modytest.npy used for testing.
9. model(test).py : Used for testing the model, generates truey.npy and predy.npy used for confusion matrix generation.
10. preprocessing.py : Used to preprocess the dataset.
11. Recognise Image.py : Use preloaded picture (test.jpg) and generate prediction (output_test.jpg).
12. Recognise Video Emoticon (Live WebCam) : Uses live web-camera to predict the expression.

Since the size of some of the file are too large so, I am hereby attaching the google drive link : https://drive.google.com/drive/folders/1pDJr9YtH4uI5sxnHZ-0JbozDjL8mk6RZ?usp=sharing

#Implementation

1. Download the repository and extract all the files.
2. After this, download the dataset from the given google drive link, copy this file on your working directory.
3. Also, copy all the downloaded repository files to your working directory.
4. Now, run the file "preprocessing.py" which will produce two numpy files "fdataX.npy" and "flabels.npy".
5. Its time to train the model, so run "model(train).py".
6. Now, run "model(test).py".
7. Now, our model is ready for prediction.
8. So, run "Recognise Image.py","Recognise Video Emoticon(Live WebCam)","Image_from_camera.py", depending upon our needs.
9. We can also draw "Confmatrix.py" for confusion matrix.


#Output

1. Dataset Distribution

![1](https://user-images.githubusercontent.com/52822987/119609457-3a802780-be15-11eb-95d4-7e713701e0d4.JPG)

2. Confusion Matrix

![2](https://user-images.githubusercontent.com/52822987/119609515-57b4f600-be15-11eb-9d22-cde9d1ab1786.JPG)

3. Image detection through live Web Camera

![3](https://user-images.githubusercontent.com/52822987/119609550-68fe0280-be15-11eb-8e8b-5ee11a25db88.JPG)

4. Image Detection through clicking image (using web camera)

![4](https://user-images.githubusercontent.com/52822987/119609609-7d41ff80-be15-11eb-9c56-51ebeaa8222e.JPG)




