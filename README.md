# Face Recognition
This is a face recognition program. Its not very heavy, and can run on a CPU, albeit with some lag.
Place your face's tight cropped images (a few) in './face_recognition/my_train_images'.
It doesn't use any pretrained model to generate embeddings and has been built from scratch, hence wont perform well if lighting 
isnt very good. And with large changes in illumination, you will need to upload new images.

After uploading your images, run the './deploy.py' script.

# Samples:

![Alt text](https://github.com/sharan-dce/Face-Recognition/blob/master/face_detect.gif)


#For training the models used yourself:
You need to use the preprocessing files over the FDDB dataset, creating bitmap images to map to, for segmentation, resize the original images and the bitmaps to 64 x 64, for the neural net and save the 2 numpy arrays as 'images.npy' and 'outputs.npy' in the './face_detection' directory.
Also, this dataset: http://conradsanderson.id.au/lfwcrop/ must be downloaded and the images must be placed in 'lfwcrop_color/faces'.
Then simply run the './face_detection/train_segmentation.py' script to train the model. The version that gives the best test results will automatically be saved.
Train the face embedding generator by running the './face_recognition/trainer_embedding_generator.py'. The best version will be autosaved.
You need to place images which have to be detected as positive (basically your images) in './face_recognition/my_train_images'. They should be tight cropped, with just your face in view.
