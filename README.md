This project shows the journey to build a hand pose image classifier. In this project, I used a depth camera made by Intel and mahcine learning library PyTorch to train and build models. The goal is to distribute this project to website that can use users' cameras to capture their hands and classify them.

utils.py
This script contains helper function, such as save_model, save_plot, reset_data_folder, move_files, random_file_move.
save_model is to save the trained model in the local environment.
save_plot is to save the plots containing results in the local environment.
reset_data_folder helps to reset the local environment folder organization related to data. The functions relocate any data in validation or test folders to train folders.
move_files is another helper function to move data from source folder to destination folder.
random_file_move is to move files from train folder to validation and test folders given a certain ratio.

save_iamges.py
This script is to save images taken by Intel RealSense Depth Camera D435 utilizing openCV library.
First, it resets the file environment by running reset_data_folder.
Then, user can take a screenshot from livestream video whenever the user wants.
The user can label the taken image, so it can be automatically saved in organized file structure.

datasets.py
This script is to prepare data sets and data loaders for training.
