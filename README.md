# Doodle Classifier
A Doodle Classifier written in 
[python](https://www.python.org)(3.7.6), using 
[NumPy](https://numpy.org)(1.18.1), 
[pygame](https://www.pygame.org)(1.9.6) and 
[Google Quick Draw data set](https://quickdraw.withgoogle.com/data).

I used my own [Multilayer Neural Netwrok](https://github.com/aryahassibi/Multilayer-Neural-Network#multilayer-neural-network) 
to create this project, you can read about it on its page.

![DoodleClassifierGIF](https://user-images.githubusercontent.com/31355913/87329602-73404680-c54c-11ea-9e82-06057b6a873f.gif)

## *How To Use DoodleClassifier.py*
After downloading the file create a folder named ***data** at the same place you are keeping DoodleClassifier.py.
Then you have to download two or more numpy bitmap files from [Google Quick Draw data set](https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/numpy_bitmap) 
> *note:* you can search the name of the category that you are interested in the search bar at the top of the page.


> For more information about the Google Quick Draw data set, check out [this page](https://github.com/googlecreativelab/quickdraw-dataset).

Now that you have downloaded ***.npy*** files that you wanted, you have to put them in ***data*** folder.
> *note:* names of the categories in the program are based on the name of the files, so it's better not to change the name of the files.

After doing all of this, you are ready to run the program and enjoy it.

## *How Does DoodleClassifier.py work?*
When you run the program it starts loading the data set and then trains the neural network(this step may take a few seconds). <br>
After training the neural network it shows you the canvas and you can draw whatever you like on the canvas, then the neural network makes a guess based on the thing that you have drawn and you can see its prediction in the red text box at the bottom of the screen.<br>
If you want to clear the canvas you can press the ***clear*** button in the bottom left corner of the screen.

> *note:* For further details read the comments in ***DoodleClassifier.py***.
    

