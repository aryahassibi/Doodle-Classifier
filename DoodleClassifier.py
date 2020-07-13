from neuralNetwork import *
from random import shuffle, randint
from os import listdir
import numpy as np
import pygame

def prepare_data():
    global data_set, training_set, test_set

    print("creating training set and test set...")

    # adding subsets from doodles to data_set
    # number of the doodles in the data set is too much for us so we just use a pat of it
    for i in range(number_of_categories):
        # margin is a random number that lets us choose a different subset of doodles each time
        margin = randint(0, 50000)

        # dividing data_set by 255 to change the range of the numbers from (0, 255) ot (0, 1)
        data_set.extend((all_of_the_doodles[i][margin:margin + size_of_the_data_set_for_each_category] / 255).tolist())

    # changing data_set elements from:  [array of doodle image], to:  (label of doodle, array of doodle image)
    for i in range(number_of_categories):
        for j in range(size_of_the_data_set_for_each_category):
            index = i * size_of_the_data_set_for_each_category + j
            data_set[index] = (label_of_categories[i], data_set[index])

    # Shuffling data set to train and test model more efficiently
    shuffle(data_set)

    # Dividing data set into training set and test set
    training_set = data_set[:size_of_training_set_for_each_category]
    test_set = data_set[size_of_training_set_for_each_category:size_of_the_data_set_for_each_category]

    # Freeing up memory by clearing data_set list (because we don't need it anymore)
    data_set.clear()


def train_model(training_set):
    print("Training the model...")
    # cycling through the training set and training the model
    for doodle in training_set:
        doodle_label = doodle[0]
        doodle_data = doodle[1]
        nn.train(doodle_data, target[doodle_label])


def test_model(test_set):
    # Tests the model based on the test set,
    # and returns a number between 0 and 1 representing the accuracy of the model
    print("Testing neural network...")
    number_of_correct_answers = 0
    for doodle in test_set:
        doodle_label = doodle[0]
        doodle_data = doodle[1]

        # model classifies each input in test set
        model_prediction = nn.predict(doodle_data)
        result = model_prediction.index(max(model_prediction))

        # if the classification was correct we increase number_of_correct_answers by one
        if result == label_of_categories.index(doodle_label):
            number_of_correct_answers += 1

    # returning model accuracy
    model_accuracy = number_of_correct_answers / size_of_test_set_for_each_category

    return model_accuracy


def is_it_npy(file_name):
    # If type of the file was "npy" it will return true, else false
    if file_name[-4:] == ".npy":
        return True
    else:
        return False


# a list of the names of the files in -data- folder
# and removing files that are not .npy
name_of_files = listdir("data")
name_of_files = [file_name for file_name in name_of_files if is_it_npy(file_name)]

number_of_categories = len(name_of_files)

# removing .npy from the file names and adding them as labels to label_of_categories
label_of_categories = [name_of_files[i][0:-4] for i in range(number_of_categories)]

# training set—a subset to train a model
size_of_training_set_for_each_category = 9000

# test set—a subset to test the trained model
size_of_test_set_for_each_category = 1000

size_of_the_data_set_for_each_category = size_of_training_set_for_each_category + size_of_test_set_for_each_category

# training set and test set are subsets of the data set
# and the data set is a subset of all of the doodles
all_of_the_doodles = []
data_set = []
training_set = []
test_set = []

# target is a dictionary that it's keys are the names of the categories
# and its values are an array that only one element (the corresponding element) is 1 and the other elements are 0.
# for example: {'apple': [1, 0, 0], 'laptop': [0, 1, 0], 'banana': [0, 0, 1]}
target = {}
for label in label_of_categories:
    # assigning 0 to all of the element
    target[label] = [0 for i in range(number_of_categories)]

    # assigning 1 to the corresponding element index
    target[label][label_of_categories.index(label)] = 1

print("loading files...")

for i in range(number_of_categories):
    # checking if it is a numpy .npy file
    if name_of_files[i][-4:] == ".npy":
        # loading all of the drawings
        all_of_the_doodles.append(np.load('data/' + name_of_files[i]))

print("whole data loaded...\n")

# creating a neural network
# number of nodes in the input layer must be 784 because doodles are 28*28
# also the number of the nodes in the output layer must be equal to the number of categories
# but you can change the number of hidden layers and also the number of nodes in each one of them
nn = NeuralNetwork(784, 128, 64, number_of_categories)

# training and testing model n times
for i in range(5):
    # creating a training set and test set
    prepare_data()

    # training model
    train_model(training_set)

    # testing model
    model_accuracy = test_model(test_set)
    print("Neural Network Accuracy:", (model_accuracy * 100), "%\n")

# deleting all the data sets because we have trained the model and we don't need them anymore
# and it's better to delete them to free up memory
all_of_the_doodles.clear()
training_set.clear()
test_set.clear()
target.clear()

# ---- Creating Canvas to Draw on it ----

pygame.init()

# height of the red box at the bottom of the screen
text_box_height = 50

# width and height of the screen
width = 500
height = 500 + text_box_height

# width and height of the canvas
# canvas is the part of the screen that you can draw on it
canvas_width = width
canvas_height = height - text_box_height

# creating the screen
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Canvas")

# a boolean to keep track of the mouse button
mouse_button_is_down = False

# This list is used to store the points of the lines drawn on the screen
# this list consists of multiple lists
# in each one of the lists inside there are tons of point coordinate, something like this:
# [  [(x1, y1), (x2, y2), (x3, y3), (x4, y4)], [(x1, y1), (x2, y2)]  ]
points_of_lines = []

# creating the text of the clear button
clear_button_text = pygame.font.Font(None, 20).render('clear', True, (255, 255, 255))

# width and height of the button
clear_button_width = text_box_height
clear_button_height = text_box_height

# colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
LIGHT_RED = (255, 100, 100)

done = False

while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
            # canvas = screen.subsurface((0, 0, width, width))
            # canvas = pygame.transform.scale(canvas, (28, 28))
            # pygame.image.save(canvas, "doodle.png")

        if event.type == pygame.MOUSEBUTTONDOWN:
            mouse_button_is_down = True

            # creating an empty list inside of points_of_lines for the new line that the user is going to draw
            points_of_lines.append([])

        if event.type == pygame.MOUSEBUTTONUP:
            mouse_button_is_down = False

            # checking if clear button is pressed
            x, y = pygame.mouse.get_pos()
            if x < clear_button_width and canvas_height < y < height:
                # clearing list to clear the screen
                points_of_lines = []

    screen.fill(BLACK)

    # adding coordinate of mouse cursor while mouse button is down
    if mouse_button_is_down:
        x, y = pygame.mouse.get_pos()

        # making sure you're only drawing in the canvas
        if y < canvas_height:
            points_of_lines[-1].append((x, y))

    # showing drawn lines on the screen
    for each_line in points_of_lines:

        # making sure there is something to show and the list is not empty
        if len(each_line) > 1:
            pygame.draw.lines(screen, WHITE, False, each_line, width // 20)

    # cropping screen and creating canvas ( because we don't need the text box at the bottom )
    canvas = screen.subsurface((0, 0, canvas_width, canvas_height))

    # resizing canvas from width, height to 28, 28 (because our neural network has been trained based on 28*28 doodles
    canvas = pygame.transform.scale(canvas, (28, 28))

    # transforming the screen into a numpy array
    canvas = pygame.surfarray.array3d(canvas)

    # canvas is a 3d array (width, height, rgb)
    # but our doodles are black and white so we just need a 2d array (width, height) so we use mean()
    # also we have to divide each element by 255 to get a number between 0 and 1 as each pixel color
    canvas = np.mean(canvas, axis=2) / 255

    # for some reason the axis of the array that pygame gives us are swapped so we have to swap it again
    canvas = canvas.swapaxes(0, 1)

    # reshaping (28, 28) array into (784, ) array so we can give it to the model
    INPUT = canvas.reshape((784,))
    model_prediction = nn.predict(INPUT)

    # choosing the category with the highest number as the model prediction
    model_prediction = model_prediction.index(max(model_prediction))
    model_prediction = label_of_categories[model_prediction]

    # creating text box and choosing a position for it
    predicted_category_text = pygame.font.Font(None, 64).render(str(model_prediction), True, (255, 255, 255))
    textRect = predicted_category_text.get_rect()
    textRect.center = (width // 2, height - (text_box_height / 2))

    # red text box
    pygame.draw.rect(screen, RED, [0, width, width, text_box_height])

    # light red clear button background
    pygame.draw.rect(screen, LIGHT_RED, [0, width, clear_button_width, clear_button_height])

    # showing texts on the screen
    screen.blit(clear_button_text, (10, width + 17))
    screen.blit(predicted_category_text, textRect)

    # updating the screen
    pygame.display.flip()

pygame.quit()
