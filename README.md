# AI-neural-networks
Artificial intelligence: networking, training, performance verification, data preparation Convolutional deep network CNN - digit recognition deep network GoogLeNet - image recognition
## 1
Operations on a shallow feedforward neural network
Creation of a simple neural network which, based on the instantaneous values of the cylinder volume and the temperature in it, will provide the pressure of the fuel-air mixture in the cylinder. The data that will be used to train and test the network concerns the diesel engine. Neural network operations performed using commands from the Deep Learning Toolbox.
The operation of a diesel engine is described by the theoretical Seiliger-Sabathé cycle. It illustrates the relationship between the instantaneous volume of the cylinder (V) and the pressure (p) that prevails in it. In addition to these two values, the temperature (T) of the fuel-air mixture in the cylinder also changes.

#### Data was used to train and test the network
* silnik_ucz.txt – data for training the network, selected from the full set
* silnik_full.txt – full set of data for the engine in question
Both files contain 4 columns of data: moments of time, volume, pressure and temperature. The file silnik_ucz.txt contains 1143 samples of these variables, and silnik_full.txt - 8000.
#### Importing data from files into the workspace:
* silnikF = load('silnik_full.txt');`
* silnik_Ucz = load('silnik_ucz.txt');
Presentation of a graph of data from the silnik_full.txt file
The known values are to be volume and temperature, and the resulting pressure, the network will have two inputs and one output.
To do this, you need to separate the data into input and output.
Then they are placed in matrices so that subsequent sets for one moment of time are placed in subsequent columns of the matrix.
After import, the data from the text files was placed in the following lines of the silnik_f and silnik_ucz variables. They should therefore be additionally transposed.
#### Datasets:
* SUWej – input training data,
* SUWyj – training data constituting output patterns,
* STestWej – test input data
#### Creating a network:
Tool (Neural Network Tool) – command:nntool
Good for shallow neural networks
Window:
* Input Data – input data for the network
* Target Data – output data for the network (output patterns)
Networks – xxi neural networks created
Output Data - Output data obtained from the operation of the network (actual results - not benchmarks)
Error Data – data regarding network errors
Variables created in the workspace (suczwe, suczwy, stestwe) are imported to the nntool tool (import button). After importing, you can close the tool window.
suczwe, stestwe – input data, suczuwy – target data
#### Creating a neural network:
* “New” button – network parameters window
* selection of the type, e.g. feed-forward backprop
* a classic multi-layer feed forward network trained with algorithms with back propagation of error.
The network is trained using the Lavenberg-Marquard method (TRAINLM. Adaptation learning function
* gradient method with momentum (LEARNGDM), performance function
* least squares method (MSE), network layers adopted 3, numbers of neurons was assumed 12, TANSIG neuron type - with hyperbolic tangent activation function. For the last layer (3), the number of neurons is not set, it results from the size of the output data. In the (vew) view you can see the diagram. If you have a network with random weights, you should teach her.
#### Teaching:
Network selection in nntool. Select by double-clicking or the open button and going to the train, training info tab.
Half complement: 
- training data
- impulse – suczwe variable
Template data target field – target variable
You can additionally set: the number of epochs, the maximum training duration.
In the current version, the number of inputs and outputs of the network is equal to 0. In the next steps, it will be automatically adjusted to the number of training data.

After creating the network, we can start training it. First, you need to define the division of the training data into actual training data, validation data and test data, e.g. in a ratio of 70:15:15%.
Network training is performed by the train() function, whose parameters are: network name, sets of training input and output data:

#### Start of learning: train Network
Performance – mean squared error of training, testing and validating the Network

Individual data sets are automatically created from the training sequence specified in the network learning parameters. In this case, data from the SUWwej and SUWwyj variables were randomly allocated to individual sets in the ratio 70:15:15%. It is also possible to select another method among the network settings.
* Training State - the rate of change of network weights and the number of validations performed after which network overfitting was detected
* Error Histogram - an error histogram showing how learning errors are distributed
* Regression- correlation between network inputs and outputs for training, test and validation data. The more the points on the graph coincide with a straight line inclined by 45°, the better the network is trained.

Based on the above, we obtain a neural network for determining the pressure in the cylinder based on volume and temperature.
Checking the correct operation: use the view() command to check whether the sizes of the network inputs and outputs have been adjusted to the sizes of the training data.

#### Operation verification:
To check the operation of the network, we will use the full set of data for the considered engine.
To use a network, simply provide its name along with the name of the input data set. For the network and STestWej data we created, it will be: STestWyj=net(STestWej);
Checking the compliance of the network operation results with the reference ones on the chart. First, we will plot the benchmark data.

Open the nntool window –> export –> variable containing the simulation results "network1_outputs -> export

Observation: the consistency of the network performance results with the reference data can be considered. In case of discrepancies, restart the network training. It is worth starting training with new starting weights. "Reinitialize Weights" tab -> click "initialize weights".
In a similar way, you can create other networks that differ in the number and type of neurons in individual layers

# 2.NARX network (nonlinear autoregressive neural network with external input)
A network implementing an autoregressive model of a data sequence, taking into account external forces.
#### Usage:
to predict future signal values based on observations of its current values.
A network has been created that will reproduce the behavior of a certain physical object. In order to obtain the data needed to identify the object, an excitation signal was supplied to its input in the form of white noise lasting 1000 s, sampled at a frequency of 100 Hz. At the same time, the object's response was observed.
#### Activation:
1. Load the data file into the workspace.
The file with the data that will be used to train the network contains three variables:
* MPobUcz – excitation (noise) signal used to identify the object,
* MOdpUcz – the object's response to the excitation signal,
* MCzasUcz – moments of time in which the above signals were recorded
  
A characteristic feature of the network used is the feeding of several consecutive samples of the excitation signal to the input and several subsequent samples of the output excitation signal (network feedback

2.Creating an empty file with the name e.g. NARXnet_1.m.

3.preliminary preparation of data for training the network.

4.The data is not in the appropriate form, i.e. a cellular matrix, required by further functions that we will use to build and train the NARX network.
#### Creating variables:
* X – training signal for the network
* T – (targets) – signal of network output patterns. (Note – the name T will not stand for time!).
Network settings and selection of its training algorithm.
Assumption: the network will accept 2 samples of the excitation and response signals (maxDelay variable). The number of excitation and response samples constituting input signals to the network may be different (inputDelays and feedbackDelays variables). The hiddenLayerSize variable specifies the number of neurons in the hidden layer.
# 22
The NARX network being built will be a single-layer network.
It is possible to create a multi-layer one, in which case this variable should contain a vector describing the number of neurons in individual layers. The trainFcn variable describes the selected network training algorithm. In this case, the Levenberg-Marquardt algorithm was used. The narxnet() function creates a NARX network.

The 'open' parameter specifies that the network is open, i.e. without feedback.
At the learning stage, instead of the actual output signal from the network, samples of the reference signal are fed to the inputs that will later receive this signal (i.e. the feedback signal). This improves and speeds up network training. The created network will be saved as an object in the net variable. The next step is to set the student parameters.

Assumption(max): 5000 epochs (the standard value of net.trainParam.epochs = 1000 epochs may be too small in this case). Additionally, you must specify how much data will be allocated to training (net.divideParam.trainRatio), validation (valRatio) and testing (testRatio) data.
In the case of a NARX network, it is necessary to additionally prepare the input data by introducing a delay depending on the number of network inputs and determining the initial states for the feedback signals using the preparets() function.

The perform() function calculates an indicator describing the quality (error) of the network. The lower its value, the lower the error. By default, the indicator is the mean squared error, but you can choose a different one by changing the net.performFcn parameter.
The created network was trained with feedback replaced by the provision of reference data.
It is then transformed using the closeloop() function, which modifies the network by closing the feedback from the network's output signal.

# Performance verification 33
Network operation test using signals stored on variables:
* MPobTest – excitation signals,
* MOdpTest – the object's response to the corresponding excitation signal from MPobTest,
* MCzasTest – moments of time in which the above signals were recorded.
  
The MPobTest and MOdpTest variables contain 15 sets of object excitations and responses in the following columns.
Creating an empty m-file and saving it, giving it a name e.g. NARXnet_2.m.
Dividing the task into two m-files will allow you to train the network separately and then, without retraining, test its operation for different signals.

To test the network, select one of the excitation signals and prepare the data using the preparets() function. This function requires an input and reference signal. The MOdpTest variable contains the system's responses, which are used as comparative data, i.e. when testing, we will assume that we only provide an excitation signal to it. The feedback signal will be the actual response of the network (it will not be taken from a previously recorded data waveform, as was previously the case during training). Therefore, the Y signal required by the preparets() function is declared by padding its place with zeros. After preparing the data, the network will be run to determine the response of the object simulated by the neural network.

The script contains a set variable („zestaw”). It allows you to easily select one of the excitation signals contained in the MPobTest variable and the corresponding response of the real object from the MOdpTest variable. In turn, the variable dt is the sampling period of the above-mentioned. signals.

After simulating the network, you can view the excitation signal on the graphs and compare the response of the object simulated by the network and the real one.

The network exit graph should be shifted by the time resulting from the delay introduced by the network.
The network responses to various excitation signals are included in the MPobTest variable.
Each signal was sampled at a frequency of 100 Hz and lasted 5 s.
In addition to testing the prepared network, you can also check how training will proceed and what the results will be for:
* networks with a different number of neurons
* networks with a different number of inputs (delays)
* networks with more than one hidden layer

# 4 Using a convolutional deep network (CNN) for digit recognition
Ready-made sample data sets provided with Matlab and derived from the MNIST data set were used to train and test the Network
(http://yann.lecun.com/exdb/mnist/)

1. Creating an empty m-file and saving it, giving it a name e.g. digits_CNN_1.m.
2. Creating a data set from available digit images, grouped in subdirectories named "0", "1", etc. up to "9".
The collection is very large (10,000 images) and loading it in its entirety into the computer's memory is not a good solution. You can use the imageDatastore() function instead. It creates an object that groups data and makes it available to further functions related to neural networks, including: reading only those data files that are needed at a given moment. This allows, among other things, save working memory. Moreover, the function automatically organizes the data and, for example, assigns them labels needed for identification and in the network training process. The sample data used is organized in subdirectories, and their names are used as labels (categories) for the data.

The fullfile() function used creates the full path to the directory with sample data based on the names of subsequent subdirectories.
Once you've created your collection, it's worth taking a look at a few randomly selected images in it.
You can also check the size of a single image (figure drawing), what labels have been identified by the imageDatastore() function and how many images from each category are in the data set.

The next step is to divide the data set into training and validation data. Please remember that you should select the same number of images from each type of digits. It was assumed that 750 patterns for each digit would be randomly selected for training.
#### Creating and training a network
Preparing a neural network begins with defining its layers. Design: 15 layers.
#### The network consists of the following types of layers:
* Image Input Layer – input layer collecting images with dimensions of 28x28 pixels and 1 color channel (grayscale image)
* Convolutional Layer – convolutional layer, which performs the operation of convolution (filtering) the image with filters of size 3x3 pixels. The second layer parameter is the number of filters (i.e. neurons). In the example in question, e.g. 8 filters mean the extraction of eight features from the image. The Padding parameter and the value themselves determine how the pixels on the image border are to be processed. With the adopted parameters, the image size after filtration will be the same as before filtration.
* Batch Normalization Layer – normalization layer, normalizing the values of activation signals and gradients, which affects the speed of network training.
* ReLU Layer – layer implementing a non-linear operation: 𝑓(𝑥)={𝑥 for 𝑥>00 for 𝑥≤0
* Max Pooling Layer – a layer that allows you to reduce the image size (after filtering in the convolutional layer) by selecting one maximum value from subsequent image regions. In the example considered, the window has a size of 2x2 pixels and is moved in subsequent steps by 2 pixels.
* Fully Connected Layer – a full layer, i.e. a "classic" layer of neurons with a full set of connections to the previous layer. The task of this layer is to combine all image features recognized by previous layers. In the example under consideration, the size of layer (10) results from the number of categories to which the result of image recognition by the network can be assigned.
* Softmax Layer – a layer that normalizes the outputs from the full layer, thanks to which the values on individual outputs can be treated as an indication of the degree of recognition of the belonging of the examined image to particular categories.
* Classification Layer – classification layer, i.e. the output layer returning the result of image recognition by the network.
After defining the network structure, you need to define its training parameters, e.g.: learning algorithm (SGDM - Stochastic Gradient Descent with Momentum), the number of epochs (4), what data set is to be used for validation and how often it is to be carried out (every 30 iterations). Then you need to train the network.
##### Run the prepared m-file.
The upper graph shows the increase in image recognition accuracy and the lower graph shows the decrease in the loss function (training error).
At this stage, we have a trained neural network. In addition to defining the network structure in the programmatic way shown, you can also use deepNetworkDesigner. In the Matlab command window, enter:
deepNetworkDesigner

In the window that opens, select the Import button (appears after hovering over the From Workspace... field) and from the list of available networks, select the network named "net", i.e. the network created at this stage. By clicking on individual blocks representing network layers, you can see and modify the parameters of these layers. You can also expand and change the network structure by dragging blocks from the list in the left panel. After completing changes to the network, remember to export it (Export button) to the Matalaba workspace or generate an m-file creating the network (one of the Generate Code options). In turn, the Analyze button allows you to detect errors in the network structure and prepares a report summarizing the parameters of all layers. The described tool can also be used to build a new network and not only modify existing ones.

# 44
Test on the same set of images.
Creating an empty m-file and saving it, giving it a name e.g. digits_CNN_2.m.
In the first step, 2 random cases are taken from the set of images for each digit (there are 20 in total) and then they are classified using a previously trained network (classify() function). Then, the examined cases and their diagnosis results are presented on the screen.
It is worth repeating the test at least several times and checking whether the diagnosis is always performed correctly. Cases of incorrect recognition should be signaled with a red color of the assigned category label.

# 5. Using the GoogLeNet deep network for image recognition 
Creating a new GoogLeNet network (in the Matlab command window):
glnet = googlenet;
The model of this network must be previously downloaded from the Mathworks website - this has already been done and the model is installed.
To view the network structure, it is most convenient to use the deepNetworkDesigner tool. It can be noticed that this network does not have a serial structure and although it consists of over 140 layers, many of them work in parallel.

The GoogLeNet network recognizes the content of an image and assigns it to one of 1000 categories. Examples of randomly selected categories can be seen after executing the commands:
* classNames = glnet.Layers(end).ClassNames;
* numClasses = numel(classNames);
disp(classNames(randperm(numClasses,10)))
By double-clicking the classNames variable name in the Workspace tab in Matlab, you can view the full list of image categories recognized by the network.
#### 5.2 Testing network operation
Create an empty m-file and save it, giving it a name, e.g. GoogleNet_test.m.
The first step will be loading the image for recognition, but remember that the network operates on images with a resolution of 224x224 pixels.

The sequence of operations that need to be performed is loading the image (imread() function), recognizing it (classify() function) and displaying the image (imshow() function) and the recognition results (e.g. with the text(10,-10,char(label) command) );). All images are saved in .png format. The use of the mentioned functions has already been shown in the previous exercise.

The cause of incorrect recognition of the content of some images is primarily the lack of an appropriate category on the list and then assigning another category from the list that best fits (although it is not necessarily correct).
