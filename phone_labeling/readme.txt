Files:
    common - should have the parameters used in the scripts and some common functions
    data_loader - reads the files and prepares the input/reference
    network -defines the neural network
    train -  training the neural network
    view - display images and statistics
    prep_dataset - prepares a csv file for training
    eval - evaluate the network and compute some metrics
    torch2tflite,run_torch2tf  - transform from pytorch to tensorflow
    data_ex - sample data used to train the neural netwok, recorded on my way to work
    networks
            - PixelLabeling.tflite - transformed model
            - checkpoint * - checkpoints during training
    CamRecorder - Android app to record and apply the neural network (app is not cleaned up but functional)


Folders need to be adapted to the specific environment. In common are most of the paths used but every script that is kind of stand alone
has some paths to be adapted.


Author: John ADAS Doe
Email: john.adas.doe@gmail.com
License: Apache-2.0
