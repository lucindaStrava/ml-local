This is my repo for local model development in python.

# Train Model with TensorFlow/Keras/Xgboost in Python
Follow the following steps to create virtual environment and install all dependencies:

    virtualenv env
    source env/bin/activate
    pip3 install -r requirements.txt

If you use PyCharm for Python Dev (recommended), follow the instruction https://www.jetbrains.com/help/pycharm/creating-virtual-environment.html to run PyCharm on the virtualenv created in previous steps.

To run script in Jupyter Notebook

    jupyter notebook
    jupyter notebook --ip=0.0.0.0   (Run within EC2 such that the port can be accessed externally)

To show tensor board, go to the root directory that contains TensorFlow logs, within the virtual environment

    tensorboard --logdir ./yourLogDirName

    