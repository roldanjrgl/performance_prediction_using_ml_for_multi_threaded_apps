Performance Prediction using Machine Learning for Multithreaded applications.
Multicore Processors: Architecture & Programming


Submitted by : Samvid Avinash Zare (sz3369) , Jorge Roldan (jlr9718@nyu.edu), Aditya Walvekar (aw4496@nyu.edu)

How to run the code:

1. To generate the data

1.1. Unzip the zip file
1.2. Run the command "cd src"
1.3. copy the data_gen.py file to parsec folder and paste it
1.4 run python3 data_gen.py

Result: The script will generate the file data.csv

2. To train the model and measure the performance

2.1 Ensure that the data.csv file is present at the same level as perf_preiction.py file.
2.2 Run the command "python3 perf_preiction.py"

Result: 
1. All the resulting plots will be generated/replaced in the directory "results/plot_images"
2. The files for error and Pearsons's coefficient will be generated/replaced in the directory "results".