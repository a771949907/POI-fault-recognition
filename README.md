# POI-fault-recognition
To maintain the up-to-dateness of a modern navigation map, massive street images are captured every day. Target recognition methods are applied to recognize POI(Point of Interest) data in those images so a navigation map can be updated. However, faults may occur in the results of target recognition methods, since every method has its limits. This project provides a method to distinguish the fault POI data in the results of target recognition methods, based on geographic information of POI data with a GBDT-based classifier to take advantage of this information.

# geographic information
Geographic information of POI data can be divided into four categories:
1. position information: indicates where the POI is located. Latitude and Longitude, city, community, etc.
2. road network information: indicates the road features and networks of POI.
3. environment information: indicates the features of environment of POI.
4. information inherited from image recognition methods.

The geographic information of POI is extracted in a navigation map system, which finally constitutes the file train_data.csv. Some of the features and their names and descriptions are listed as follows:

![image](https://github.com/chenyu-se/POI-fault-recognition/assets/17283947/433a7f6a-d83e-44d3-bb72-3f437bd64abf)

# Activity diagram


# Model
The structure of the classifier is shown as follow:
![image](https://github.com/chenyu-se/POI-fault-recognition/assets/17283947/82b0eb15-a454-437f-b33e-256e59b0658e)

POI geographic information is denoised by CRF(Conditional Random Forests). Then, four GBDT models are trained and finally stacked.




