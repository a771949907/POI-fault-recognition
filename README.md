# POI-fault-recognition
To maintain the up-to-dateness of a modern navigation map, massive street images are captured every day. Target recognition methods are applied to recognize POI(Point of Interest) data in those images so changes of POI in the real world can be discovered, and therefore, a navigation map can be updated. However, faults may occur in the results of target recognition methods due to their accuracy limits, which leads to incorrect updates and is extremely harmful to the up-to-dateness of a naigation map. This project provides a method to distinguish the fault POI data in the results of target recognition methods, based on geographic information of POI data with a gradient-boosting-based model to take advantage of this information.



# POI Geographic Information
Geographic information of POI data can be divided into four categories:
1. position information: indicates where the POI is located. Latitude and Longitude, city, community, etc.
2. road network information: indicates the road features and networks of POI.
3. environment information: indicates the features of environment of POI.
4. information inherited from image target recognition methods


The first three kinds of geographic information of POI can be extracted in a navigation map system, and the image information can be inherited from image target recognition methods, both of which finally constitute the file train_data.csv. Some of the features and their names and descriptions are listed as follows:

![image](https://github.com/chenyu-se/POI-fault-recognition/assets/17283947/37116a33-61ec-4632-bf42-a109b3d2f551)


# Activity Diagram
The activity diagram of POI fault recogition is shown as follown:
![image](https://github.com/chenyu-se/POI-fault-recognition/assets/17283947/0972affc-48b3-4981-bc61-2112b3c8b019)





# Model
The structure of the POI fault recognition model is shown as follow:
![image](https://github.com/chenyu-se/POI-fault-recognition/assets/17283947/82b0eb15-a454-437f-b33e-256e59b0658e)

POI geographic information is denoised by CRF(Conditional Random Forests). Then, four gradient boosting models are trained and finally combined via stacking. Stacking is a model ensembling technique that combines multiple individual models to improve overall performance. It involves training a meta-model, also known as a stacking model or a meta-learner, on the predictions of the base models. The base models are trained on the original dataset, while the meta-model learns to combine their predictions. 

In POI fault recognition, the meta-model is LR(Logistic Regression), and the base models of POI fault recognition are:
1. XGBoost and LightGBM: two popular open-source gradient boosting frameworks used for machine learning tasks.
2. GBDT1: Gradient Boosting Decision Tree with different parameters to GBDT2.
3. GBDT2: Gradient Boosting Decision Tree with different parameters to GBDT1.






