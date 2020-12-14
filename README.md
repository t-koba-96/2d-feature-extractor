# 2D CNN Feature Extractor  


## Models  

Check [Models_Output_Shape](./main/models/models.txt)

- Resnet50  

- Resnet101

- VGG16  


## Extract Usage  

```
python extract.py [dataset_dir] [save_dir] [model] [--spatial_feature] [--cuda]  
```

- [dataset_dir] : path to dataset(videos) directory

- [save_dir] : path to the directory you want to save video features

- [model] : model architecture. [ resnet50 | resnet101 | vgg16 ]

- [--spatial_feature] : if you want to keep the extracted feature in 2d shape, store_true

- [--cuda] : choose cuda num