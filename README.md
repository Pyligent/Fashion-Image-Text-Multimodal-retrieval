## Image-Text Multimodal Style Retrieval

### Jeans & Pants Image Text Multimodal Retrevial

- Dataset: pants and Jeans dataset from [Fashion200k](https://github.com/xthan/fashion-200k)
  - Data Label Format: file_path, _ , attributes
  - examples: **women/pants/cropped_pants/88482428/88482428_0.jpeg**  	_   	**purple bristol printed slim-leg capri jeans only macy's**

- Test Query Format:
  - Source Image:
  - Query Format: Change **One Attribues**
  - Return: Target Images
  - Example: 
    - Source Image: **women/pants/wide-leg_and_palazzo_pants/91340759/91340759_2.jpeg**
    - Query: change **black** to **green**

- Methods:
  - Image Features: ResNet18 AvgPool2D feature layer: 512, <img src="https://render.githubusercontent.com/render/math?math=$f_i$"> 
  - Textula Features: LSTM text encoding: hidden dim: 512, <img src="https://render.githubusercontent.com/render/math?math=$f_t$"> 
  - Joint embedding: concatenate two vectors, concat(<img src="https://render.githubusercontent.com/render/math?math=$f_i$">,<img src="https://render.githubusercontent.com/render/math?math=$f_t$"> )
  - Train Network: 2 layer MLPs with RELU, with batch-norm and dropout(0.1)
  - Loss Function: mini-batch retreival loss (paired images)
  - Metric: Recall@Top-K(1,5,10,50,100)

- Training Process:
  - batch_size : 32
  - learning_rate : 1e-2
  - weight_decay : 1e-06
  - num_iters : 150243
  - epoch: 182
  - Trainset size: 26301
  - Testset size: 5343
  - Results: 
      - train recall_top1_correct_composition 0.8945
      - train recall_top5_correct_composition 0.9828
      - train recall_top10_correct_composition 0.9956
      - train recall_top50_correct_composition 0.9999
      - train recall_top100_correct_composition 1.0
      - test recall_top1_correct_composition 0.1035
      - test recall_top5_correct_composition 0.2227
      - test recall_top10_correct_composition 0.2867
      - test recall_top50_correct_composition 0.4862
      - test recall_top100_correct_composition 0.6393
      
- Training Environment: 
  - GCP: n1-highmem-8(8 vCPUs, 52 GB memory) & 1 Tesla P100

- Code information:
  - datasets.py: load dataset
  - text_model.py: LSTM encoding
  - img_text_composition_models.py: extract ResNet18, concat image and textual embedding vectors
  - test_retrieval.py: train/test retrieval
  - inference_top10.py: inference top10 examples



### Query Demo
[!img2](https://github.com/Pyligent/image-text-multimodal-retrieval/blob/master/img/result1.png)


[!img1](https://github.com/Pyligent/image-text-multimodal-retrieval/blob/master/img/result2.png)

[!img3](https://github.com/Pyligent/image-text-multimodal-retrieval/blob/master/img/result3.png)
