#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import torch, torchvision
from tqdm import tqdm as tqdm
import PIL
import skimage.io
import datasets

import img_text_composition_models

from matplotlib.pyplot import figure, imshow, axis
from matplotlib.image import imread



def infer_top10(opt, model, testset):
  """Tests a model over the given testset."""
  model.eval()
  test_queries = testset.get_test_queries()

  all_imgs = []
  all_captions = []
  all_queries = []
  all_target_captions = []
  if test_queries:
    # compute test query features
    imgs = []
    mods = []
    for t in tqdm(test_queries):
      imgs += [testset.get_img(t['source_img_id'])]
      mods += [t['mod']['str']]
      if len(imgs) >= opt.batch_size or t is test_queries[-1]:
        if 'torch' not in str(type(imgs[0])):
          imgs = [torch.from_numpy(d).float() for d in imgs]
        imgs = torch.stack(imgs).float()
        imgs = torch.autograd.Variable(imgs).cuda()
        mods = [t for t in mods]
        f = model.compose_img_text(imgs, mods).data.cpu().numpy()
        all_queries += [f]
        imgs = []
        mods = []
    all_queries = np.concatenate(all_queries)
    all_target_captions = [t['target_caption'] for t in test_queries]

    # compute all image features
    imgs = []
    for i in tqdm(range(len(testset.imgs))):
      imgs += [testset.get_img(i)]
      if len(imgs) >= opt.batch_size or i == len(testset.imgs) - 1:
        if 'torch' not in str(type(imgs[0])):
          imgs = [torch.from_numpy(d).float() for d in imgs]
        imgs = torch.stack(imgs).float()
        imgs = torch.autograd.Variable(imgs).cuda()
        imgs = model.extract_img_feature(imgs).data.cpu().numpy()
        all_imgs += [imgs]
        imgs = []
    all_imgs = np.concatenate(all_imgs)
    all_captions = [img['captions'][0] for img in testset.imgs]

  else:
    # use training queries to approximate training retrieval performance
    imgs0 = []
    imgs = []
    mods = []
    for i in range(10000):
      item = testset[i]
      imgs += [item['source_img_data']]
      mods += [item['mod']['str']]
      if len(imgs) > opt.batch_size or i == 9999:
        imgs = torch.stack(imgs).float()
        imgs = torch.autograd.Variable(imgs)
        mods = [t for t in mods]
        f = model.compose_img_text(imgs.cuda(), mods).data.cpu().numpy()
        all_queries += [f]
        imgs = []
        mods = []
      imgs0 += [item['target_img_data']]
      if len(imgs0) > opt.batch_size or i == 9999:
        imgs0 = torch.stack(imgs0).float()
        imgs0 = torch.autograd.Variable(imgs0)
        imgs0 = model.extract_img_feature(imgs0.cuda()).data.cpu().numpy()
        all_imgs += [imgs0]
        imgs0 = []
      all_captions += [item['target_caption']]
      all_target_captions += [item['target_caption']]
    all_imgs = np.concatenate(all_imgs)
    all_queries = np.concatenate(all_queries)

  # feature normalization
  for i in range(all_queries.shape[0]):
    all_queries[i, :] /= np.linalg.norm(all_queries[i, :])
  for i in range(all_imgs.shape[0]):
    all_imgs[i, :] /= np.linalg.norm(all_imgs[i, :])

  # match test queries to target images, get nearest neighbors
  sims = all_queries.dot(all_imgs.T)
  if test_queries:
    for i, t in enumerate(test_queries):
      sims[i, t['source_img_id']] = -10e10  # remove query image
  nn_result = [np.argsort(-sims[i, :])[:10] for i in range(sims.shape[0])]
  nn_result1 = nn_result

  # compute recalls
  out = []
  nn_result = [[all_captions[nn] for nn in nns] for nns in nn_result]
  for k in [1, 5, 10]:
    r = 0.0
    for i, nns in enumerate(nn_result):
      if all_target_captions[i] in nns[:k]:
        r += 1
    r /= len(nn_result)
    out += [('recall_top' + str(k) + '_correct_composition', r)]

    if opt.dataset != 'fashion200k':
      r = 0.0
      for i, nns in enumerate(nn_result):
        if all_target_captions[i].split()[0] in [c.split()[0] for c in nns[:k]]:
          r += 1
      r /= len(nn_result)
      out += [('recall_top' + str(k) + '_correct_adj', r)]

      r = 0.0
      for i, nns in enumerate(nn_result):
        if all_target_captions[i].split()[1] in [c.split()[1] for c in nns[:k]]:
          r += 1
      r /= len(nn_result)
      out += [('recall_top' + str(k) + '_correct_noun', r)]

  return nn_result1,nn_result,out



def show_source(path,source_id):
    
    idx = test_queries[source_id]['source_img_id']
    img_path = testset.imgs[idx]['file_path']
    img_path = path + '/' +img_path
    
    pil_im = Image.open(img_path, 'r')
    print('Product Attributes:   ',test_queries[source_id]['source_caption'])
    print('Query:   ',test_queries[source_id]['mod']['str'])
    imshow(np.asarray(pil_im))
    
def show_results(path,result_id):
    
    
    img_path = testset.imgs[result_id]['file_path']
    img_path = path + '/' +img_path
    
    pil_im = Image.open(img_path, 'r')
    print('Product Attributes:   ',testset.imgs[result_id]['captions'])
    imshow(np.asarray(pil_im))

    
    
def get_result(array_list):
    result_files = []
    for i in range(len(array_list)):
        img_path = testset.imgs[array_list[i]]['file_path']
        img_path = path + '/' +img_path
        result_files += [img_path]
    
    return result_files


def show_result_all(result_files):
    
    
    fig = figure(figsize = (25,25))
    number_of_files = len(result_files)
    for i in range(number_of_files):
        a=fig.add_subplot(1,number_of_files,i+1)
        image = imread(result_files[i])
        imshow(image)
        axis('off')

