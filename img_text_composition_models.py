import numpy as np
import torch
import torchvision
import torch.nn.functional as F
import text_model
import torch_functions


class ConCatModule(torch.nn.Module):

  def __init__(self):
    super(ConCatModule, self).__init__()

  def forward(self, x):
    x = torch.cat(x, dim=1)
    return x


class ImgTextCompositionBase(torch.nn.Module):
  """Base class for image + text composition."""

  def __init__(self):
    super(ImgTextCompositionBase, self).__init__()
    self.normalization_layer = torch_functions.NormalizationLayer(
        normalize_scale=4.0, learn_scale=True)
    self.soft_triplet_loss = torch_functions.TripletLoss()

  def extract_img_feature(self, imgs):
    raise NotImplementedError

  def extract_text_feature(self, texts):
    raise NotImplementedError

  def compose_img_text(self, imgs, texts):
    raise NotImplementedError

  def compute_loss(self,
                   imgs_query,
                   modification_texts,
                   imgs_target,
                   soft_triplet_loss=True):
    mod_img1 = self.compose_img_text(imgs_query, modification_texts)
    mod_img1 = self.normalization_layer(mod_img1)
    img2 = self.extract_img_feature(imgs_target)
    img2 = self.normalization_layer(img2)
    assert (mod_img1.shape[0] == img2.shape[0] and
            mod_img1.shape[1] == img2.shape[1])
    if soft_triplet_loss:
      return self.compute_soft_triplet_loss_(mod_img1, img2)
    else:
      return self.compute_batch_based_classification_loss_(mod_img1, img2)

  def compute_soft_triplet_loss_(self, mod_img1, img2):
    triplets = []
    labels = range(mod_img1.shape[0]) + range(img2.shape[0])
    for i in range(len(labels)):
      triplets_i = []
      for j in range(len(labels)):
        if labels[i] == labels[j] and i != j:
          for k in range(len(labels)):
            if labels[i] != labels[k]:
              triplets_i.append([i, j, k])
      np.random.shuffle(triplets_i)
      triplets += triplets_i[:3]
    assert (triplets and len(triplets) < 2000)
    return self.soft_triplet_loss(torch.cat([mod_img1, img2]), triplets)

  def compute_batch_based_classification_loss_(self, mod_img1, img2):
    x = torch.mm(mod_img1, img2.transpose(0, 1))
    labels = torch.tensor(range(x.shape[0])).long()
    labels = torch.autograd.Variable(labels).cuda()
    return F.cross_entropy(x, labels)


class ImgEncoderTextEncoderBase(ImgTextCompositionBase):
  """Base class for image and text encoder."""

  def __init__(self, texts, embed_dim):
    super(ImgEncoderTextEncoderBase, self).__init__()

    # img model
    img_model = torchvision.models.resnet18(pretrained=True)

    class GlobalAvgPool2d(torch.nn.Module):

      def forward(self, x):
        return F.adaptive_avg_pool2d(x, (1, 1))

    img_model.avgpool = GlobalAvgPool2d()
    img_model.fc = torch.nn.Sequential(torch.nn.Linear(512, embed_dim))
    self.img_model = img_model

    # text model
    self.text_model = text_model.TextLSTMModel(
        texts_to_build_vocab=texts,
        word_embed_dim=embed_dim,
        lstm_hidden_dim=embed_dim)

  def extract_img_feature(self, imgs):
    return self.img_model(imgs)

  def extract_text_feature(self, texts):
    return self.text_model(texts)


class SimpleModelImageOnly(ImgEncoderTextEncoderBase):

  def compose_img_text(self, imgs, texts):
    return self.extract_img_feature(imgs)


class SimpleModelTextOnly(ImgEncoderTextEncoderBase):

  def compose_img_text(self, imgs, texts):
    return self.extract_text_feature(texts)


class Concat(ImgEncoderTextEncoderBase):
  """Concatenation model."""

  def __init__(self, texts, embed_dim):
    super(Concat, self).__init__(texts, embed_dim)

    # composer
    class Composer(torch.nn.Module):
      """Inner composer class."""

      def __init__(self):
        super(Composer, self).__init__()
        self.m = torch.nn.Sequential(
            torch.nn.BatchNorm1d(2 * embed_dim), torch.nn.ReLU(),
            torch.nn.Linear(2 * embed_dim, 2 * embed_dim),
            torch.nn.BatchNorm1d(2 * embed_dim), torch.nn.ReLU(),
            torch.nn.Dropout(0.1), torch.nn.Linear(2 * embed_dim, embed_dim))

      def forward(self, x):
        f = torch.cat(x, dim=1)
        f = self.m(f)
        return f

    self.composer = Composer()

  def compose_img_text(self, imgs, texts):
    img_features = self.extract_img_feature(imgs)
    text_features = self.extract_text_feature(texts)
    return self.compose_img_text_features(img_features, text_features)

  def compose_img_text_features(self, img_features, text_features):
    return self.composer((img_features, text_features))

