import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import face_recognition
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
import copy
import os

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class ContentLoss(nn.Module):

    def __init__(self, target):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        #self.loss +=F.mse_loss(input[:, :, :, 1:], input[:, :, :, :-1])
        #self.loss += F.mse_loss(input[:, :, 1:, :], input[:, :, :-1, :])
        return input

def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)

class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

# desired depth layers to compute style/content losses :
content_layers_default = ['conv_3']
style_layers_default = ['conv_1', 'conv_2', 'conv_3']

def get_style_model_and_losses(cnn,
                               style_img, content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    cnn = copy.deepcopy(cnn)

    # normalization module
    #normalization = Normalization(normalization_mean, normalization_std).to(device)

    # just in order to have an iterable access to or list of content/syle
    # losses
    content_losses = []
    style_losses = []

    # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential()

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ContentLoss
            # and StyleLoss we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        elif layer.__class__.__name__:
            name = 'flatten_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            # add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses

def get_input_optimizer(input_img):
    # this line to show that input is a parameter that requires a gradient
    optimizer = optim.LBFGS([input_img.requires_grad_()], lr=5e-2)
    return optimizer

def run_style_transfer(cnn,
                       content_img, style_img, input_img, verbose, num_steps=450,
                       style_weight=100000, content_weight=10):
    """Run the style transfer."""
    if verbose:
        print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
                                                                     style_img,
                                                                     content_img)
    optimizer = get_input_optimizer(input_img)

    if verbose:
        print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:

        def closure():
            # correct the values of updated input image
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0 and verbose:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()

            return style_score + content_score

        optimizer.step(closure)

    # a last correction...
    input_img.data.clamp_(0, 1)
    #input_img.data = (input_img.data - input_img.data.min()) / (input_img.data.max() - input_img.data.min())

    return input_img

def get_poi(landmarks):

    output = []

    left_eye = tuple(np.array(landmarks[0]['left_eye']).mean(axis=0, dtype=int))
    nose_top = tuple(np.array(landmarks[0]['top_lip']).mean(axis=0, dtype=int))
    right_eye = tuple(np.array(landmarks[0]['right_eye']).mean(axis=0, dtype=int))

    output.append(left_eye)
    output.append(nose_top)
    output.append(right_eye)

    output = np.array(output).astype(np.float32)

    return output

def face_remap(shape):
    shape = np.array(shape)
    remapped_image = cv2.convexHull(shape)
    remapped_image = remapped_image.reshape(-1, 2)
    return remapped_image

def face_crop(img):
    #face location and face encoding
    img = np.array(img)
    locations = face_recognition.face_locations(img)[0]
    img = img[locations[0]:locations[2], locations[3]:locations[1]]
    return Image.fromarray(img), img.shape, locations

def style_transfer(encoder, src_img, src_transformed_img, dst_img, visualize=False, verbose=False):

    input_image = src_transformed_img.clone()
    output = run_style_transfer(encoder, src_transformed_img, dst_img, input_image, verbose)

    if visualize:

        fig, axes = plt.subplots(nrows=1, ncols=4, dpi=150)

        original_image = src_img

        c_img = src_transformed_img.cpu()[0].permute(1, 2, 0).numpy()
        c_img = (c_img * 255.0).astype(np.uint8)
        c_img = Image.fromarray(c_img)

        s_img = dst_img.cpu()[0].permute(1, 2, 0).numpy()
        s_img = (s_img * 255.0).astype(np.uint8)
        s_img = Image.fromarray(s_img)

        o_img = output.detach()[0].cpu().permute(1, 2, 0).numpy()
        o_img = (o_img * 255.0).astype(np.uint8)
        o_img = Image.fromarray(o_img)

        axes[0].imshow(original_image)
        axes[0].set_title('Original \n (Src)')

        axes[1].imshow(c_img)
        axes[1].set_title('Content \n swapped')

        axes[2].imshow(s_img)
        axes[2].set_title('Style \n (Dst)')

        axes[3].imshow(o_img)
        axes[3].set_title('Output \n styled')

    return output

def swap_face(s_img, d_img, visualize=False):

    '''function with faces'''
    landmarks_s = face_recognition.face_landmarks(np.array(s_img))
    landmarks_d = face_recognition.face_landmarks(np.array(d_img))

    poi_s = get_poi(landmarks_s)
    poi_d = get_poi(landmarks_d)

    matrix = cv2.getAffineTransform(poi_s, poi_d)

    #img = np.array(c_img.copy())
    #for key in landmarks_content[0]:
    #    for point in landmarks_style[0][key]:
    #        transformed_point = tuple((matrix @ np.array([point[0], point[1], 1.0])).astype(int))
    #        cv2.circle(img, transformed_point, 1, (0,0,255), -1)

    convex_landmarks = []
    keys = ('left_eyebrow', 'right_eyebrow', 'chin')
    for key in keys:
          for point in landmarks_d[0][key]:
                convex_landmarks.append(point)

    feature_mask = np.zeros((100, 100), np.uint8)

    mask = cv2.fillConvexPoly(feature_mask, face_remap(convex_landmarks), 255)
    affined_style_img = cv2.warpAffine(np.array(s_img), matrix, dsize=(100, 100))

    mask = mask * np.any(affined_style_img > 0, axis=-1)
    mask = cv2.GaussianBlur(mask.astype(np.uint8), ksize=(15, 15), sigmaX=5) / 255.0

    output_image = np.empty((100, 100, 3), np.uint8)
    for i in range(3):
        output_image[..., i] = mask * affined_style_img[..., i] + (1-mask) * np.array(d_img)[..., i]
    output_image = output_image.astype(np.uint8)
    if visualize:
        fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(10, 10))

        axes[0].imshow(s_img)
        axes[0].axis('off')

        axes[1].imshow(d_img)
        axes[1].axis('off')

        axes[2].imshow(Image.fromarray(output_image))
        axes[2].axis('off')

        axes[3].imshow(Image.fromarray(cv2.warpAffine(np.array(s_img), matrix, dsize=(100, 100))))
        axes[3].axis('off')

        axes[4].imshow(mask)
        axes[4].axis('off')

    return output_image

class FaceSwapper:

    def __init__(self, model, device):

        self.device = device
        self.model = model
        self.input_size = 100

    def swap(self, src_image, dst_image, vis=True, verbose=False):

        src_face, src_face_shape, src_face_locs = face_crop(src_image)
        dst_face, dst_face_shape, dst_face_locs = face_crop(dst_image)

        src_face = src_face.resize((self.input_size, self.input_size))
        dst_face = dst_face.resize((self.input_size, self.input_size))

        output = swap_face(src_face, dst_face, visualize=False)

        content_image = torch.FloatTensor(output).permute(2, 0, 1).unsqueeze(0).to(self.device)
        content_image = content_image / 255.0

        style_image = torch.FloatTensor(np.array(dst_face)).permute(2, 0, 1).unsqueeze(0).to(self.device)
        style_image = style_image / 255.0

        output_image = style_transfer(self.model, src_face, content_image, style_image, visualize=vis, verbose=verbose)

        o_img = output_image.detach()[0].cpu().permute(1, 2, 0).numpy()
        o_img = (o_img * 255.0).astype(np.uint8)
        o_img = Image.fromarray(o_img).resize((dst_face_shape[1], dst_face_shape[0]))

        dst_image = np.array(dst_image)
        dst_image[dst_face_locs[0]:dst_face_locs[2], dst_face_locs[3]:dst_face_locs[1], :] = np.array(o_img)

        return dst_image
