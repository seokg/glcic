# Globally and Locally Consistent Image Completion (GLCIC)

This is an implementation of the image completion model proposed in the paper
([Globally and Locally Consistent Image Completion](
http://hi.cs.waseda.ac.jp/%7Eiizuka/projects/completion/data/completion_sig2017.pdf))
with TensorFlow.


# Requirements

- Python 3
- TensorFlow 1.3
- OpenCV


# Results



# Usage

## I. Prepare the training data

Put the images for training the "data/images" directory and convert images to npy format.

```
$ cd data
$ python to_npy.py
```

When you use the face images for training, it is recommended that you align the face landmarks with dlib before training.
If you have no time for preprocessing, utilize [mirror padding](
https://github.com/tadax/cvtools/tree/master/face_alignment/mirror_padding).


## II. Train the GLCIC model

```
$ cd src
$ python train.py
```

You can download the trained model file: [glcic_model.tar.gz](
https://drive.google.com/open?id=1jvP2czv_gX8Q1l0tUPNWLV8HLacK6n_Q)


# Approach

# Future Work

# License

MIT License

Copyright (c) 2018 tadax

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

