# QSharpNet
A Sharpness Based Loss Function for Removing Out-of-Focus Blur

Please contact {aurangau}@tcd{dot}ie for a link to the pre-trained weights

## How-to
1. Put clean images in folder 'CleanImages'
2. Run cells in the Python Notebook - demoscript.ipynb. This will generate a blurry image with a blur kernel of size 5 * 5
3. Run subsequent cells to generate 3 images - Image generated using only MAE, Image generated using Fine-Tuned Model (E9), Image generated using Fine-Tuned Model (E11)

Please note that the image must be of size 128 $\times$ 128 $\times$ 1 and NOT a 3-channel image
