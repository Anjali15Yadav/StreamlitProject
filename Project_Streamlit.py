import streamlit.web.cli as stcli
# from streamlit.cli import main
import streamlit as st
from PIL import ImageTk,Image
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

st.title("Digital Image Processor")
#File Uploader
upload = st.file_uploader(label="Upload Image:",type=["png","jpg","jpeg"])

#Making Options/Menus
convert = (
    "Gray",
    "Histogram",
    "RGB/BGR"
)
Filters = (
    "Mean Filter",
    "Median Filter",
    "Gaussian Filter",
    "Sobel Filter",
    "Prewitt Filter",
    "Canny Filter"
)
Transformation = (
    "Fourier Transformation",
)
M_Transformation = (
    "Erosion",
    "Dilation",
    "Opening",
    "Closing"
)
Segmentation = (
    "Using K-Means",
    "Contour Detection"
)


Conversion = st.selectbox("Convert",convert,placeholder="Choose an option")
Filters = st.selectbox("Filters",Filters,placeholder = "Choose an option")
Transform = st.selectbox("Transformation",Transformation)
Morphological_Transformation = st.selectbox("Morphological Transformation",M_Transformation)
Segmentation = st.selectbox("Segmentation",Segmentation)

#Uploading And Showing

if upload:
    img = Image.open(upload)
    st.write("""### Input Image""")
    main_fig = plt.figure(figsize=(6,6))
    ax = main_fig.add_subplot(111)
    plt.imshow(img)
    plt.xticks([],[])
    plt.yticks([],[])
    st.pyplot(main_fig,use_container_width=True)

