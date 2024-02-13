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
global my_img

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

st.sidebar.title("Processes")
Conversion = st.sidebar.selectbox("Convert",convert,placeholder="Choose an option")
b1 = st.sidebar.button("Show output 1")
Filters = st.sidebar.selectbox("Filters",Filters,placeholder = "Choose an option")
b2 = st.sidebar.button("Show output 2")
Transform = st.sidebar.selectbox("Transformation",Transformation)
b3 = st.sidebar.button("Show output 3")
Morphological_Transformation = st.sidebar.selectbox("Morphological Transformation",M_Transformation)
b4 = st.sidebar.button("Show output 4")
Segmentation = st.sidebar.selectbox("Segmentation",Segmentation)
b5 = st.sidebar.button("Show output 5")

#Uploading And Showing

if upload:
    my_img = np.array(Image.open(upload))
    img = Image.open(upload)
    st.write("""### Input Image""")
    main_fig = plt.figure(figsize=(6,6))
    ax = main_fig.add_subplot(111)
    plt.imshow(img)
    plt.xticks([],[])
    plt.yticks([],[])
    st.pyplot(main_fig,use_container_width=True)

if b1:
    
    st.write("""### Output 1""")
    
    if Conversion=="Gray":
        
        #Conversion
        my_img = np.array(Image.open(upload))
        recolour = cv.cvtColor(my_img,cv.COLOR_BGR2GRAY)
        main_fig = plt.figure(figsize=(6,6))
        ax = main_fig.add_subplot(111)
        plt.imshow(recolour)
        plt.xticks([],[])
        plt.yticks([],[])
        st.pyplot(main_fig,use_container_width=True)
        
    elif Conversion=="RGB/BGR":
        recolour = cv.cvtColor(my_img,cv.COLOR_BGR2RGB)
        main_fig = plt.figure(figsize=(6,6))
        ax = main_fig.add_subplot(111)
        plt.imshow(recolour)
        plt.xticks([],[])
        plt.yticks([],[])
        st.pyplot(main_fig,use_container_width=True)
        
    elif Conversion=="Histogram":

        main_fig = plt.figure(figsize=(6,6))
        plt.title('Histogram')
        plt.xlabel('Bins')
        plt.ylabel('# of pixels')
        colour = ('b','g','r')
        for i,col in enumerate(colour):
            hist = cv.calcHist([my_img], [i], None, [256], [0,256])
            plt.plot(hist,color = col)
            plt.xlim([0,256])
            
        plt.show()
        st.pyplot(main_fig,use_container_width=True)

elif b2:
    st.write("""### Output 2""")
    if Filters=="Mean Filter":
        m_img = cv.blur(my_img,(5,5))
        # m_img = cv.cvtColor(m_img, cv.COLOR_BGR2RGB)
        main_fig = plt.figure(figsize=(6,6))
        ax = main_fig.add_subplot(111)
        plt.imshow(m_img)
        plt.xticks([],[])
        plt.yticks([],[])
        st.pyplot(main_fig,use_container_width=True)
        
    elif Filters=="Median Filter":
        median_img = cv.medianBlur(my_img,7)
        # median_img = cv.cvtColor(median_img, cv.COLOR_BGR2RGB)
        main_fig = plt.figure(figsize=(6,6))
        ax = main_fig.add_subplot(111)
        plt.imshow(median_img)
        plt.xticks([],[])
        plt.yticks([],[])
        st.pyplot(main_fig,use_container_width=True)
        
         
    elif Filters=="Gaussian Filter":
        blur_img = cv.cvtColor(my_img, cv.COLOR_BGR2GRAY)
        blur_img = cv.GaussianBlur(my_img, ksize = (5,5), sigmaX = 30,sigmaY = 300)
        # blur_img = cv.cvtColor(blur_img, cv.COLOR_BGR2GRAY)
        main_fig = plt.figure(figsize=(6,6))
        ax = main_fig.add_subplot(111)
        plt.imshow(blur_img)
        plt.xticks([],[])
        plt.yticks([],[])
        st.pyplot(main_fig,use_container_width=True)

    elif Filters=="Sobel Filter":
        gray = cv.cvtColor(my_img,cv.COLOR_RGB2GRAY)
        sobelx = cv.Sobel(gray,cv.CV_64F, 1, 0)
        sobely = cv.Sobel(gray,cv.CV_64F, 0, 1)
        combined_sobel = cv.bitwise_or(sobelx,sobely)
        main_fig = plt.figure(figsize=(6,6))
        ax = main_fig.add_subplot(111)
        plt.imshow(combined_sobel)
        plt.xticks([],[])
        plt.yticks([],[])
        st.pyplot(main_fig,use_container_width=True)

    elif Filters=="Prewitt Filter":
        gray = cv.cvtColor(my_img,cv.COLOR_BGR2GRAY)
        g_img = cv.GaussianBlur(gray, (3,3), 0)
        kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
        kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
        img_prewittx = cv.filter2D(g_img, -1, kernelx)
        img_prewitty = cv.filter2D(g_img, -1, kernely)
        combined_prewitt = img_prewittx + img_prewitty
        main_fig = plt.figure(figsize=(6,6))
        ax = main_fig.add_subplot(111)
        plt.imshow(combined_prewitt)
        plt.xticks([],[])
        plt.yticks([],[])
        st.pyplot(main_fig,use_container_width=True)

    elif Filters=="Canny Filter":
        blur = cv.GaussianBlur(my_img, (3,3), 0)
        canny_img = cv.Canny(blur,threshold1=205,threshold2=210)
        resized_img = cv.resize(canny_img, (int(my_img.shape[1]), int(my_img.shape[0])))
        main_fig = plt.figure(figsize=(6,6))
        ax = main_fig.add_subplot(111)
        plt.imshow(resized_img)
        plt.xticks([],[])
        plt.yticks([],[])
        st.pyplot(main_fig,use_container_width=True)
    
elif b3:
    st.write("""### Output 3""")

elif b4:
    st.write("""### Output 4""")

elif b5:
    st.write("""### Output 5""")
        