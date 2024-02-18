import streamlit.web.cli as stcli
#from streamlit.cli import main
import streamlit as st
from PIL import Image
import opencv as cv
import matplotlib.pyplot as plt
import numpy as np

st.title("Digital Image Processor")
#File Uploader
upload = st.file_uploader(label="Upload Image:",type=["png","jpg","jpeg"])
global my_img


#Display Function
def display(i1):
    main_fig = plt.figure(figsize=(6,6))
    ax = main_fig.add_subplot(111)
    plt.imshow(i1,cmap='gray')
    plt.xticks([],[])
    plt.yticks([],[])
    st.pyplot(main_fig,use_container_width=True)
    
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
        # my_img = np.array(Image.open(upload))
        # recolour1 = cv.cvtColor(my_img,cv.COLOR_BGR2RGB)
        recolour = cv.cvtColor(my_img,cv.COLOR_RGB2GRAY)
        display(recolour)
        
    elif Conversion=="RGB/BGR":
        recolour = cv.cvtColor(my_img,cv.COLOR_BGR2RGB)
        display(recolour)
        
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
        display(m_img)
        
    elif Filters=="Median Filter":
        median_img = cv.medianBlur(my_img,7)
        # median_img = cv.cvtColor(median_img, cv.COLOR_BGR2RGB)
        display(median_img)
        
         
    elif Filters=="Gaussian Filter":
        blur_img = cv.cvtColor(my_img, cv.COLOR_RGB2GRAY)
        blur_img = cv.GaussianBlur(blur_img, ksize = (5,5), sigmaX = 30,sigmaY = 300)
        # blur_img = cv.cvtColor(blur_img, cv.COLOR_BGR2GRAY)
        display(blur_img)

    elif Filters=="Sobel Filter":
        gray = cv.cvtColor(my_img,cv.COLOR_RGB2GRAY)
        sobelx = cv.Sobel(gray,cv.CV_64F, 1, 0)
        sobely = cv.Sobel(gray,cv.CV_64F, 0, 1)
        combined_sobel = cv.bitwise_or(sobelx,sobely)
        display(combined_sobel)

    elif Filters=="Prewitt Filter":
        gray = cv.cvtColor(my_img,cv.COLOR_BGR2GRAY)
        g_img = cv.GaussianBlur(gray, (3,3), 0)
        kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
        kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
        img_prewittx = cv.filter2D(g_img, -1, kernelx)
        img_prewitty = cv.filter2D(g_img, -1, kernely)
        combined_prewitt = img_prewittx + img_prewitty
        display(combined_prewitt)
    
    
    elif Filters=="Canny Filter":
        blur = cv.GaussianBlur(my_img, (3,3), 0)
        canny_img = cv.Canny(blur,threshold1=205,threshold2=210)
        resized_img = cv.resize(canny_img, (int(my_img.shape[1]), int(my_img.shape[0])))
        display(resized_img)
    
elif b3:
    st.write("""### Output 3""")
    if Transform=="Fourier Transformation":
        # gray1 = cv.cvtColor(my_img,cv.COLOR_BGR2RGB)
        gray = cv.cvtColor(my_img,cv.COLOR_RGB2GRAY)
        f = cv.dft(np.float32(gray), flags=cv.DFT_COMPLEX_OUTPUT)
        fshift = np.fft.fftshift(f)
        magnitude = 20*np.log(cv.magnitude(fshift[:,:,0],fshift[:,:,1]))
        # Scale the magnitude for display
        magnitude = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX, cv.CV_8UC1)
        display(magnitude)
    
        
    

elif b4:
    st.write("""### Output 4""")
    kernel_MT = np.ones((5,5),np.uint8)
    
    if Morphological_Transformation=="Erosion":
        erode_img = cv.erode(my_img,kernel = kernel_MT,iterations = 1)
        # erode_img = cv.cvtColor(erode_img,cv.COLOR_BGR2RGB)
        display(erode_img)
    
        
    elif Morphological_Transformation=="Dilation":
        d_img = cv.dilate(my_img,kernel = kernel_MT,iterations = 1)
        # d_img = cv.cvtColor(d_img,cv.COLOR_BGR2RGB)
        display(d_img)
    elif Morphological_Transformation=="Opening":
        o_img = cv.morphologyEx(my_img,cv.MORPH_OPEN,kernel_MT)
        display(o_img)
    elif Morphological_Transformation=="Closing":
        c_img = cv.morphologyEx(my_img,cv.MORPH_CLOSE,kernel_MT)
        display(c_img)
elif b5:
    st.write("""### Output 5""")
    if Segmentation=="Contour Detection":
        img_gray = cv.cvtColor(my_img, cv.COLOR_RGB2GRAY)
        ret, thresh = cv.threshold(img_gray, 150, 255, cv.THRESH_BINARY)
        contours, hierarchy = cv.findContours(image=thresh, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_NONE)
        image_copy = my_img.copy()
        cv.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv.LINE_AA)
        display(image_copy)
        
    elif Segmentation=="Using K-Means":
        # s_img = cv.cvtColor(my_img,cv.COLOR_BGR2RGB)
        pixel_vals = my_img.reshape((-1,3))
        pixel_vals = np.float32(pixel_vals)
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.85)
        k = 10
        retval, labels, centers = cv.kmeans(pixel_vals, k, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        segmented_data = centers[labels.flatten()]
        segmented_image = segmented_data.reshape((my_img.shape))
        display(segmented_image)
