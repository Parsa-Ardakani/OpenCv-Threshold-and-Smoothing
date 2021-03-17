# Librareis
import numpy as np
from cv2 import cv2
from matplotlib import pyplot as plt
# Font
font = cv2.FONT_HERSHEY_SIMPLEX


#______________________________________________________________
#----------------Setting Window----------------------------------

cv2.namedWindow('Setting') # Creat a window named Setting
Setting_Background = np.zeros((300,700,3), np.uint8) # create a black image
Setting_Background[:] = [255,255,255] # Change the color of image to white


#----------------Description Window----------------------------------
cv2.namedWindow('Description') # Creat a windows named Setting
Description_Background = np.zeros((400,1000,3), np.uint8) # create a black image
Description_Background[:] = [255,255,255] # Change the color of image to white


#----------------Receiving Images----------------------------------
#Our Images
img1 = cv2.imread('Balloon.jpg')# inside parantesies put the location of image
img2 = cv2.imread('bird1.jpg')# if they are in the same file as the code only write the name


#______________________________________________________________
#---------------------Creating Bars----------------------------

#Please check the links below for Taskbar
#https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_trackbar/py_trackbar.html
#https://docs.opencv.org/3.4/da/d6a/tutorial_trackbar.html

def nothing(x):
    #https://docs.opencv.org/3.4/d7/dfc/group__highgui.html#gaf78d2155d30b728fc413803745b67a9b
    pass

cv2.createTrackbar('Threshold','Setting',100,255,nothing) #Treshold Range
cv2.createTrackbar('1def Thr','Setting',255,255,nothing) #Extra number to fill definitions
cv2.createTrackbar('2def Thr','Setting',2,255,nothing) #Extra number to fill definitions
cv2.createTrackbar('blur i','Setting',1,100,nothing) #Matrix i Range or fill extra definitions
cv2.createTrackbar('blur j','Setting',1,100,nothing) #Matrix j Range or fill extra definitions
cv2.createTrackbar('/ || 3','Setting',1,500,nothing) #Matrix division or fill extra definitions
cv2.createTrackbar('def Blur','Setting',1,200,nothing) #fourth number to fill extra definitions
cv2.createTrackbar("Treshold Method", 'Setting',0,7,nothing) #Treshold Type
cv2.createTrackbar('Blur Method', 'Setting',0,4,nothing) #Blur Type
cv2.createTrackbar('Order', 'Setting',1,1,nothing) # First, do blur and then Threshold or vise versa

#_______________________________________________________________________
#--------------------------Blur-----------------------------------------

def blur(Blur_type, def1, def2, def3, def4, image):
    # please check website below for smoothing(bluring) images 
    # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_filtering/py_filtering.html

    if Blur_type == 0:
        # 2D-Convolution Filter https://en.wikipedia.org/wiki/Kernel_(image_processing)

        # values can not be equal to zero
        if def1 == 0:
            def1 = 1
            cv2.createTrackbar('blur i','Setting',def1,100,nothing)
        if def2 == 0:
            def2 = 1
            cv2.createTrackbar('blur j','Setting',def2,100,nothing)
        if def3 == 0:
            def3 = 1
            cv2.createTrackbar('/ || 3','Setting',def3,500,nothing)

        # doing Kernel
        kernel = np.ones((def1,def2),np.float32)/def3
        # Apply kernel
        image_blur = cv2.filter2D(image,-1,kernel)

    elif Blur_type == 1:
        #Averaging

        # values can not be equal to zero
        if def1 == 0:
            def1 = 1
            cv2.createTrackbar('blur i','Setting',def1,100,nothing)
        if def2 == 0:
            def2 = 1
            cv2.createTrackbar('blur j','Setting',def2,100,nothing)

        # Blur Image
        image_blur = cv2.blur(image,(def1,def2))
    
    elif Blur_type == 2:
        #Gaussian Filtering

        # values can not be equal to zero
        if def1 == 0:
            def1 = 1
            cv2.createTrackbar('blur i','Setting',def1,100,nothing)
        if def2 == 0:
            def2 = 1
            cv2.createTrackbar('blur j','Setting',def2,100,nothing)
        
        # values should be odd not even
        if def1 % 2 == 0:
            def1 += 1
            cv2.createTrackbar('blur i','Setting',def1,100,nothing)
        if def2 % 2 == 0:
            def2 += 1
            cv2.createTrackbar('blur j','Setting',def2,100,nothing)

        # Bluring Image
        image_blur = cv2.GaussianBlur(image,(def1,def2),def4)
    

    elif Blur_type == 3:
        #Median Filtering
        
        # values can not be equal to zero
        if def4 == 0:
            def4 = 1
            cv2.createTrackbar('def Blur','Setting',def4,200,nothing)

        # values should be odd not even
        if def4 % 2 == 0:
            def2 += 1
            cv2.createTrackbar('def Blur','Setting',def4,200,nothing)
        
        image_blur = cv2.medianBlur(image,def4)
    
    else:
        #Bilateral Filtering
        image_blur =  cv2.bilateralFilter(image,def1,def2,def4)
    
    return image_blur # Output of Function


#______________________________________________________________________
#---------------------Threshold----------------------------------------
def threshold(threshold_type, thr_range, def1, def2, _image):
    #Please check links below for Thresholding
    #https://docs.opencv.org/master/d7/d4d/tutorial_py_thresholding.html
    #https://docs.opencv.org/master/db/d8e/tutorial_threshold.html

    _image = cv2.cvtColor(_image,cv2.COLOR_BGR2GRAY) # Change image color to gray
    if threshold_type == 0:
        #Global thresholding
        ret, image_thre = cv2.threshold(_image,thr_range,def1,cv2.THRESH_BINARY)
    
    elif threshold_type == 1:
        #Global thresholding Inverse
        ret, image_thre = cv2.threshold(_image,thr_range,def1,cv2.THRESH_BINARY_INV)
    
    elif threshold_type == 2:
        #Global thresholding TRUNC
        ret, image_thre = cv2.threshold(_image,thr_range,def1,cv2.THRESH_TRUNC)
    
    elif threshold_type == 3:
        #Global thresholding TOZERO
        ret, image_thre = cv2.threshold(_image,thr_range,def1,cv2.THRESH_TOZERO)
    
    elif threshold_type == 4:
        #Global thresholding TOZERO Inverse
        ret, image_thre = cv2.threshold(_image,thr_range,def1,cv2.THRESH_TOZERO_INV)
    
    elif threshold_type == 5:
        #Adaptive Mean Thresholding
        #Addapive mean block size should be odd
        if def1 % 2 == 0 or def1 == 0:
            def1 += 1
            cv2.createTrackbar('1def Thr','Setting',def1,255,nothing)
        image_thre = cv2.adaptiveThreshold(_image,thr_range,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,def1,def2)
        
    elif threshold_type == 6:
        #Adaptive Gaussian Thresholding
        #Addapive Gaussian block size should be odd
        if def1 % 2 == 0 or def1 == 0:
            def1 = 1
            cv2.createTrackbar('1def Thr','Setting',def1,255,nothing)
        image_thre = cv2.adaptiveThreshold(_image,thr_range,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,def1,def2)
    
    else:
        #Otsu Thresholding
        ret, image_thre = cv2.threshold(_image,thr_range,def1,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    return image_thre # Output of Function





#_______________________________________________________________________
#--------------Placing the first image on the Second One----------------


def Combination(Mask):
    #Please check the link below for Arithmetic Operations on Images
    #https://docs.opencv.org/3.4/d0/d86/tutorial_py_image_arithmetics.html#:~:text=You%20can%20add%20two%20images,just%20be%20a%20scalar%20value.


    # ROI and Image Rows, Columns, and Channels-------------------------------------
    
    
    Mask_inv = cv2.bitwise_not(Mask) # Invers the threshold
    rows, columns, channels = img1.shape #Storing the dimensions and channel of image
    """ we want to place an image on the top left so we create an ROI (for more information
    please check https://en.wikipedia.org/wiki/Region_of_interest) that
    starts from (0,0) and it is continued up to the dimensions of the first image"""
    roi = img2[0:rows, 0:columns] #place on the left-top corner

    # Black out the area of that our second picture would be placed on the second image-----------
    """We blackout the area because the value of black is 0 and has no effect on the colors when 
    we do the bitwise operation. If we do not do it and place the first image on the second one directly, 
    the second one will change colors of the first one
    For more informations please check https://en.wikipedia.org/wiki/Bitwise_operation and 
    https://docs.opencv.org/master/d0/d86/tutorial_py_image_arithmetics.html"""
    img2_BlackROI = cv2.bitwise_and(roi,roi,mask = Mask_inv)

    # Take out the object (what is in the white area in the threshold)----------------------------
    img1_TakeOut = cv2.bitwise_and(img1,img1,mask = Mask)

    # Add the first image to the second one
    dst = cv2.add(img2_BlackROI,img1_TakeOut)
    img2[0:rows, 0:columns] = dst
    #cv2.imshow('Image', img2)


#_______________________________________________________________
#-----------------------Main Loop-------------------------------

while True:
    img2 = cv2.imread('bird1.jpg') # Refresh image to avoid image overlapping
    #-------------------------Task Bar------------------------------------
    #Storing the value of each bar
    Threshold_Rnage = cv2.getTrackbarPos('Threshold','Setting')
    Threshold_def1 = cv2.getTrackbarPos('1def Thr','Setting')
    Threshold_def2 = cv2.getTrackbarPos('2def Thr','Setting')
    blur_i = cv2.getTrackbarPos('blur i','Setting')
    blur_j = cv2.getTrackbarPos('blur j','Setting')
    blur_division = cv2.getTrackbarPos('/ || 3','Setting')
    blur_def = cv2.getTrackbarPos('def Blur','Setting')
    Threshold_type = cv2.getTrackbarPos('Treshold Method','Setting')
    blur_type = cv2.getTrackbarPos('Blur Method','Setting')
    Order = cv2.getTrackbarPos('Order','Setting')

    #------------------------Blur & Threshold----------------------------------
    # Here we do blur and threshold and order can be changed

    if Order == 0:
        # First Blur and Then Threshold

        #Call Threshold function and Store Image made by this function in the img1_threshold / we directly call blur function in the function parameters 
        img1_threshold = threshold(Threshold_type, Threshold_Rnage, Threshold_def1, Threshold_def2, blur(blur_type, blur_i, blur_j, blur_division, blur_def, img1))
        # Store Blured image in the img1_blur
        img1_blur = blur(blur_type, blur_i, blur_j, blur_division, blur_def, img1)
        # Call combination and do the final operation on the image to remove black area
        Combination(img1_threshold)
    else:
        # First Threshold and then Blur

        #Call Threshold function and Store Image made by this function in the img1_blur / we directly call treshold function in the function parameters
        img1_blur = blur(blur_type, blur_i, blur_j, blur_division, blur_def, threshold(Threshold_type, Threshold_Rnage, Threshold_def1, Threshold_def2, img1)) 
        # Store threshold image in the img1_threshold
        img1_threshold = threshold(Threshold_type, Threshold_Rnage, Threshold_def1, Threshold_def2, img1)
        # Call combination and do the final operation on the image to remove black area
        Combination(img1_blur)
    
    #----------------Description-----------------------------------------------
    Description_Background = np.zeros((400,1000,3), np.uint8) # create a black image to refresh to avoid text overlapping
    Description_Background[:] = [255,255,255] # Change the color of image to white
    #\_Threshold Type_/
    if(Threshold_type == 0):
        # Global thresholding
        cv2.putText(Description_Background,'Threshold Method: Global thresholding',(10,25), font, 1,(0,0,0),2,cv2.LINE_AA)
        cv2.putText(Description_Background,'cv2.threshold(_image,thr_range,def1,cv2.THRESH_BINARY)',(10,60), font, 0.5,(0,0,0),0,cv2.LINE_AA)
        cv2.putText(Description_Background,'thr_range (Threshold):' + str(Threshold_Rnage) + ' def1 (1def Thr):' + str(Threshold_def1),(10,80), font, 0.5,(0,0,0),0,cv2.LINE_AA)
    
    elif(Threshold_type == 1):
        # Global thresholding Inverse
        cv2.putText(Description_Background,'Threshold Method: Global thresholding Inverse',(10,25), font, 1,(0,0,0),2,cv2.LINE_AA)
        cv2.putText(Description_Background,'cv2.threshold(_image,thr_range,def1,cv2.THRESH_BINARY_INV)',(10,60), font, 0.5,(0,0,0),0,cv2.LINE_AA)
        cv2.putText(Description_Background,'thr_range (Threshold):' + str(Threshold_Rnage) + ' def1 (1def Thr):' + str(Threshold_def1),(10,80), font, 0.5,(0,0,0),0,cv2.LINE_AA)
    
    elif(Threshold_type == 2):
        # Global thresholding TRUNC
        cv2.putText(Description_Background,'Threshold Method: Global thresholding TRUNC',(10,25), font, 1,(0,0,0),2,cv2.LINE_AA)
        cv2.putText(Description_Background,'cv2.threshold(_image,thr_range,def1,cv2.THRESH_TRUNC)',(10,60), font, 0.5,(0,0,0),0,cv2.LINE_AA)
        cv2.putText(Description_Background,'thr_range (Threshold):' + str(Threshold_Rnage) + ' def1 (1def Thr):' + str(Threshold_def1),(10,80), font, 0.5,(0,0,0),0,cv2.LINE_AA)
    
    elif(Threshold_type == 3):
        # Global thresholding TOZERO
        cv2.putText(Description_Background,'Threshold Method: Global thresholding TOZERO',(10,25), font, 1,(0,0,0),2,cv2.LINE_AA)
        cv2.putText(Description_Background,'cv2.threshold(_image,thr_range,def1,cv2.THRESH_TOZERO)',(10,60), font, 0.5,(0,0,0),0,cv2.LINE_AA)
        cv2.putText(Description_Background,'thr_range (Threshold):' + str(Threshold_Rnage) + ' def1 (1def Thr):' + str(Threshold_def1),(10,80), font, 0.5,(0,0,0),0,cv2.LINE_AA)
    
    elif(Threshold_type == 4):
        # Global thresholding TOZERO Inverse
        cv2.putText(Description_Background,'Threshold Method: Global thresholding TOZERO Inverse',(10,25), font, 1,(0,0,0),2,cv2.LINE_AA)
        cv2.putText(Description_Background,'cv2.threshold(_image,thr_range,def1,cv2.THRESH_TOZERO_INV)',(10,60), font, 0.5,(0,0,0),0,cv2.LINE_AA)
        cv2.putText(Description_Background,'thr_range (Threshold):' + str(Threshold_Rnage) + ' def1 (1def Thr):' + str(Threshold_def1),(10,80), font, 0.5,(0,0,0),0,cv2.LINE_AA)
    
    elif(Threshold_type == 5):
        # Adaptive Mean Thresholding
        cv2.putText(Description_Background,'Threshold Method: Adaptive Mean Thresholding',(10,25), font, 1,(0,0,0),2,cv2.LINE_AA)
        cv2.putText(Description_Background,'cv2.adaptiveThreshold(_image,thr_range,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,def1,def2))',(10,60), font, 0.5,(0,0,0),0,cv2.LINE_AA)
        cv2.putText(Description_Background,'thr_range (Threshold):' + str(Threshold_Rnage) + ' def1 (1def Thr):' + str(Threshold_def1)+ ' def2 (2def Thr):' + str(Threshold_def2),(10,80), font, 0.5,(0,0,0),0,cv2.LINE_AA)
    
    elif(Threshold_type == 6):
        # Adaptive Gaussian Thresholding
        cv2.putText(Description_Background,'Threshold Method: Adaptive Gaussian Thresholding',(10,25), font, 1,(0,0,0),2,cv2.LINE_AA)
        cv2.putText(Description_Background,'cv2.adaptiveThreshold(_image,thr_range,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,def1,def2))',(10,60), font, 0.5,(0,0,0),0,cv2.LINE_AA)
        cv2.putText(Description_Background,'thr_range (Threshold):' + str(Threshold_Rnage) + ' def1 (1def Thr):' + str(Threshold_def1)+ ' def2 (2def Thr):' + str(Threshold_def2),(10,80), font, 0.5,(0,0,0),0,cv2.LINE_AA)
    
    elif(Threshold_type == 7):
        # Otsu Thresholding
        cv2.putText(Description_Background,'Threshold Method: Otsu Thresholding',(10,25), font, 1,(0,0,0),2,cv2.LINE_AA)
        cv2.putText(Description_Background,'ret, image_thre = cv2.threshold(_image,thr_range,def1,cv2.THRESH_BINARY+cv2.THRESH_OTSU))',(10,60), font, 0.5,(0,0,0),0,cv2.LINE_AA)
        cv2.putText(Description_Background,'thr_range (Threshold):' + str(Threshold_Rnage) + ' def1 (1def Thr):' + str(Threshold_def1)+ ' def2 (2def Thr):' + str(Threshold_def2),(10,80), font, 0.5,(0,0,0),0,cv2.LINE_AA)

    
    #\_Blur Method_/
    if(blur_type == 0):
        # 2D-Convolution Filter
        cv2.putText(Description_Background,'Blur Method: 2D-Convolution Filter',(10,180), font, 1,(0,0,0),2,cv2.LINE_AA)
        cv2.putText(Description_Background,'np.ones((def1,def2),np.float32)/def3',(10,215), font, 0.5,(0,0,0),0,cv2.LINE_AA)
        cv2.putText(Description_Background,'def1 (blur i):' + str(blur_i) + ' def2 (blur j):' + str(blur_j) + ' def3 (/ || 3):' + str(blur_division),(10,250), font, 0.5,(0,0,0),0,cv2.LINE_AA)
    
    elif(blur_type == 1):
        # Averagin
        cv2.putText(Description_Background,'Blur Method: Averaging',(10,180), font, 1,(0,0,0),2,cv2.LINE_AA)
        cv2.putText(Description_Background,'cv2.blur(image,(def1,def2))',(10,215), font, 0.5,(0,0,0),0,cv2.LINE_AA)
        cv2.putText(Description_Background,'def1 (blur i):' + str(blur_i) + ' def2 (blur j):' + str(blur_j),(10,250), font, 0.5,(0,0,0),0,cv2.LINE_AA)
    
    elif(blur_type == 2):
        # Gaussian Filtering
        cv2.putText(Description_Background,'Blur Method: Gaussian Filtering',(10,180), font, 1,(0,0,0),2,cv2.LINE_AA)
        cv2.putText(Description_Background,'cv2.GaussianBlur(image,(def1,def2),def4)',(10,215), font, 0.5,(0,0,0),0,cv2.LINE_AA)
        cv2.putText(Description_Background,'def1 (blur i):' + str(blur_i) + ' def2 (blur j):' + str(blur_j) + ' def4 (def Blur):' + str(blur_def),(10,250), font, 0.5,(0,0,0),0,cv2.LINE_AA)

    elif(blur_type == 3):
        # Median Filtering
        cv2.putText(Description_Background,'Blur Method: Median Filtering',(10,180), font, 1,(0,0,0),2,cv2.LINE_AA)
        cv2.putText(Description_Background,'cv2.medianBlur(image,def4)',(10,215), font, 0.5,(0,0,0),0,cv2.LINE_AA)
        cv2.putText(Description_Background,' def4 (def Blur):' + str(blur_def),(10,250), font, 0.5,(0,0,0),0,cv2.LINE_AA)
    
    elif(blur_type == 4):
        # Bilateral Filtering
        cv2.putText(Description_Background,'Blur Method: Bilateral Filtering',(10,180), font, 1,(0,0,0),2,cv2.LINE_AA)
        cv2.putText(Description_Background,'cv2.bilateralFilter(image,def1,def2,def4)',(10,215), font, 0.5,(0,0,0),0,cv2.LINE_AA)
        cv2.putText(Description_Background,'def1 (blur i):' + str(blur_i) + ' def2 (blur j):' + str(blur_j) + ' def4 (def Blur):' + str(blur_def),(10,250), font, 0.5,(0,0,0),0,cv2.LINE_AA)
        


    #----------------Displaying------------------------------------------------
    cv2.imshow('Setting',Setting_Background) #Displaying Tasknavbar Window
    cv2.imshow('threshold', img1_threshold) #Dusplaying Threshold Window
    cv2.imshow('Blur', img1_blur) #Dusplaying Blur Window
    cv2.imshow('Image', img2) #Dusplaying Final Image Window
    cv2.imshow('description', Description_Background) #Description Window

    k=cv2.waitKey(10) & 0XFF # Every 10 ms Refresh or exit when Esc button is pressed
    if k== 27 :
        break
    
    

cv2.waitKey(0)
cv2.destroyAllWindows()
