
import math
import cv2
import numpy as np
Import cupy as cp
  
# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture('test_video.mp4')#
#fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#out = cv2.VideoWriter('lab5-panels.mp4', fourcc, 50, (640, 360*4))
 
# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")
 
# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(3,3),0)
    edge = cv2.Canny(blur,25,75)
    roi = np.array([[540,310],[230,720],[1180,720],[590,310]])
    
    mask = np.zeros(edge.shape[:2], dtype="uint8")
    new_mask = np.zeros(edge.shape, edge.dtype)
    cv2.fillPoly (mask,pts=[roi], color=(255))
    
    cmask = frame.copy()
    cv2.fillPoly (cmask,pts=[roi], color=(0,255,0))
    
    alpha=0.1
    beta=(1.0 - alpha)
    roi=cv2.addWeighted(cmask, alpha, frame, beta, 0.0)
    
    crop=cv2.bitwise_and(mask, edge)
    
    ccrop=cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    Hlines=np.copy(ccrop)
    Hlines2=np.copy(ccrop)
    
    linesP = cv2.HoughLinesP(crop, 1, np.pi / 180, 50, None, 30, 100)
    
    if linesP is not None:
      for i in range(0, len(linesP)):
        l = linesP[i][0]
        cv2.line(Hlines, (l[0], l[1]), (l[2], l[3]), (0,255,0), 2, cv2.LINE_AA)
        cv2.line(frame, (l[0], l[1]), (l[2], l[3]), (0,255,0), 2, cv2.LINE_AA)    
    
    #linesP2 = cv2.HoughLinesP(crop, 1, np.pi / 180, 50, None, 50, 100)
    
    #if linesP2 is not None:
      #for i in range(0, len(linesP2)):
        #l = linesP2[i][0]
        #cv2.line(Hlines2, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)
        #cv2.line(frame, (l[0], l[1]), (l[2], l[3]), (0,0,255), 1, cv2.LINE_AA)
    
    # Display the resulting frame
    cv2.imshow('frame', frame)
    #cv2.imshow('gray', gray)
    #cv2.imshow('blur', blur)
    #cv2.imshow('edge', edge)
    #cv2.imshow('lines', lines)
    #cv2.imshow('roi', roi)
    #cv2.imshow('crop', crop)
    #cv2.imshow('Hlines1',Hlines)
    #cv2.imshow('Hlines2',Hlines2)
    
    # To display the steps taken we will put the resulting videos into a panel
    
    #convert back to color
    #cgray=cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cblur=cv2.cvtColor(blur, cv2.COLOR_GRAY2BGR)
    cedge=cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)
    
    #resize videos for panel window
    #frame1 =cv2.resize(frame,(640,360),interpolation=cv2.INTER_LINEAR)
    #gray1 =cv2.resize(cgray,(640,360),interpolation=cv2.INTER_LINEAR)
    blur1 =cv2.resize(cblur,(640,360),interpolation=cv2.INTER_LINEAR)
    #edge1 =cv2.resize(cedge,(640,360),interpolation=cv2.INTER_LINEAR)
    crop1 =cv2.resize(ccrop,(640,360),interpolation=cv2.INTER_LINEAR)
    Hlines0 =cv2.resize(Hlines,(640,360),interpolation=cv2.INTER_LINEAR)
    #Hlines1 =cv2.resize(Hlines2,(640,360),interpolation=cv2.INTER_LINEAR)
    roi1 =cv2.resize(roi,(640,360),interpolation=cv2.INTER_LINEAR)
   
    #combine videos we want into panel window and shift to right for main video
    panels = cv2.vconcat([blur1, roi1, crop1, Hlines0])
    cv2.moveWindow('panels',1280,0)
    cv2.imshow('panels',panels)
    
    #Write Video file 
    #out.write(panels)
    
    # Press Q on keyboard to  exit
    if cv2.waitKey(50) & 0xFF == ord('q'):
      break
 
  # Break the loop
  else: 
    break
 
# When everything done, release the video capture object
cap.release()
#out.release()
 
# Closes all the frames
cv2.destroyAllWindows()