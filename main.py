#Cumhuriye Tüluğ Küçüköğüt S025791 Departmant of Computer Engineering
import cv2  
import numpy as np  
import os
from os import path
def takeFiles(path):
    files = []
    for fileName in os.listdir(path): #For taking files from the path
        image = os.path.join(path,fileName)
        if image is not None:
            files.append(image)
    return files
def calculateCloseNumbers(points):
    diff_list = []
    for i in range(1,len(points)):
        x = points[i] - points[i-1]
        diff_list.append(x)
    avg = np.mean(diff_list) 

    newPoint = []
    newPoint.append(points[0])
    for i in range(1,len(points)):
        if points[i] - points[i-1] > avg :
            newPoint.append(points[i])

    return newPoint # lines are grouped according to the difference between them
#reading image
imagePath = "Please enter the image path"
if path.exists(imagePath):
    images = takeFiles(imagePath)
    if images:
        for i in range(len(images)) :
            if not os.path.isfile(images[i]):
                pass
            elif cv2.imread(images[i]) is None:
                pass
            else :
                kernel = np.ones((5,5),np.uint8)
                image = cv2.imread(images[i])  
                dim = (640,480)
                image = cv2.resize(image, dim)
                img = cv2.GaussianBlur(image, (5, 5), 0)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
                
                thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                          cv2.THRESH_BINARY, 51, 5) 

                edges = cv2.Canny(thresh, 50, 150, apertureSize=3)
                edges = cv2.dilate(edges,kernel,iterations = 1)
                edges = cv2.erode(edges,kernel,iterations = 1)
                mask = np.ones(image.shape, dtype="uint8") * 255
                
                
                contours,hier = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                if len(contours) > 0:
                    cnt = None
                    max_area = 0
                    for c in contours:
                        area = cv2.contourArea(c)
                        if area > max_area :
                            cnt = c
                            max_area = area

                    cv2.drawContours(mask, [cnt], -1, 0, -1)
                
                mask = 255 - mask
                
                result = cv2.bitwise_and(image, mask)
                result[mask==0] = 255
                resultHorizontal = result.copy()
                resultVertical = result.copy()
                

                verticalKernel = np.ones((25,1),np.uint8)
                horizontalKernel = np.ones((1,25),np.uint8)
                img = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY) 
                mask = np.ones(result.shape, dtype="uint8") * 255

                thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                          cv2.THRESH_BINARY, 51, 5) 


                ################################## VERTICAL LINE ######################################################
                maskVertical = np.ones(image.shape, dtype="uint8") * 255

                edges = cv2.Canny(thresh, 10, 30, apertureSize=3)
                edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, verticalKernel)
               

                edges = cv2.dilate(edges,verticalKernel,iterations = 50)
                edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, verticalKernel)
                edges =cv2.erode(edges,verticalKernel,iterations =50)  
                lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
                startPoints = []
                verticalLineCounter = 0
                if lines is not None :
                    for line in lines:
                        rho,theta = line[0]
                        if theta == 0.0 :
                            a = np.cos(theta)
                            b = np.sin(theta)
                            x0 = a * rho
                            y0 = b * rho
                            x1 = int(x0 + 1000 * (-b))
                            y1 = int(y0 + 1000 * (a))
                            x2 = int(x0 - 1000 * (-b))
                            y2 = int(y0 - 1000 * (a))
                            startPoints.append(x1)


                    startPoints = sorted(startPoints,key=lambda x:int(x))
                    closeListVertical = calculateCloseNumbers(startPoints)
                    repeat = []
                    for line in lines:
                        rho,theta = line[0]
                        a = np.cos(theta)
                        b = np.sin(theta)
                        x0 = a * rho
                        y0 = b * rho
                        x1 = int(x0 + 1000 * (-b))
                        if x1 in closeListVertical and  x1 not in repeat:
                            y1 = int(y0 + 1000 * (a))
                            x2 = int(x0 - 1000 * (-b))
                            y2 = int(y0 - 1000 * (a))
                            repeat.append(x1)
                            cv2.line(maskVertical, (x1, y1), (x2, y2), (0, 0, 255), 1)
                            verticalLineCounter = verticalLineCounter + 1

                ###############################################HORIZONTAL LINE############################################
                startPointsForHorizontal = []
                maskHorizontal = np.ones(image.shape, dtype="uint8") * 255
                edges = cv2.Canny(thresh, 100, 200, apertureSize=3)
                edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontalKernel)
              
                edges = cv2.dilate(edges,horizontalKernel,iterations = 50)
                edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, horizontalKernel)
                edges =cv2.erode(edges,horizontalKernel,iterations =50) 

                lines = cv2.HoughLines(edges, 1, np.pi / 2, 200)
                horizontalLineCounter = 0
                if lines is not None :
                    for line in lines:
                        rho,theta = line[0]
                        a = np.cos(theta)
                        b = np.sin(theta)
                        x0 = a * rho
                        y0 = b * rho
                        x1 = int(x0 + 1000 * (-b))
                        y1 = int(y0 + 1000 * (a))
                        x2 = int(x0 - 1000 * (-b))
                        y2 = int(y0 - 1000 * (a))
                        startPointsForHorizontal.append(y1)
                    startPointsForHorizontal = sorted(startPointsForHorizontal,key=lambda x:int(x))
                    closeListHorizontal = calculateCloseNumbers(startPointsForHorizontal)
                    repeat = []
                    for line in lines:
                        rho,theta = line[0]
                        a = np.cos(theta)
                        b = np.sin(theta)
                        x0 = a * rho
                        y0 = b * rho
                        x1 = int(x0 + 1000 * (-b))

                        y1 = int(y0 + 1000 * (a))
                        if y1 in closeListHorizontal  and y1 not in repeat:
                            repeat.append(x1)
                            x2 = int(x0 - 1000 * (-b))
                            y2 = int(y0 - 1000 * (a))
                            cv2.line(maskHorizontal, (x1, y1), (x2, y2), (0, 0, 255), 1)
                            horizontalLineCounter = horizontalLineCounter + 1

                concataneteOfLines = cv2.bitwise_and(maskVertical, maskHorizontal)
                concatanete = cv2.bitwise_and(result, concataneteOfLines)
                img = cv2.cvtColor(concataneteOfLines, cv2.COLOR_BGR2GRAY) 
                ret,thresh = cv2.threshold(img,127,255,1)

                contours,h = cv2.findContours(thresh,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)

                if len(contours) > 0:
                    counter = 0
                    for cnt in contours:
                        approx = cv2.approxPolyDP(cnt,.01*cv2.arcLength(cnt,True),True)
                        if len(approx)==4:
                            x, y, w, h = cv2.boundingRect(cnt)
                            cv2.rectangle(concatanete, (x, y), (x+w, y+h), (0, 0, 255), 2)
                            counter = counter + 1


                
                checkSudoku = concatanete.copy()
                cv2.imshow('rectangles', concatanete) 
                if horizontalLineCounter == 10 and verticalLineCounter == 10 and counter == 81:
                    for cnt in contours:
                        approx = cv2.approxPolyDP(cnt,.01*cv2.arcLength(cnt,True),True)
                        if len(approx)==4:
                            x, y, w, h = cv2.boundingRect(cnt)
                            cv2.rectangle(checkSudoku, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    print("The image is sudoku")
                else :
                    print("The image is not sudoku")

                cv2.imshow('determinationOfSudoku', checkSudoku) 
                
                k = cv2.waitKey(10)
                cv2.destroyAllWindows()

    else :
        print("The path does not include images")           
        pass

else    :
    print("Entered path can not be found")


