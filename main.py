import solver
from utilities import *
import cv2
import numpy as np

pathImage = ".\sudokus\5"
heightImg = 450
widthImg = 450
model = intializePredectionModel()


# preparing the image - preprocessing
img = cv2.imread(pathImage)
img = cv2.resize(img, (widthImg, heightImg))  
imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8) 
imgThreshold = preProcess(img)

# finding contours
imgContours = img.copy() # saves all of the contours
imgBigContour = img.copy() # stores the biggest one of all contours
contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 3)

# finding the biggest countour - the sudoku plane
biggest, maxArea = biggestContour(contours) 
# print(biggest)
if biggest.size != 0:
    biggest = reorder(biggest)
    # print(biggest)
    cv2.drawContours(imgBigContour, biggest, -1, (0, 0, 255), 25) # Draw the biggest countour
    pts1 = np.float32(biggest) # Prepare points for wrap
    pts2 = np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]]) # Prepare points for wrap
    matrix = cv2.getPerspectiveTransform(pts1, pts2) 
    imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
    imgDetectedDigits = imgBlank.copy()
    imgWarpColored = cv2.cvtColor(imgWarpColored,cv2.COLOR_BGR2GRAY)

    # spliting image and findigs avaible digits
    imgSolvedDigits = imgBlank.copy()
    boxes = splitBoxes(imgWarpColored)
    # print(len(boxes))
    # cv2.imshow("Sample",boxes[65])
    numbers = getPredection(boxes, model)
    # print(numbers)
    imgDetectedDigits = displayNumbers(imgDetectedDigits, numbers, color=(255, 0, 255))
    numbers = np.asarray(numbers)
    posArray = np.where(numbers > 0, 0, 1)
    # print(posArray)


    # finding the solution of sudoku
    board = np.array_split(numbers,9)
    # print(board)
    try:
        solver.solve(board)
    except:
        pass
    # print(board)
    flatList = []
    for sublist in board:
        for item in sublist:
            flatList.append(item)
    solvedNumbers =flatList*posArray
    imgSolvedDigits= displayNumbers(imgSolvedDigits,solvedNumbers)

    # overlaying the solution on the image
    pts2 = np.float32(biggest)
    pts1 =  np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]]) 
    matrix = cv2.getPerspectiveTransform(pts1, pts2)  
    imgInvWarpColored = img.copy()
    imgInvWarpColored = cv2.warpPerspective(imgSolvedDigits, matrix, (widthImg, heightImg))
    inv_perspective = cv2.addWeighted(imgInvWarpColored, 1, img, 0.5, 1)


    
    cv2.imshow('Solved Sudoku',  inv_perspective)

else:
    print("No Sudoku Found")

cv2.waitKey(0)