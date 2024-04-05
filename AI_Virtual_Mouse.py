import cv2
import time
import handtrackingmodule as htm
import numpy as np
import pyautogui

# Default parameters
sensitivity = 8
wCam, hCam = 640, 480
frameR = 100  # frame reduction
pTime = 0
plocX, plocY = 0, 0  # previous locations of x and y
clocX, clocY = 0, 0  # current locations of x and y

cap = cv2.VideoCapture(0)
cap.set(3, wCam)  # width
cap.set(4, hCam)  # height

detector = htm.handDetector(detectionCon=0.60, maxHands=1)  # only one hand at a time
wScr, hScr = pyautogui.size()

while True:
    # 1. Find hand Landmarks
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)

    # 2. Get the tip of the index and middle fingers
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        # 3. Check which fingers are up
        fingers = detector.fingersUp()

        # in moving mouse it was easy to move mouse upwards but in downward direction
        #  it is tough so we are setting region
        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)

        # 4. Only Index Finger : Moving Mode
        if fingers[1] == 1 and fingers[2] == 0:
            # 5. Convert Coordinates
            x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))  # converting x coordinates
            y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))  # converting y

            # 6. Smoothen Values
            clocX = plocX + (x3 - plocX) / sensitivity
            clocY = plocY + (y3 - plocY) / sensitivity

            # 7. Move Mouse
            pyautogui.moveTo(wScr - clocX, clocY)  # Move mouse
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
             # circle shows that we are in moving mode
            plocX, plocY = clocX, clocY

        # 8. Both Index and middle fingers are up :
            #  Clicking Mode but only if both fingers are near to each other
        if fingers[1] == 1 and fingers[2] == 1:
            # 9. Find distance between fingers so that we can make sure fingers are together
            length, img, lineInfo = detector.findDistance(8, 12, img)
            # 10. Click mouse if distance short
            if length < 30:
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                pyautogui.click()

    # 11. Frame Rate
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    # 12. Display
    cv2.imshow("Image", img)

    # Handle key press events
    key = cv2.waitKey(1)
    if key == 27:  # Press 'Esc' to exit
        break
    elif key == ord('s'):  # Press 's' to change sensitivity
        sensitivity = int(input("Enter new sensitivity: ") or sensitivity)
    elif key == ord('w'):  # Press 'w' to change camera window width
        wCam = int(input("Enter new camera window width: ") or wCam)
        cap.set(3, wCam)
    elif key == ord('h'):  # Press 'h' to change camera window height
        hCam = int(input("Enter new camera window height: ") or hCam)
        cap.set(4, hCam)

cv2.destroyAllWindows()
cap.release()
