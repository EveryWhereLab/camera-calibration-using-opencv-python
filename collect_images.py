'''This script is for collecting images.
1. Specify the path to store images.
2. Specify the size of chessboard for detecting chessboard and displaying detection results.
3. Press 'c' to capture image.
4. Press 'q' to quit.
'''

import cv2
camera = cv2.VideoCapture(0)
ret, img = camera.read()
path = "./cam1_images/"
count = 0
CHESSBOARD_CORNER_NUM_X = 9
CHESSBOARD_CORNER_NUM_Y = 6

while True:
    ret, img = camera.read()
    img2 = img.copy()
    cv2.putText(img2, 'Captured image: {}. Press \'c\' to capture image. Press \'q\' to quit.'.format(count), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (CHESSBOARD_CORNER_NUM_X,CHESSBOARD_CORNER_NUM_Y), flags=cv2.CALIB_CB_FAST_CHECK)
    delay = 10
    if ret == True:
        # Draw and display the corners
        cv2.drawChessboardCorners(img2, (CHESSBOARD_CORNER_NUM_X,CHESSBOARD_CORNER_NUM_Y), corners, ret)
        delay = 50
    cv2.imshow("img", img2)
    key = cv2.waitKey(delay) & 0xFF
    if key == ord('c'):
        name = path + str(count)+".jpg"
        cv2.imwrite(name, img)
        count += 1
    if key == ord('q'):
        break;
cv2.destroyAllWindows()

