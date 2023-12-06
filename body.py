def extrace_object_deom():
    capture = cv.VideoCapture("b.mp4")#
    while(True):
        ret,frame = capture.read()# Не ложь
        if ret == False:
            break
        hsv=cv.cvtColor(frame,cv.COLOR_BGR2HSV)# Преобразовать цветное пространство в HSV
        lower_hsv = np.array([100,43,46])#  .
        upper_hsv = np.array([124,255,255])# Высокое значение
        mask = cv.inRange(hsv,lowerb=lower_hsv,upperb=upper_hsv)# Найти зеленую зону
        dst = cv.bitwise_and(frame,frame,mask=mask)#
        cv.imshow("video",frame)
        cv.imshow("mask",mask);
        cv.imshow("dst", dst);
        c = cv.waitKey(40)#esc выход
        if c == 27:
            break
