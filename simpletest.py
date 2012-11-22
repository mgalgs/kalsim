import cv

sz = (1280, 720)
fps = 30
# codec = cv.CV_FOURCC('P', 'I', 'M', '1')
# codec = cv.CV_FOURCC('f', 'f', 'd', 's')
codec = cv.CV_FOURCC('I', '4', '2', '0')
filename = "testing.avi"

vw = cv.CreateVideoWriter(filename, codec, fps, sz)
if not vw:
    print 'could not create writer!'

img = cv.CreateImage(sz, 8, 3)
cv.Circle(img, (500,500), 50, cv.RGB(0,255,255), -1)

if not cv.WriteFrame(vw, img):
    print 'write failed!'
else:
    print 'write succeeded!'
