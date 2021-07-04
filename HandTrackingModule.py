import cv2
import mediapipe as mp
import time  # for checking frame rate
import math
import datetime

class handDetector():
    def __init__(self,mode = False, maxHands = 2, detecCon = 0.5, trackCon = 0.5):

        self.mode = mode
        self.maxHands = maxHands
        self.detecCon = detecCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode,self.maxHands,self.detecCon,self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIDs = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pink

    def findHands(self, img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handlns in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handlns, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0 , draw=True, boundbox=False):

        xlist = []  # x values
        ylist = []  # y values
        self.bbox = []  # bounding box
        self.lnList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, ln in enumerate(myHand.landmark):
                # print(id, ln)

                h, w, c = img.shape
                cx, cy = int(ln.x * w), int(ln.y * h)
                xlist.append(cx)
                ylist.append(cy)
                self.lnList.append([id, cx, cy])
                if draw:
                    # print(id, cx, cy)  # positions of landmark id
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
            xmin, xmax = min(xlist), max(xlist)
            ymin, ymax = min(ylist), max(ylist)
            self.bbox = xmin-10, ymin-10, xmax+10, ymax+10

            if boundbox:
                cv2.rectangle(img, (self.bbox[0:2]), (self.bbox[2:]), (0, 255, 0), 2)

        if boundbox:
            return self.lnList, self.bbox
        else:
            return self.lnList

    def fingersUp(self):

        # tips are 4,8,12,16,20
        fingers = []
        if len(self.lnList) != 0:
            if self.lnList[self.tipIDs[0]][1] < self.lnList[self.tipIDs[4]][1]:
                # for thumb - left hand
                if self.lnList[self.tipIDs[0]][1] < self.lnList[self.tipIDs[0] - 1][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            else:
                # for thumb - right hand
                if self.lnList[self.tipIDs[0]][1] > self.lnList[self.tipIDs[0] - 1][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            # for other fingers..
            for id in range(1, 5):
                if self.lnList[self.tipIDs[id]][2] < self.lnList[self.tipIDs[id] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            return fingers
        return [None]

    def findDistance(self, p1, p2, img, draw=True):
        x1, y1 = self.lnList[p1][1:]  # p1 position
        x2, y2 = self.lnList[p2][1:]  # p2 position
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)
        ABCpoints = [x1, y1, x2, y2, cx, cy]
        return length, img, ABCpoints
def main():
    pTime = 0
    cap = cv2.VideoCapture(0)  # webcam no. 0
    detector = handDetector()
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lnList= detector.findPosition(img)
        if len(lnList) != 0:
            print(lnList)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        # print(detector.fingersUp())
        cv2.rectangle(img, (20, 30), (200, 70), (0, 0, 0), cv2.FILLED)
        cv2.putText(img, f"FPS : {str(int(fps))}", (30, 60), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 140, 255), 2)
        cv2.rectangle(img, (330, 450), (640, 480), (0, 0, 0), cv2.FILLED)
        cv2.putText(img, f"{str(datetime.datetime.now())}", (335, 475), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 140, 255), 1)
        cv2.imshow("Image", img)
        cv2.waitKey(1)  # delay of 1ms

if __name__ == '__main__':
    main()