# 얼굴 모자이크

얼굴을 detect하고 그 얼굴에 모자이크 처리를 한다.

## 코드
```python
import timeit
import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def videoDetector(cam,cascade):
    rate = 10
    while True:
        if(cam.get(cv2.CAP_PROP_POS_FRAMES) == cam.get(cv2.CAP_PROP_FRAME_COUNT)):
            cam.open('data/sample.mp4')
            
        start_t = timeit.default_timer()
        
        ret,img = cam.read()
        img = cv2.resize(img,dsize=None,fx=1.0,fy=1.0)
  
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
   
        results = cascade.detectMultiScale(gray,            
                                           scaleFactor= 1.1,
                                           minNeighbors=5,  
                                           minSize=(20,20)  
                                           )
                                                                           
        for box in results:
            x, y, w, h = box
            #cv2.rectangle(img, (x,y), (x+w, y+h), (255,255,255), thickness=2)
            if w and h:
                roi = img[y:y+h, x:x+w]   #영역 지정
                roi = cv2.resize(roi, (w//rate, h//rate)) # 1/rate 비율로 축소
                roi = cv2.resize(roi, (w,h), interpolation=cv2.INTER_AREA)  # 원래 크기로 확대
                img[y:y+h, x:x+w] = roi   # 원본 이미지에 적용
        
         # 영상 출력        
        cv2.imshow('mosaic',img)
        key = cv2.waitKey(1)
        if key == 27: # ESC
            break
    cv2.destroyAllWindows() 
    

# 영상 파일 
cam = cv2.VideoCapture('data/sample.mp4')

# 영상 탐지기
videoDetector(cam,face_cascade)
```
![image](https://github.com/rleoprleop/face-mosaic/assets/55969680/f4a6e946-19e5-474b-9a1a-f9f7c92566f0)


# 볼록 거울

얼굴을 detect하고 볼록거울 효과를 내었다

```python
import timeit
import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def videoDetector(cam,cascade):
    exp = 2   #볼록 1.1 이상 값
    scale = 1
    while True:
        if(cam.get(cv2.CAP_PROP_POS_FRAMES) == cam.get(cv2.CAP_PROP_FRAME_COUNT)):
            cam.open('data/sample.mp4')
            
        start_t = timeit.default_timer()
        
    
        ret,img = cam.read()
 
        img = cv2.resize(img,dsize=None,fx=1.0,fy=1.0)
      
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
      
        results = cascade.detectMultiScale(gray,            
                                           scaleFactor= 1.1,
                                           minNeighbors=5,  
                                           minSize=(20,20)  
                                           )
                                                                           
        for box in results:
            x, y, w, h = box
            #cv2.rectangle(img, (x,y), (x+w, y+h), (255,255,255), thickness=2)
            if w and h:
                roi = img[y:y+h, x:x+w]   # 관심영역 지정
                rows, cols = roi.shape[:2]

                # 매핑 배열 생성
                mapy, mapx = np.indices((rows, cols),dtype=np.float32)

                # 좌상단 기준좌표에서 -1~1로 정규화된 중심점 기준 좌표로 변경 
                mapx = 2*mapx/(cols)-1
                mapy = 2*mapy/(rows)-1

                # 직교좌표를 극 좌표로 변환 
                r, theta = cv2.cartToPolar(mapx, mapy)

                # 왜곡 영역만 중심확대/축소 지수 적용
                r[r< scale] = r[r<scale] **exp  

                # 극 좌표를 직교좌표로 변환
                mapx, mapy = cv2.polarToCart(r, theta)

                # 중심점 기준에서 좌상단 기준으로 변경
                mapx = ((mapx + 1)*cols)/2
                mapy = ((mapy + 1)*rows)/2
                # 재매핑 변환
                roi = cv2.remap(roi,mapx,mapy,interpolation=cv2.INTER_LINEAR)
                img[y:y+h, x:x+w] = roi   # 원본 이미지에 적용

              # 영상 출력        
        cv2.imshow('convex',img)
        key = cv2.waitKey(1)
        if key == 27: # ESC
            break
    cv2.destroyAllWindows() 
    

# 영상 파일 
cam = cv2.VideoCapture('data/sample.mp4')

# 영상 탐지기
videoDetector(cam,face_cascade)
```

![image](https://github.com/rleoprleop/face-mosaic/assets/55969680/ba7b9f31-e835-40c9-98d2-d48fe3c09e60)

# 오목거울

얼굴 detect하고 오목거울 효과를 내었다.

```python
import timeit
import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def videoDetector(cam,cascade):
    exp = 0.5  #오목 거울 0.1~1.0 사이 
    scale = 1
    while True:
        if(cam.get(cv2.CAP_PROP_POS_FRAMES) == cam.get(cv2.CAP_PROP_FRAME_COUNT)):
            cam.open('data/sample.mp4')
            
        start_t = timeit.default_timer()
        
        ret,img = cam.read()
  
        img = cv2.resize(img,dsize=None,fx=1.0,fy=1.0)
   
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
     
        results = cascade.detectMultiScale(gray,            
                                           scaleFactor= 1.1,
                                           minNeighbors=5,  
                                           minSize=(20,20)  
                                           )
                                                                           
        for box in results:
            x, y, w, h = box
            #cv2.rectangle(img, (x,y), (x+w, y+h), (255,255,255), thickness=2)
            if w and h:
                roi = img[y:y+h, x:x+w]   # 영역 지정
                rows, cols = roi.shape[:2]

                # 매핑 배열 생성
                mapy, mapx = np.indices((rows, cols),dtype=np.float32)

                # 좌상단 기준좌표에서 -1~1로 정규화된 중심점 기준 좌표로 변경 
                mapx = 2*mapx/(cols)-1
                mapy = 2*mapy/(rows)-1

                # 직교좌표를 극 좌표로 변환 
                r, theta = cv2.cartToPolar(mapx, mapy)

                # 왜곡 영역만 중심확대/축소 지수 적용
                r[r< scale] = r[r<scale] **exp  

                # 극 좌표를 직교좌표로 변환
                mapx, mapy = cv2.polarToCart(r, theta)

                # 중심점 기준에서 좌상단 기준으로 변경
                mapx = ((mapx + 1)*cols)/2
                mapy = ((mapy + 1)*rows)/2
                # 재매핑 변환
                roi = cv2.remap(roi,mapx,mapy,interpolation=cv2.INTER_LINEAR)
                img[y:y+h, x:x+w] = roi   # 원본 이미지에 적용
        
         # 영상 출력        
        cv2.imshow('concave',img)
        key = cv2.waitKey(1)
        if key == 27: # ESC
            break
    cv2.destroyAllWindows() 
    

# 영상 파일 
cam = cv2.VideoCapture('data/sample.mp4')

# 영상 탐지기
videoDetector(cam,face_cascade)
```
![image](https://github.com/rleoprleop/face-mosaic/assets/55969680/d5691170-cd3f-4ef5-accb-3fec3e0ae635)



