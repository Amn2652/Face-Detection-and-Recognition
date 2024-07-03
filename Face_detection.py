import cv2
import time

def generate_Dataset(img, id, img_id):
    cv2.imwrite(f"data/user.{id}.{img_id}.jpg", img)

def draw_boundary(img, classifier, scaleFactor, minNeighbours, color, text, clf):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbours)
    coords = []
    for (x, y, w, h) in features:
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        user_id, _ = clf.predict(gray_img[y:y + h, x:x + w])
        if user_id == 2:
            cv2.putText(img, "Amn", (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            cv2.putText(img, text, (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)

        coords.append([x, y, w, h])

    return coords, img

def recognize(img, clf, faceCascade):
    color = {"blue": (255, 0, 0), "red": (0, 0, 255), "green": (0, 255, 0)}
    coords, img = draw_boundary(img, faceCascade, 1.1, 10, color['green'], "Face", clf)
    return img

def detectFace(img, faceCascade, img_id, clf, user_id):
    color = {"blue": (255, 0, 0), "red": (0, 0, 255), "green": (0, 255, 0)}
    face_coords, img = draw_boundary(img, faceCascade, 1.1, 10, color['green'], "Face", clf)

    if len(face_coords) > 0:
        for (x, y, w, h) in face_coords:
            roi_area = img[y:y + h, x:x + w]
            generate_Dataset(roi_area, user_id, img_id)

    return img

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
clf = cv2.face.LBPHFaceRecognizer_create()
clf.read("classifier.yml")

video_capture = cv2.VideoCapture(0)
img_id = 0
user_id = 1
detecting_recognition = False
detecting_dataset = False
start_time = 0

while True:
    ret, img = video_capture.read()
    if not ret:
        break

    img = recognize(img, clf, faceCascade)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('n'):
        user_id += 1
        start_time = time.time()
        detecting_dataset = True

    if detecting_dataset:
        img = detectFace(img, faceCascade, img_id, clf, user_id)
        if time.time() - start_time > 5:
            detecting_dataset = False

    img_id += 1
    cv2.imshow("Face detection", img)

    if key == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
