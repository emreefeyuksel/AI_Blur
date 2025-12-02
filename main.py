import cv2
import mediapipe as mp
import numpy as np

mp_selfie = mp.solutions.selfie_segmentation
selfie_seg = mp_selfie.SelfieSegmentation(model_selection=1)  # 0: Genel, 1: Manzara/Daha hassas

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

print("Bulanıklaştırma başladı! Çıkmak için 'q' bas.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = selfie_seg.process(img_rgb)
    mask = results.segmentation_mask > 0.5
    mask_3d = np.stack((mask,) * 3, axis=-1)

    blurred_frame = cv2.GaussianBlur(frame, (55, 55), 0)

    output = np.where(mask_3d, frame, blurred_frame)

    cv2.imshow("AI Background Blur", output)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()