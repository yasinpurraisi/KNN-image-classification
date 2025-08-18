
import cv2

knn = cv2.ml.KNearest_create()
knn = knn.load('knn_best_model.yml')

# enter your k_best value
k_best = 1

image_path = "enteryourimagepath.jpg"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = cv2.resize(image, (30, 30))
image_pixel = image.flatten().astype('float32')
image_pixel = image_pixel.reshape(1, -1)

ret, result, neighbours, dist = knn.findNearest(image_pixel, k=k_best)
prediction = "🐈cat" if int(result[0][0])==0 else "🐕dog"
print("Predicted label:", prediction)