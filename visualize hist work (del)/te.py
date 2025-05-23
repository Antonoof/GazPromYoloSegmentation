import cv2

# Загрузка
image = cv2.imread('test1.png')

# преобразование в черно-белый (на всякий случай)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# выравнивание по гистограмме изображения
equalized_image = cv2.equalizeHist(gray_image)

# результат сохраняем
cv2.imwrite('test1_res.png', equalized_image)