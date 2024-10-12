try:
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt

    image = cv2.imread('cat.jpg')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blue_channel, green_channel, red_channel = cv2.split(image)

    fig, ax = plt.subplots(1, 4, figsize=(20, 5))

    ax[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax[0].set_title("Original Image")
    ax[0].axis("off")

    ax[1].imshow(gray_image, cmap='gray')
    ax[1].set_title("Grayscale Image")
    ax[1].axis("off")

    ax[2].imshow(red_channel, cmap='Reds')
    ax[2].set_title("Red Channel")
    ax[2].axis("off")

    ax[3].imshow(green_channel, cmap='Greens')
    ax[3].set_title("Green Channel")
    ax[3].axis("off")

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(5, 5))
    plt.imshow(blue_channel, cmap='Blues')
    plt.title("Blue Channel")
    plt.axis("off")
    plt.show()

    print("Task 1 Completed!")

    rows, cols = image.shape[:2]

    translation_matrix = np.float32([[1, 0, 100], [0, 1, 50]])
    translated_image = cv2.warpAffine(image, translation_matrix, (cols, rows))
    translated_image_rgb = cv2.cvtColor(translated_image, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(6, 6))
    plt.imshow(translated_image_rgb)
    plt.title('Translated Image')
    plt.axis('off')
    plt.show()

    print("Task 2 Completed!")

    (h, w) = image.shape[:2]

    center = (w // 2, h // 2)
    angles = [90, 180, 270, 360]

    fig, axs = plt.subplots(1, 4, figsize=(20, 5))

    for i, angle in enumerate(angles):
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))
        rotated_image_rgb = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB)
        axs[i].imshow(rotated_image_rgb)
        axs[i].set_title(f'Rotated {angle}°')
        axs[i].axis('off')

    plt.tight_layout()
    plt.show()

    print("Task 3 Completed!")

    if image is None:
        print("Error: Image not loaded!")
    else:
        original_height, original_width = image.shape[:2]
        scale_percent = 15
        width_scaled = int(original_width * scale_percent / 100)
        height_scaled = int(original_height * scale_percent / 100)
        scaled_image = cv2.resize(image, (width_scaled, height_scaled), interpolation=cv2.INTER_LINEAR)

        width_enlarged = int(original_width * 2)
        height_enlarged = int(original_height * 2)
        enlarged_image = cv2.resize(image, (width_enlarged, height_enlarged), interpolation=cv2.INTER_CUBIC)

        resized_image = cv2.resize(image, (200, 400), interpolation=cv2.INTER_AREA)

        scaled_image_rgb = cv2.cvtColor(scaled_image, cv2.COLOR_BGR2RGB)
        enlarged_image_rgb = cv2.cvtColor(enlarged_image, cv2.COLOR_BGR2RGB)
        resized_image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        axs[0].imshow(scaled_image_rgb)
        axs[0].set_title('Scaled (15%, Linear)')
        axs[0].axis('off')

        axs[1].imshow(enlarged_image_rgb)
        axs[1].set_title('Enlarged (2x, Cubic)')
        axs[1].axis('off')

        axs[2].imshow(resized_image_rgb)
        axs[2].set_title('Resized (200x400, Area)')
        axs[2].axis('off')

        plt.tight_layout()
        plt.show()

    print("Task 4 Completed!")

    if image is None:
        print("Error: Image not loaded!")
    else:
        blurred_image = cv2.GaussianBlur(image, (15, 15), 0)
        blurred_image_rgb = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(8, 6))
        plt.imshow(blurred_image_rgb)
        plt.title('Blurred Image (Gaussian Blur)')
        plt.axis('off')
        plt.show()

    print("Task 5 Completed!")

    if image is None:
        print("Error: Image not loaded!")
    else:
        image_float32 = np.float32(image)

        alpha = 1.5
        beta = 50

        adjusted_image = cv2.convertScaleAbs(image_float32, alpha=alpha, beta=beta)
        adjusted_image_rgb = cv2.cvtColor(adjusted_image, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(8, 6))
        plt.imshow(adjusted_image_rgb)
        plt.title('Adjusted Brightness and Contrast')
        plt.axis('off')
        plt.show()

    print("Task 6 Completed!")

    if image is None:
        print("Error: Image not loaded!")
    else:
        startY = 50
        height = 200
        startX = 50
        width = 200

        cropped_image = image[startY:startY + height, startX:startX + width]
        cropped_image_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(6, 6))
        plt.imshow(cropped_image_rgb)
        plt.title('Cropped Image')
        plt.axis('off')
        plt.show()

    print("Task 7 Completed!")

    if image is None:
        print("Error: Image not loaded!")
    else:
        height, width = image.shape[:2]
        center_x, center_y = width // 2, height // 2

        radius = 50
        color = (0, 255, 0)
        thickness = 5
        cv2.circle(image, (center_x, center_y), radius, color, thickness)

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(6, 6))
        plt.imshow(image_rgb)
        plt.title('Image with Circle')
        plt.axis('off')
        plt.show()

    print("Task 8 Completed!")

    if image is None:
        print("Error: Image not loaded!")
    else:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        lower_threshold = 100
        upper_threshold = 200
        edges = cv2.Canny(gray_image, lower_threshold, upper_threshold)

        plt.figure(figsize=(6, 6))
        plt.imshow(edges, cmap='gray')
        plt.title('Canny Edge Detection')
        plt.axis('off')
        plt.show()

    print("Task 9 Completed!")

    image = cv2.imread('surveillance.png')

    if image is None:
        print("Error: Image not loaded!")
    else:
        alpha = 1.5
        beta = 50
        enhanced_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        blurred_image = cv2.GaussianBlur(enhanced_image, (15, 15), 0)
        gray_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_image, 100, 200)

        plt.figure(figsize=(16, 8))

        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(cv2.cvtColor(blurred_image, cv2.COLOR_BGR2RGB))
        plt.title('Enhanced + Blurred Image')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(edges, cmap='gray')
        plt.title('Edge Detected Image')
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    print("Task 10 Completed!")

    image = cv2.imread('cat.jpg')

    if image is None:
        print("Error: Image not loaded!")
    else:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])

        blue_hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        green_hist = cv2.calcHist([image], [1], None, [256], [0, 256])

        plt.figure(figsize=(12, 8))

        plt.subplot(2, 3, 1)
        plt.plot(gray_hist, color='black')
        plt.title('Grayscale Histogram')
        plt.xlim([0, 256])
        plt.grid(False)

        plt.subplot(2, 3, 2)
        plt.plot(blue_hist, color='blue')
        plt.title('Blue Channel Histogram')
        plt.xlim([0, 256])
        plt.grid(False)

        plt.subplot(2, 3, 3)
        plt.plot(green_hist, color='green')
        plt.title('Green Channel Histogram')
        plt.xlim([0, 256])
        plt.grid(False)

        plt.subplot(2, 3, 4)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    print("Task 11 Completed!")

    if image is None:
        print("Error: Image not loaded!")
    else:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kernel_blur = np.ones((5, 5), np.float32) / 25
        smoothed_image = cv2.filter2D(gray_image, -1, kernel_blur)

        kernel_sharpen = np.array([[0, -1, 0],
                                   [-1, 5, -1],
                                   [0, -1, 0]])
        sharpened_image = cv2.filter2D(gray_image, -1, kernel_sharpen)

        sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        sobel_edges = cv2.magnitude(sobelx, sobely)

        plt.figure(figsize=(12, 8))

        plt.subplot(2, 2, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(2, 2, 2)
        plt.imshow(smoothed_image, cmap='gray')
        plt.title('Smoothed (Blurred) Image')
        plt.axis('off')

        plt.subplot(2, 2, 3)
        plt.imshow(sharpened_image, cmap='gray')
        plt.title('Sharpened Image')
        plt.axis('off')

        plt.subplot(2, 2, 4)
        plt.imshow(sobel_edges, cmap='gray')
        plt.title('Sobel Edge Detection')
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    print("Task 12 Completed!")

except ImportError as e:
    print(f"Debugging failed: {e}")

except Exception as e:
    print(f"Error: {e}")