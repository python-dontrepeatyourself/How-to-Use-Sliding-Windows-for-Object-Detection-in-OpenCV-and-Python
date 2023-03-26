import cv2


def pyramid(image, scale=1.5, min_size=(40, 40)):
    yield image

    # Generate pyramid levels until minimum size is reached
    while True:
        # Calculate the new image size based on
        # the scale factor and resize the image
        w = int(image.shape[1] / scale)
        h = int(image.shape[0] / scale)
        image = cv2.resize(image, (w, h))

        # If the new level is too small, stop generating more levels
        if image.shape[0] < min_size[1] or image.shape[1] < min_size[0]:
            break

        yield image


def sliding_window(image, step_size, window_size):
    # get the window and image sizes
    h, w = window_size
    image_h, image_w = image.shape[:2]

    # loop over the image, taking steps of size `step_size`
    for y in range(0, image_h, step_size):
        for x in range(0, image_w, step_size):
            # define the window
            window = image[y:y + h, x:x + w]
            # if the window is below the minimum window size, ignore it
            if window.shape[:2] != window_size:
                continue
            # yield the current window
            yield (x, y, window)


image = cv2.imread("1.jpg")
w, h = 156, 156


for resized in pyramid(image):
    for (x, y, window) in sliding_window(resized, step_size=40, window_size=(w, h)):

        # in our case we are just going to display the window, but for a complete
        # object detection algorithm, this is where you would classify the window
        # using a pre-trained machine learning classifier (e.g., SVM, logistic regression, etc.)

        clone = resized.copy()
        cv2.rectangle(clone, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow("Window", clone)
        cv2.waitKey(100)
