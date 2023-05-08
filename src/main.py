import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

MIN_MATCH_COUNT = 10

template = cv.imread('../img/Nike.jpeg',0)          # queryImage

# Initiate SIFT detector
# Initiate SIFT detector
sift = cv.SIFT_create()
# find the keypoints and descriptors with SIFT
kp_template, des_template = sift.detectAndCompute(template,None)

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
matcher = cv.FlannBasedMatcher(index_params, search_params)

cap=cv.VideoCapture(2)

while (cap.isOpened()):
    # Read a frame from the video capture object
    ret, frame = cap.read()
    if not ret:
        print("Frame Missed")

    # Convert frame to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Compute keypoints and descriptors for current frame
    kp_frame, des_frame = sift.detectAndCompute(gray, None)
    if des_frame is None:
        img_homography=frame
    else:

        # Match features between template and frame
        matches = matcher.knnMatch(des_template, des_frame, k=2)

        # Apply ratio test to filter out false matches
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

        # Draw matched keypoints on frame
        img_match = cv.drawMatches(template, kp_template, gray, kp_frame, good_matches, None)

        # Find homography between template and frame
        if len(good_matches) > 10:
            src_pts = np.float32([kp_template[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

            h, w = template.shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv.perspectiveTransform(pts, M)
            img_homography = cv.polylines(frame, [np.int32(dst)], True, 255, 3, cv.LINE_AA)
        else:
            img_homography = frame

    # Display the frame with matched keypoints and homography
    cv.imshow('Video', img_homography)
    # Exit loop if 'q' key is pressed
    if cv.waitKey(1) & 0xFF == ord('q'):
        break




# Release the video capture object and destroy all windows
cap.release()
cv.destroyAllWindows()
# # store all the good matches as per Lowe's ratio test.
# good = []
# for m,n in matches:
#     if m.distance < 0.7*n.distance:
#         good.append(m)

# if len(good)>MIN_MATCH_COUNT:
#     src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
#     dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
#     M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
#     matchesMask = mask.ravel().tolist()
#     h,w = img1.shape
#     pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
#     dst = cv.perspectiveTransform(pts,M)
#     img2 = cv.polylines(img2,[np.int32(dst)],True,255,3, cv.LINE_AA)
# else:
#     print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
#     matchesMask = None

# draw_params = dict(matchColor = (0,255,0), # draw matches in green color
#                    singlePointColor = None,
#                    matchesMask = matchesMask, # draw only inliers
#                    flags = 2)
# img3 = cv.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
# plt.imshow(img3, 'gray'),plt.show()