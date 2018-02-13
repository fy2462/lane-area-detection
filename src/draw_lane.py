import imageio

imageio.plugins.ffmpeg.download()
from moviepy.editor import VideoFileClip

# import helper methods from other files
from src.threshold_helpers import *

'''
load undistortion matrix from camera 
'''
with open('test_dist_pickle.p', 'rb') as pick:
    dist_pickle = pickle.load(pick)

mtx = dist_pickle['mtx']
dist = dist_pickle['dist']

bot_width = .6
mid_width = .15
height_pct = .7
bottom_trim = .9


def get_roi_mapping(img):
    img_size = (img.shape[1], img.shape[0])
    offset = img_size[0] * .25
    src = np.float32([[img.shape[1] * (.5 - mid_width / 2), img.shape[0] * height_pct],
                      [img.shape[1] * (.5 + mid_width / 2), img.shape[0] * height_pct], \
                      [img.shape[1] * (.5 + bot_width / 2), img.shape[0] * bottom_trim],
                      [img.shape[1] * (.5 - bot_width / 2), img.shape[0] * bottom_trim]])
    dst = np.float32(
        [[offset, 0], [img_size[0] - offset, 0], [img_size[0] - offset, img_size[1]], [offset, img_size[1]]])
    return src, dst


def change_perspective(img):
    img_size = (img.shape[1], img.shape[0])
    src, dst = get_roi_mapping(img)

    # used to show that src points matched line
    # cv2.fillConvexPoly(img, src.astype('int32'), 1)
    # plt.imshow(img)
    # plt.title('lines')
    # plt.show()

    # create a transformation matrix based on the src and destination points
    M = cv2.getPerspectiveTransform(src, dst)

    # transform the image to birds eye view given the transform matrix
    warped = cv2.warpPerspective(img, M, (img_size[0], img_size[1]))
    return warped


'''
get the pixels for the left and right lanes and return them.
most of the code from Udacity's lectures on calculating the curvature
'''


def lr_curvature(binary_warped):
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    y_piex_num = binary_warped.shape[0]
    x_piex_num = binary_warped.shape[1]
    start_y = int(y_piex_num * 0.8)
    aa = binary_warped[start_y:, :]
    histogram = np.sum(aa, axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines

    # plt.plot(histogram)
    # plt.title('histo')
    # plt.show()

    # plt.imshow(out_img)
    # plt.title('before windows')
    # plt.show()

    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    # revert_array = histogram[::-1]
    # revert_left_base = np.argmax(revert_array[:midpoint])
    # rightx_base = histogram.shape[0] - revert_left_base;
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 50
    dacay_rate = 0.8
    # Set height of windows
    window_height = np.int(binary_warped.shape[0] / nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 50
    # Set minimum number of pixels found to recenter window
    minpix = 40
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 140, 0), 2)
        print('rectangle 1', (win_xleft_low, win_y_low), (win_xleft_high, win_y_high))
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 140, 0), 2)
        print('rectangle 2', (win_xright_low, win_y_low), (win_xright_high, win_y_high))
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
            nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
            nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = int(
                dacay_rate * leftx_current + (1 - dacay_rate) * np.int(np.mean(nonzerox[good_left_inds])))
        if len(good_right_inds) > minpix:
            rightx_current = int(
                dacay_rate * rightx_current + (1 - dacay_rate) * np.int(np.mean(nonzerox[good_right_inds])))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # At this point, you're done! But here is how you can visualize the result as well:
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [30, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 30]
    # plt.imshow(out_img)
    # plt.plot(left_fitx, ploty, color='yellow')
    # plt.plot(right_fitx, ploty, color='yellow')
    # plt.xlim(0, x_piex_num)
    # plt.ylim(y_piex_num, 0)
    # plt.title('rancge_test')
    # plt.show()

    # convert from pixel space to meter space
    ym_per_pix = 30 / y_piex_num
    xm_per_pix = 3.7 / 700

    left_fit_cr = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)

    # calculate radisu of curvature
    left_eval = np.max(lefty)
    right_eval = np.max(righty)
    left_curverad = ((1 + (2 * left_fit_cr[0] * left_eval + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * right_eval + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])

    # calculate left_min by finding minimum value in first index of array
    left_min = np.amin(leftx, axis=0)
    print('left_min', left_min)
    right_max = np.amax(rightx, axis=0)
    print('right max', right_max)
    actual_center = (right_max + left_min) / 2
    dist_from_center = actual_center - int(x_piex_num / 2)
    print('pix dist from center', dist_from_center)

    meters_from_center = xm_per_pix * dist_from_center
    string_meters = str(round(meters_from_center, 2))

    full_text = 'left: ' + str(round(left_curverad, 2)) + ', right: ' + \
                str(round(right_curverad, 2)) + ', dist from center: ' + string_meters
    print('full text', full_text)

    if abs(left_curverad - right_curverad) < 5000 or not lane.curve['full_text']:
        # try without: \and right_max < 1100
        # dont remember what this does: and rightx.shape[0] > 100
        print('setting vals now')
        lane.curve['left_fitx'] = left_fitx
        lane.curve['lefty'] = lefty
        lane.curve['right_fitx'] = right_fitx
        lane.curve['righty'] = righty
        lane.curve['ploty'] = ploty
        lane.curve['full_text'] = full_text
    else:
        print('getting previous vals')
        left_fitx = lane.curve['left_fitx']
        lefty = lane.curve['lefty']
        right_fitx = lane.curve['right_fitx']
        righty = lane.curve['righty']
        ploty = lane.curve['ploty']
        full_text = lane.curve['full_text']

    return left_fitx, lefty, right_fitx, righty, ploty, full_text


def region_of_interest(img, vertices):
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


'''
given left and right lines values, add to original image
'''
def draw_on_road(img, warped, left_fitx, left_yvals, right_fitx, right_yvals, ploty):
    # create img to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # recast x and y into usable format for cv2.fillPoly
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    print('pts left', pts_left.shape, 'pts right', pts_right.shape)
    pts = np.hstack((pts_left, pts_right))

    # draw the lane onto the warped blank img
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    # plt.imshow(color_warp)
    # plt.title('color_warp')
    # plt.show()

    img_size = (img.shape[1], img.shape[0])
    dst, src = get_roi_mapping(img)

    # image = np.zeros(img_size, dtype=np.int8)
    # cv2.fillConvexPoly(image, src.astype('int32'), 1)
    # plt.imshow(image)
    # plt.title('lines')
    # plt.show()
    Minv = cv2.getPerspectiveTransform(src, dst)

    # warp the blank back oto the original image using inverse perspective matrix
    newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0]))

    # combine the result with the original
    result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)
    print('result shape', result.shape)
    # plt.imshow(result)
    # plt.title('lines')
    # plt.show()
    return result


def process_image(img):
    undist_img = undist(img, mtx, dist)
    # plt.imshow(undist_img)
    # plt.title('undist_img')
    # plt.show()

    # if want to perform mask, do it here
    # trapezoid, dts = get_roi_mapping(undist_img);
    # masked_image = region_of_interest(undist_img, [trapezoid.astype('int32')])
    # plt.imshow(masked_image, cmap='gray')
    # plt.title('masked_image')
    # plt.show()

    combo_image = combo_thresh(undist_img)
    # plt.imshow(combo_image, cmap='gray')
    # plt.title('combo_image')
    # plt.show()

    warped_image = change_perspective(combo_image)
    # plt.imshow(warped_image, cmap='gray')
    # plt.title('warped_image')
    # plt.show()

    left_fitx, lefty, right_fitx, righty, ploty, full_text = lr_curvature(warped_image)
    result = draw_on_road(img, warped_image, left_fitx, lefty, right_fitx, righty, ploty)
    cv2.putText(result, full_text, (200, 100), cv2.FONT_HERSHEY_COMPLEX, 1, 255)

    # sci.imsave('./output_images/5_final.jpg', result)
    return result


class Lane():
    def __init__(self):
        # if line was deteced in last iteration
        self.curve = {'full_text': ''}


if __name__ == '__main__':
    images = get_file_images('process_image')
    # show_images(images)
    lane = Lane()
    # set video variables
    proj_output = 'output3.mp4'
    clip1 = VideoFileClip('video_3.mp4')
    # clip1 = VideoFileClip('challenge_video.mp4')

    # run process image on each video clip and save to file
    output_clip = clip1.fl_image(process_image)
    output_clip.write_videofile(proj_output, audio=False)
    i = 0

    # thresh_images = threshold_all('test_images', process_image)
    # show_images(thresh_images)

    # image = mpimg.imread('test_images/straight_road_1x.jpg')
    for num in range(0, len(images)):
        colored_image = process_image(images[num])
        plt.imshow(colored_image)
        plt.title('colored_image' + str(num))
        plt.show()
        i = 0
