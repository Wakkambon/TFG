
#include <iostream>
#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/background_segm.hpp>
#include <opencv2/video/tracking.hpp>

using namespace std;
using namespace cv;

const int max_value = 255;
const String window_capture_name = "Video Capture";
const String window_detection_name = "Opciones";
int low_B = 0, low_G = 0, low_R = 0, low_areac=200;
int high_B = max_value, high_G = max_value, high_R = max_value, high_areac = 1500;
int areacirc=600, arearect=1200;
int high_difareas= 40, low_difareas=5, low_AR=60, high_AR=150, valR = 120;

static void on_low_B_thresh_trackbar(int, void *)
{
    low_B = min(high_B-1, low_B);
    setTrackbarPos("Low B", window_detection_name, low_B);
}
static void on_high_B_thresh_trackbar(int, void *)
{
    high_B = max(high_B, low_B+1);
    setTrackbarPos("High B", window_detection_name, high_B);
}
static void on_low_G_thresh_trackbar(int, void *)
{
    low_G = min(high_G-1, low_G);
    setTrackbarPos("Low G", window_detection_name, low_G);
}
static void on_high_G_thresh_trackbar(int, void *)
{
    high_G = max(high_G, low_G+1);
    setTrackbarPos("High G", window_detection_name, high_G);
}
static void on_low_R_thresh_trackbar(int, void *)
{
    low_R = min(high_R-1, low_R);
    setTrackbarPos("Low R", window_detection_name, low_R);
}
static void on_high_R_thresh_trackbar(int, void *)
{
    high_R = max(high_R, low_R+1);
    setTrackbarPos("High R", window_detection_name, high_R);
}
static void on_high_difareas_thresh_trackbar(int, void *)
{
	high_difareas = max(high_difareas, low_difareas+1);
    setTrackbarPos("DifareasH", window_detection_name, high_difareas);
}
static void on_low_difareas_thresh_trackbar(int, void *)
{
    low_difareas = min(high_difareas-1, low_difareas);
    setTrackbarPos("DifareasL", window_detection_name, low_difareas);
}
static void on_high_AR_thresh_trackbar(int, void *)
{
	high_AR = max(high_AR, low_AR+1);
    setTrackbarPos("ARhigh", window_detection_name, high_AR);
}
static void on_low_AR_thresh_trackbar(int, void *)
{
    low_AR = min(high_AR-1, low_AR);
    setTrackbarPos("ARlow", window_detection_name, low_difareas);
}
static void rectangularidad_trackbar(int, void *)
{
    setTrackbarPos("rectangularidad", window_detection_name, valR);
}
static void areacirc_trackbar(int, void *)
{
    setTrackbarPos("Area Circulo", window_detection_name, areacirc);
}
static void arearect_trackbar(int, void *)
{
    setTrackbarPos("Area Rectangulo", window_detection_name, arearect);
}
/*--------------------------------------------------------------------------*/

// Returns the distance in pixels between the centers of two rectangles
int getCentersDistance(cv::Rect r1, cv::Rect r2){

    /* Add your code here for implementing this functionality. You must complete the body of this function */
	cv::Point centro1, centro2;
	centro1 = Point2f( (r1.x + r1.width / 2) , (r1.y + r1.height / 2) );
	centro2 = Point2f( (r2.x + r2.width / 2) , (r2.y + r2.height / 2) );

	int distancia = sqrt(abs(pow(centro1.x,2)-pow(centro2.x,2))+abs(pow(centro1.y,2)-pow(centro2.y,2)));
	return distancia;
}

/*--------------------------------------------------------------------------*/

// Main program
int main( int argc, char *argv[] ){


    // Open the video for processing it
    cv::VideoCapture video;
    video.open("videos/video8.mp4");

    // Check if the video is correctly opened
    if (!video.isOpened()){

        cout << "Can not open the video file" << endl;
        return -1;

    }

    cv::Mat frame, Mask, trackingFrame, hsv;
    cv::Mat up_amarillo,down_amarillo;
    cv::namedWindow("Video", cv::WINDOW_NORMAL);
    cv::namedWindow("Opciones", cv::WINDOW_NORMAL);
    //cv::namedWindow("Mask", cv::WINDOW_NORMAL);
    cv::namedWindow("Out", cv::WINDOW_NORMAL);

	//cv::namedWindow("Blue Channel", cv::WINDOW_NORMAL);
	//cv::namedWindow("Green Channel", cv::WINDOW_NORMAL);
	//cv::namedWindow("Red Channel", cv::WINDOW_NORMAL);

    // Trackbars to set thresholds for HSV values
    createTrackbar("Low B", window_detection_name, &low_B, max_value, on_low_B_thresh_trackbar);
    createTrackbar("High B", window_detection_name, &high_B, max_value, on_high_B_thresh_trackbar);
    createTrackbar("Low G", window_detection_name, &low_G, max_value, on_low_G_thresh_trackbar);
    createTrackbar("High G", window_detection_name, &high_G, max_value, on_high_G_thresh_trackbar);
    createTrackbar("Low R", window_detection_name, &low_R, max_value, on_low_R_thresh_trackbar);
    createTrackbar("High R", window_detection_name, &high_R, max_value, on_high_R_thresh_trackbar);
    createTrackbar("ARLow", window_detection_name, &low_AR, 200, on_low_AR_thresh_trackbar);
    createTrackbar("ARHigh", window_detection_name, &high_AR, 200, on_high_AR_thresh_trackbar);
    createTrackbar("DifareasL", window_detection_name, &low_difareas, 60, on_low_difareas_thresh_trackbar);
    createTrackbar("DifareasH", window_detection_name, &high_difareas, 60, on_high_difareas_thresh_trackbar);
    createTrackbar("Area Circulo", window_detection_name, &areacirc, 1500, areacirc_trackbar);
    createTrackbar("rectangularidad", window_detection_name, &valR, 200, rectangularidad_trackbar);
    createTrackbar("Area Rectangulo", window_detection_name, &arearect, 2000, arearect_trackbar);

    char key = 0;
    //para cada frame del video
    while(key != 'q' && video.get(CV_CAP_PROP_POS_FRAMES) < video.get(CV_CAP_PROP_FRAME_COUNT)){

        // Read a frame
        video >> frame;

        // Check if the frame is correctly read
        if (frame.empty())
            break;

        Mask = cv::Mat::zeros(frame.rows, frame.cols, CV_8U);
        hsv = cv::Mat::zeros(frame.rows, frame.cols, CV_8U);

    	//cv::Mat bgr[3];
    	//cv::split(frame,bgr);

    	//cv::imshow("Blue Channel", bgr[0]);
    	//cv::imshow("Green Channel", bgr[1]);
    	//cv::imshow("Red Channel", bgr[2]);

    	//cv::equalizeHist(bgr[0],bgr[0]);
    	//cv::equalizeHist(bgr[1],bgr[1]);
    	//cv::equalizeHist(bgr[2],bgr[2]);

    	//cv::merge(bgr,3,hsv);


        // Resize the frame for a better visualization
       // cv::resize(frame, frame, cv::Size(frame.cols*2, frame.rows*2), cv::INTER_LINEAR);


        //cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);

        //up_amarillo = cv::Mat::zeros(3,1,CV_8U);
        //down_amarillo = cv::Mat::zeros(3,1,CV_8U);

       // up_amarillo = (200,90,70);
		//down_amarillo = (20,50,30);

		//cv::inRange(hsv,Scalar(40,5,5),Scalar(70,95,95),Mask);
        cv::inRange(frame,Scalar(low_B, low_G, low_R),Scalar(high_B, high_G, high_R), Mask);
		//cv::inRange(frame,Scalar(0,70,0),Scalar(80,255,80),Mask); //verdeBGR
        //cv::inRange(frame,Scalar(0,0,60),Scalar(90,70,255),Mask); //rojoBGR
        //cv::inRange(frame,Scalar(0,70,70),Scalar(80,255,255),Mask); //amarilloBGR


        // Realizamos un Opening para eliminar pequeños pixeles de ruido en la mascara
        cv:: Mat  kernel;

        kernel = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(5, 5));
        cv::erode(Mask, Mask, kernel);
        cv::dilate(Mask, Mask, kernel);

        kernel = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(7, 7));
        cv::erode(Mask, Mask, kernel);
        cv::erode(Mask, Mask, kernel);

        cv::dilate(Mask, Mask, kernel);
        cv::dilate(Mask, Mask, kernel);
        cv::dilate(Mask, Mask, kernel);

       // kernel = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(7, 7));
      //  cv::dilate(Mask, Mask, kernel);
   //     kernel = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(9, 9));
    //    cv::erode(Mask, Mask, kernel);
    //    cv::dilate(Mask, Mask, kernel);

 //añado

        int TBvalThresh = int(cv::threshold(Mask,Mask,120,255,CV_THRESH_OTSU));

        // Find the objects that appear in the background subtraction mask
                vector<vector<cv::Point> > contours;
                /* Add your code here for implementing this functionality. Use cv::findContours */
                //busco los contornos en la imagen
                vector<Vec4i> hierarchy;

               // GaussianBlur(Mask,Mask,Size(5,5),2,2);
                findContours(Mask, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

                vector<Vec3f> circles;


                //cv::HoughCircles(Mask,circles, CV_HOUGH_GRADIENT,1.5, Mask.rows/2);
                // Manage the Kalman filters and draw it jointly with their numerical identifier (in blue)
                // We draw the objects if there is someone and it has an acceptable size
                //frame.copyTo(trackingFrame);
                cv::Rect contoursRect;
                cv::RotatedRect contoursRRect;
                cv::Point pt1, pt2;
                cv::Point2f vertices2f[4];

                //for(size_t j = 0 ; j < circles.size(); j++){
                	//Point center((circles[j][0]),(circles[j][1]));
                	//int radius = (circles[j][2]);
                	//circle(Mask,center,radius,CV_RGB(255,0,0),2);
                	//Vec3i c = circles[j];
                	//circle(frame, Point(c[0], c[1]), c[2], Scalar(0,0,255), 3, CV_AA);
                //}

                //cout << circles[0][2] << endl;

                unsigned int circuloverde=0,rectanguloverde=0;
                float cxcircv, cycircv, cxrectv, cyrectv, radiocircv;
                double AR, A, W, L, difareas,R;

                // para cada uno de los contornos
                for (unsigned int i=0; i<contours.size(); i++){

                	RotatedRect box;

                	contoursRect = cv::boundingRect(contours[i]);
                	box = minAreaRect( contours[i] );
                	A = contourArea(contours[i],true);

                	W=box.size.width;
                	L=box.size.height;

                	AR=W/L;
                	difareas=1-abs(A)/(W*L);

                	R=W*L/abs(A);


                	if (abs(A) > areacirc && abs(A) < arearect) // compruebo que el rectangulo se aproxime a un cuadrado
                		{
                			if (difareas < float(float(high_difareas)/100) && difareas > float(float(low_difareas)/100))//compruebo que la diferencia de areas entre el cuadrado y el circulo tenga un valor aproximado al calculado teoricamente (dif de areas = area de cuadrado * 0.2146)
                				//es un circulo
                			{
                				if(AR < float(float(high_AR)/100) && AR > float(float(low_AR)/100))
                				{
                					circuloverde = i;
                					cxcircv = contoursRect.x + contoursRect.width/2;
                					cycircv = contoursRect.y + contoursRect.height/2;
                					radiocircv = contoursRect.width/4 + contoursRect.height/4;
                					//cv::drawContours(frame,contours,i,CV_RGB(255,0,0),2);
                				}
                			}
                		}
                	else
                	if(R<float(float(valR)/100) && abs(A) > arearect)
                	{
                		rectanguloverde=i;
                		cxrectv = contoursRect.x + contoursRect.width/2;
                		cyrectv = contoursRect.y + contoursRect.height/2;
                	}


                }



        cv::drawContours(frame,contours,-1,CV_RGB(255,255,255),2);
        if(circuloverde>0)
        {
        	circle(frame, Point(cxcircv,cycircv), radiocircv,CV_RGB(255,0,0),2 );
        	//cv::drawContours(frame,contours,circuloverde,CV_RGB(255,0,0),2);
        }
        if(rectanguloverde>0)
        {
        	contoursRRect = cv::minAreaRect(contours[rectanguloverde]);
        	contoursRRect.points(vertices2f);
        	line(frame,vertices2f[0],vertices2f[1],CV_RGB(0,0,255), 2);
        	line(frame,vertices2f[1],vertices2f[2],CV_RGB(0,0,255), 2);
        	line(frame,vertices2f[2],vertices2f[3],CV_RGB(0,0,255), 2);
        	line(frame,vertices2f[3],vertices2f[0],CV_RGB(0,0,255), 2);
        	//cv::drawContours(frame,contours,rectanguloverde,CV_RGB(0,0,255),2);
        }

        if(circuloverde>0 && rectanguloverde>0)
        cv::arrowedLine(frame,Point(cxcircv,cycircv),Point(cxrectv,cyrectv),CV_RGB(0,255,0), 2);


        // Show background subtraction results
        imshow("Video", frame);
        //imshow("Mask", hsv);
        imshow("Out", Mask);

        // Wait before read the next frame
       // cout << "Press any key for reading the next frame (" << video.get(CV_CAP_PROP_POS_FRAMES) << "/" << video.get(CV_CAP_PROP_FRAME_COUNT) << ")" << endl;
        key = char(cv::waitKey(1000));

		}
    // Close the video
    video.release();
}
