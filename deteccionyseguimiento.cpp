
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <sys/time.h>
#include <sys/types.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/background_segm.hpp>
#include <opencv2/video/tracking.hpp>

#define ROJO 		1
#define VERDE 		2
#define AMARILLO 	3

using namespace std;
using namespace cv;

std::string filename0 = "tiempos.csv";
std::string filename1 = "recVerde.csv";
std::string filename2 = "recRojo.csv";
std::string filename3 = "recAmarillo.csv";
ofstream outFile1/*, outFile2, outFile3*/;
ofstream fs;

struct timeval t1,t2,tf;
int fotogmax, fotogmin,fotogtot;
struct datosCoche{
	float cx,cy;
	float rx,ry;
	float rad;
	Mat maskout;
};
/*--------------------------------------------------------------------------*/
// Devuelve los centros del circulo y el cuadrado segmentado dado un color de coche
datosCoche buscaCoche(Mat frame, int color){

    cv:: Mat  kernel, mascara;
    mascara = cv::Mat::zeros(frame.rows, frame.cols, CV_8U);

    //segun el color pedido en la llamada a la funcion se hace una segmentacion distinta
	if(color == ROJO)
		cv::inRange(frame,Scalar(0,0,60),Scalar(90,80,255),mascara); //rojoBGR
	else if(color==VERDE)
		cv::inRange(frame,Scalar(0,75,0),Scalar(85,255,80),mascara); //verdeBGR
	else if(color==AMARILLO)
		cv::inRange(frame,Scalar(0,80,80),Scalar(80,255,255),mascara); //amarilloBGR


    // Realizamos un Opening para eliminar pequeños pixeles de ruido en la mascara


    kernel = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(5, 5));
    cv::erode(mascara, mascara, kernel);
    cv::dilate(mascara, mascara, kernel);

    //seguidamente otro doble con un dilate extra para obtener la mejor calidad de la mascara
    kernel = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(7, 7));
    cv::erode(mascara, mascara, kernel);
    cv::erode(mascara, mascara, kernel);

    cv::dilate(mascara, mascara, kernel);
    cv::dilate(mascara, mascara, kernel);
    cv::dilate(mascara, mascara, kernel);

    //umbralizo la mascara para encontrar los contornos correctamente
    int TBvalThresh = int(cv::threshold(mascara,mascara,120,255,CV_THRESH_OTSU));

    vector<vector<cv::Point> > contours;
    vector<Vec4i> hierarchy;


    //GaussianBlur(mascara,mascara,Size(5,5),2,2); //filtro gaussiano que no mejora los contornos
    //busqueda de contornos en la mascara
    findContours(mascara, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
    cv::Rect contoursRect;
    cv::RotatedRect contoursRRect;
    cv::Point pt1, pt2;
    cv::Point2f vertices2f[4];

    unsigned int circulo=0,rectangulo=0;
    float cx=0, cy=0, rx=0, ry=0, radio=0;
    double AR, A, W, L, difareas,R,Areacirc=0;

    //cv::drawContours(frame,contours,-1,CV_RGB(255,255,255),2); //para dibujar todos los contornos (debug)

    // para cada uno de los contornos
    for (unsigned int i=0; i<contours.size(); i++){

    	RotatedRect box;

    	contoursRect = cv::boundingRect(contours[i]);
    	box = minAreaRect( contours[i] );
    	A = contourArea(contours[i],true);

    	//se calculan los descriptores de cada contorno para distinguir circulos y rectangulos
    	W=box.size.width;
    	L=box.size.height;

    	AR=W/L;
    	difareas=1-abs(A)/(W*L);

    	R=W*L/abs(A);


    	if (AR < 1.25 && AR > 0.75) // compruebo que el rectangulo se aproxime a un cuadrado
    		{
    			if (difareas < 0.28 && difareas > 0.12)//compruebo que la diferencia de areas entre el cuadrado y el circulo tenga un valor aproximado al calculado teoricamente (dif de areas = area de cuadrado * 0.2146)
    				//es un circulo
    			{
    				if(abs(A) > 600)
    				{
    					//if(circulo=0 || abs(A)>Areacirc) //prueba para quedarme con el circulo mayor
    					{
    					circulo = 1;
    					Areacirc=abs(A);
    					cx = contoursRect.x + contoursRect.width/2;
    					cy = contoursRect.y + contoursRect.height/2;
    					radio = contoursRect.width/4 + contoursRect.height/4;
    					//para pintar el contorno del circulo (debug)
    					//circle(frame, Point(cx,cy), radio,CV_RGB(255,0,0),2 );
    					}
   				}
    			}
    		}
    	else
    	{
    	if(R<1.3 && abs(A) > 1200)
    	{
    		rectangulo=1;
    		rx = contoursRect.x + contoursRect.width/2;
    		ry = contoursRect.y + contoursRect.height/2;
        	contoursRRect = cv::minAreaRect(contours[i]);
        	//para pintar el contorno del rectangulo (debug)
        	/*contoursRRect.points(vertices2f);
        	line(frame,vertices2f[0],vertices2f[1],CV_RGB(0,0,255), 2);
        	line(frame,vertices2f[1],vertices2f[2],CV_RGB(0,0,255), 2);
        	line(frame,vertices2f[2],vertices2f[3],CV_RGB(0,0,255), 2);
        	line(frame,vertices2f[3],vertices2f[0],CV_RGB(0,0,255), 2);*/
    	}
    	}

    }

    //si se ha encontrado un circulo y un rectangulo adecuados se devuelven las coordenadas
    if(circulo  && rectangulo && abs(cx-rx)<150 && abs(cy-ry)<150)
    {
    	return {cx,cy,rx,ry,radio,mascara};
    }
    else//si no se devuelve 0
    {
    	return{0,0,0,0,0,mascara};
    }
}
/*--------------------------------------------------------------------------*/
bool rectContienePunto(Rect rectangulo,Point punto)
{
	if(punto.x+50>rectangulo.x && punto.x-50<rectangulo.x+rectangulo.width)
	{
		if(punto.y+50>rectangulo.y && punto.y-50<rectangulo.y+rectangulo.height)
		{
			return true;
		}
		else
			return false;
	}
	else
		return false;

}
/*--------------------------------------------------------------------------*/

double devuelveOrientacion(Point pt1, Point pt2)
{
	double ori;

	if(pt1.x<pt2.x)
	{
		if(pt1.y<pt2.y) //cuadrante 1  atan((cocheRojo.cy - cocheRojo.ry) / (cocheRojo.cx - cocheRojo.rx))*180/3.14;
		{
			ori = atan((float)(pt1.y - pt2.y) / (float)(pt1.x - pt2.x))*180/3.14;
		}
		else if(pt1.y>=pt2.y)//cuadrante 4
		{
			ori = (360+(atan((float)(pt1.y-pt2.y)/(float)(pt1.x-pt2.x))*180/3.14));
		}
	}
	else if(pt1.x>=pt2.x)
	{ori = (180+(atan((float)(pt1.y-pt2.y)/(float)(pt1.x-pt2.x))*180/3.14));
		/*
		if(pt1.y<pt2.y) //cuadrante 2
				{
					ori = (180+(atan((pt1.y-pt2.y)/(pt1.x-pt2.x))*180/3.14));
				}
				else if(pt1.y>=pt2.y)//cuadrante 3
				{
					ori = (180+(atan((pt1.y-pt2.y)/(pt1.x-pt2.x))*180/3.14));
				}*/
	}
	return ori;
}
/*--------------------------------------------------------------------------*/

// Initializes a Kalman filter and puts the initial values in the matrices used by the algorithm
cv::KalmanFilter initKalman(int x, int y, float sigmaR1, float sigmaQ1, float sigmaP){

    cv::KalmanFilter kf(4,2,0);

    kf.transitionMatrix = (cv::Mat_<float>(4,4) << 1,0,1,0,
    											   0,1,0,1,
												   0,0,1,0,
												   0,0,0,1);

    kf.measurementMatrix = (cv::Mat_<float>(2,4) << 1,0,0,0,
                                                    0,1,0,0);

    kf.processNoiseCov = (cv::Mat_<float>(4,4) << sigmaQ1,0,0,0,
                                                  0,sigmaQ1,0,0,
                                                  0,0,sigmaQ1,0,
                                                  0,0,0,sigmaQ1);

    kf.measurementNoiseCov = (cv::Mat_<float>(2,2) << sigmaR1,0,
                                                      0,sigmaR1);

    kf.errorCovPost = (cv::Mat_<float>(4,4) << sigmaP,0,0,0,
                                               0,sigmaP,0,0,
											   0,0,sigmaP,0,
                                               0,0,0,sigmaP);

    kf.statePost.at<float>(0) = x;
    kf.statePost.at<float>(1) = y;
    kf.statePost.at<float>(2) = 0;
    kf.statePost.at<float>(3) = 0;

    kf.statePre.at<float>(0) = x; //To control the initial values
    kf.statePre.at<float>(1) = y;

    return kf;

}

/*--------------------------------------------------------------------------*/

// Updates Kalman filter values, recalculates its prediction and corrects the measurement
void updateKalman(cv::KalmanFilter &kf, int x, int y, bool useMeasurement){

	cv::Mat prediction;
	    cv::Mat measurement(2,1,CV_32FC1);

	    prediction = kf.predict();


	    if (useMeasurement == false){

	        measurement.at<float>(0) = kf.statePre.at<float>(0);
	        measurement.at<float>(1) = kf.statePre.at<float>(1);

	    }else{

	        measurement.at<float>(0) = x;
	        measurement.at<float>(1) = y;

	    }


	  kf.correct(measurement);
}

/*--------------------------------------------------------------------------*/

// Returns the x, y, w, h values of Kalman filter prediction
cv::Point getKalmanPrediction(cv::KalmanFilter kf){
	return cv::Point(kf.statePre.at<float>(0),kf.statePre.at<float>(1));

}

/*--------------------------------------------------------------------------*/
int getDistance(cv::Point r1, cv::Point r2){


	int distancia = sqrt(abs(pow(r1.x,2)-pow(r2.x,2))+abs(pow(r1.y,2)-pow(r2.y,2)));
	return distancia;
}
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

    cv::Ptr<cv::BackgroundSubtractor> bgs; //MOG2 Background subtractor
    bgs = cv::createBackgroundSubtractorMOG2(100,200,false); //MOG2 approach

    cv::Mat frame, Mask, bgsMask;
    int fotog=0,flagV,flagR,flagA, detV=0, detR=0, detA=0, TTLV, TTLR, TTLA;
    int flagFilV=0, flagFilR=0, flagFilA=0;
    float oriV,oriR,oriA;
    cv::Point centroV,centroR,centroA, kalmanV, kalmanR, kalmanA,antCentroV,antCentroR,antCentroA;
    vector<cv::Point> recoV,recoR,recoA;
    vector<int> recoVcoord,recoRcoord,recoAcoord;
    vector<struct timeval> tfoto;
    cv::KalmanFilter filtroV, filtroR, filtroA;
    cv::Rect RectV, RectR, RectA;


    cv::namedWindow("Mask", cv::WINDOW_NORMAL);
    cv::namedWindow("bgsMask", cv::WINDOW_NORMAL);
    cv::namedWindow("Video", cv::WINDOW_NORMAL);

    char key = 0;
    //para cada frame del video
    while(key != 'q' && video.get(CV_CAP_PROP_POS_FRAMES) < video.get(CV_CAP_PROP_FRAME_COUNT)){

    	gettimeofday(&t1,NULL);
        // Read a frame
        video >> frame;

        // Check if the frame is correctly read
        if (frame.empty())
            break;

        fotog++;
        bgsMask = cv::Mat::zeros(frame.rows, frame.cols, CV_64F);

       /* if(fotog==30)
        bgs->apply(frame, bgsMask,0.999);
        else*/
        bgs->apply(frame, bgsMask,-1);

        cv:: Mat  kernel;
        kernel = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(5, 5));
        cv::erode(bgsMask, bgsMask, kernel);
        cv::dilate(bgsMask, bgsMask, kernel);

		int TBvalThresh = int(cv::threshold(bgsMask,bgsMask,120,255,CV_THRESH_OTSU));

        // Find the objects that appear in the background subtraction mask

        vector<vector<cv::Point> > contours;
        vector<Vec4i> hierarchy;
        findContours(bgsMask, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);



        datosCoche cocheRojo, cocheVerde, cocheAmarillo;

        //se llama a la funciona para cada color
        cocheRojo = buscaCoche(frame,ROJO);
        cocheVerde = buscaCoche(frame,VERDE);
        cocheAmarillo = buscaCoche(frame,AMARILLO);

        //si la funcion ha detectado circulo y rectangulo en cada color se actualiza el centro del coche detectado y su orientacion
        if(cocheRojo.cx>0 && cocheRojo.rx>0)
        {
        	centroR=Point(abs(cocheRojo.cx+cocheRojo.rx)/2,abs(cocheRojo.cy+cocheRojo.ry)/2);
        	flagR++;
        	detR++;
        	//oriR = atan((cocheRojo.cy - cocheRojo.ry) / (cocheRojo.cx - cocheRojo.rx))*180/3.14;
        	oriR = devuelveOrientacion(Point(cocheRojo.cx,cocheRojo.cy),Point(cocheRojo.rx,cocheRojo.ry));
        	//cv::arrowedLine(frame,Point(cocheRojo.cx,cocheRojo.cy),Point(cocheRojo.rx,cocheRojo.ry),CV_RGB(255,0,0), 2);
        }
        if(cocheVerde.cx>0 && cocheVerde.rx>0){
        	centroV=Point(abs(cocheVerde.cx+cocheVerde.rx)/2,abs(cocheVerde.cy+cocheVerde.ry)/2);
        	flagV++;
        	//if(!detV)
        	detV++;
        	//oriV = atan((cocheVerde.cy - cocheVerde.ry) / (cocheVerde.cx - cocheVerde.rx))*180/3.14;
        	oriV = devuelveOrientacion(Point(cocheVerde.cx,cocheVerde.cy),Point(cocheVerde.rx,cocheVerde.ry));
         	//cv::arrowedLine(frame,Point(cocheVerde.cx,cocheVerde.cy),Point(cocheVerde.rx,cocheVerde.ry),CV_RGB(0,255,0), 2);
        }
        if(cocheAmarillo.cx>0 && cocheAmarillo.rx>0){
        	centroA=Point(abs(cocheAmarillo.cx+cocheAmarillo.rx)/2,abs(cocheAmarillo.cy+cocheAmarillo.ry)/2);
        	flagA++;
        	detA++;
        	//oriA = atan((cocheAmarillo.cy - cocheAmarillo.ry) / (cocheAmarillo.cx - cocheAmarillo.rx))*180/3.14;
        	oriA = devuelveOrientacion(Point(cocheAmarillo.cx,cocheAmarillo.cy),Point(cocheAmarillo.rx,cocheAmarillo.ry));
        	//cv::arrowedLine(frame,Point(cocheAmarillo.cx,cocheAmarillo.cy),Point(cocheAmarillo.rx,cocheAmarillo.ry),CV_RGB(255,255,0), 2);
        }

        cv::Rect contoursRect;
        cv::Point pt1, pt2;

        //para cada contorno encontrado con BGS busca a qué coche pertenece, si le asocia un coche se actualzia el centro
        for (unsigned int i=0; i<contours.size(); i++)
        {

            contoursRect = cv::boundingRect(contours[i]);
            if (contoursRect.width*contoursRect.height > 12000 && contoursRect.width*contoursRect.height < 40000)
            {

            	//rectangle(frame, contoursRect, CV_RGB(255,255,255), 2);
            	if(rectContienePunto(contoursRect,centroV) || rectContienePunto(contoursRect,Point(cocheVerde.cx,cocheVerde.cy)) || rectContienePunto(contoursRect,Point(cocheVerde.rx,cocheVerde.ry)))
            	{
            		centroV=Point(contoursRect.x+contoursRect.width/2,contoursRect.y+contoursRect.height/2);
            		flagV++;
            		//if(!detV)
            			detV++;
            		RectV = contoursRect;
            		//cv::drawMarker(frame,centroV,CV_RGB(0,255,0),MARKER_CROSS,10,3);
            		//rectangle(frame, contoursRect, CV_RGB(0,255,0), 2);
            	}
            	else if(rectContienePunto(contoursRect,centroR) || rectContienePunto(contoursRect,Point(cocheRojo.cx,cocheRojo.cy)) || rectContienePunto(contoursRect,Point(cocheRojo.rx,cocheRojo.ry)))
            	{
            		centroR=Point(contoursRect.x+contoursRect.width/2,contoursRect.y+contoursRect.height/2);
            		flagR++;
            		detR++;
            		RectR = contoursRect;
            		//cv::drawMarker(frame,centroR,CV_RGB(255,0,0),MARKER_CROSS,10,3);
            		//rectangle(frame, contoursRect, CV_RGB(255,0,0), 2);
            	}
            	else if(rectContienePunto(contoursRect,centroA) || rectContienePunto(contoursRect,Point(cocheAmarillo.cx,cocheAmarillo.cy)) || rectContienePunto(contoursRect,Point(cocheAmarillo.rx,cocheAmarillo.ry)))
            	{
            		centroA=Point(contoursRect.x+contoursRect.width/2,contoursRect.y+contoursRect.height/2);
            		flagA++;
            		detA++;
            		RectA = contoursRect;
            		//cv::drawMarker(frame,centroA,CV_RGB(255,255,0),MARKER_CROSS,10,3);
            		//rectangle(frame, contoursRect, CV_RGB(255,255,0), 2);
            	}
            }
        }



        //la mascara que se muestra es la suma de las mascaras de todos los colores
        Mask = cocheVerde.maskout + cocheRojo.maskout + cocheAmarillo.maskout;

        //si se han actualizado los centros se marcan
        if(flagV)
        {
        	if(RectV.x>0 )
        	{
        		if(!flagFilV)
        		{
        			filtroV=initKalman(centroV.x,centroV.y,1,0.1,0.1);
        			flagFilV++;
        			TTLV=0;
        		}
        		else if(detV>2)
        		{
        			TTLV=0;
        			kalmanV=getKalmanPrediction(filtroV);
        			updateKalman(filtroV,centroV.x,centroV.y,true);
                	cv::drawMarker(frame,Point(kalmanV.x, kalmanV.y),CV_RGB(130,255,130),MARKER_DIAMOND,10,2);
                	//rectangle(frame, kalRectV, CV_RGB(0,0,0), 2);
        		}
        	}
        cv::drawMarker(frame,antCentroV,CV_RGB(0,255,0),MARKER_CROSS,10,3);
        flagV=0;
        }
        else if(!flagV && detV>2 && flagFilV)
        {

        	kalmanV=getKalmanPrediction(filtroV);
        	updateKalman(filtroV,kalmanV.x,kalmanV.y, false);

        	// si la prevision del kalman es coherente se actualiza el centro
        	if(abs(centroV.x-kalmanV.x)<100 && abs(centroV.y-kalmanV.y)<100 )
        	{
        		centroV=kalmanV;
            	cv::drawMarker(frame,Point(kalmanV.x, kalmanV.y),CV_RGB(0,150,0),MARKER_CROSS,10,3);
        	}
        	TTLV++;
        	if(TTLV>10)
        	{
        		detV=0;
        	}
        }

        if(flagR)
        {
        	if(RectR.x>0 )
        	{
        		if(!flagFilR)
        		{
        			filtroR=initKalman(centroR.x,centroR.y,1,0.1,0.1);
        			flagFilR++;
        			TTLR=0;
        		}
        		else if(detR>2)
        		{
        			TTLR=0;
        			kalmanR=getKalmanPrediction(filtroR);
        			updateKalman(filtroR,centroR.x,centroR.y,true);
                	cv::drawMarker(frame,Point(kalmanR.x, kalmanR.y),CV_RGB(255,130,130),MARKER_DIAMOND,10,2);
                	//rectangle(frame, kalRectV, CV_RGB(0,0,0), 2);
        		}
        	}
        cv::drawMarker(frame,antCentroR,CV_RGB(255,0,0),MARKER_CROSS,10,3);
        flagR=0;
        }
        else if(!flagR && detR>2 && flagFilR)
        {

        	kalmanR=getKalmanPrediction(filtroR);
        	updateKalman(filtroR,kalmanR.x,kalmanR.y, false);

        	// si la prevision del kalman es coherente se actualiza el centro
        	if(abs(centroR.x-kalmanR.x)<150 && abs(centroR.y-kalmanR.y)<150 )
        	{
        		centroR=kalmanR;
            	cv::drawMarker(frame,Point(kalmanR.x, kalmanR.y),CV_RGB(150,0,0),MARKER_CROSS,10,3);
        	}
        	TTLR++;
        	if(TTLR>10)
        	{
        		detR=0;
        	}
        }

        if(flagA)
        {
        	if(RectA.x>0 )
        	{
        		if(!flagFilA)
        		{
        			filtroA=initKalman(centroA.x,centroA.y,1,0.1,0.1);
        			flagFilA++;
        			TTLR=0;
        		}
        		else if(detA>2)
        		{
        			TTLA=0;
        			kalmanA=getKalmanPrediction(filtroA);
        			updateKalman(filtroA,centroA.x,centroA.y,true);
                	cv::drawMarker(frame,Point(kalmanA.x, kalmanA.y),CV_RGB(255,255,130),MARKER_DIAMOND,10,2);
                	//rectangle(frame, kalRectA, CV_RGB(0,0,0), 2);
        		}
        	}
        cv::drawMarker(frame,antCentroA,CV_RGB(255,255,0),MARKER_CROSS,10,3);
        flagA=0;
        }
        else if(!flagA && detA>2 && flagFilA)
        {

        	kalmanA=getKalmanPrediction(filtroA);
        	updateKalman(filtroA,kalmanA.x,kalmanA.y, false);

        	// si la prevision del kalman es coherente se actualiza el centro
        	if(abs(centroA.x-kalmanA.x)<150 && abs(centroA.y-kalmanA.y)<150 )
        	{
        		centroA=kalmanA;
            	cv::drawMarker(frame,Point(kalmanA.x, kalmanA.y),CV_RGB(150,150,0),MARKER_CROSS,10,3);
        	}
        	TTLA++;
        	if(TTLA>10)
        	{
        		detA=0;
        	}
        }

        //marcador de frames
        char str[200];
        sprintf(str,"frame: %d",(int)fotog);
        putText(frame, str, Point(10,30) , FONT_ITALIC, 0.75, CV_RGB(0,0,255), 2, 8, false );

        if(detV>2 && flagFilV && getDistance(centroV,kalmanV)<400)
        centroV=kalmanV;
        if(detR>2 && flagFilR && getDistance(antCentroR,kalmanR)<400)
        centroR=kalmanR;
        if(detA>2 && flagFilA && getDistance(centroA,kalmanA)<400)
        centroA=kalmanA;



        //pinta el ultimo centro conocido de cada coche detectado
        if(centroR.x>0)
        {
        recoR.push_back(antCentroR);
        recoRcoord.push_back(fotog);
        sprintf(str,"Rojo: ( %d, %d) %.1f deg",centroR.x, centroR.y, oriR);
        putText(frame, str, Point(centroR.x, centroR.y) , FONT_ITALIC, 0.75, CV_RGB(255,0,0), 2, 8, false );
        }

        if(centroA.x>0)
        {
        recoA.push_back(antCentroA);
        recoAcoord.push_back(fotog);
        sprintf(str,"Amarillo: ( %d, %d) %.1f deg",centroA.x, centroA.y, oriA);
        putText(frame, str, Point(centroA.x, centroA.y) , FONT_ITALIC, 0.75, CV_RGB(255,255,0), 2, 8, false );
        }

        if(centroV.x>0)
        {
        recoV.push_back(antCentroV);
        recoVcoord.push_back(fotog);
        sprintf(str,"Verde: ( %d, %d) %.1f deg",centroV.x, centroV.y, oriV);
        putText(frame, str, Point(centroV.x, centroV.y) , FONT_ITALIC, 0.75, CV_RGB(0,255,0), 2, 8, false );
        }

        //pinta el recorrido de cada coche detectado si la distancia entre puntos es menor de 300px en vertical u horizontal
        if(recoR.size()>2 && recoR[1].x>0)
        {
        	for(int i = 0; i < recoR.size()-1 ; i++)
        	{
        		if(abs(recoR[i].x-recoR[i+1].x)<300 && abs(recoR[i].y-recoR[i+1].y)<300)
        		cv::line(frame,recoR[i],recoR[i+1],CV_RGB(255,0,0), 2);
        	}
        }
        if(recoV.size()>2 && recoV[1].x>0)
        {
        	for(int i = 0; i < recoV.size()-1 ; i++)
        	{
        		if(abs(recoV[i].x-recoV[i+1].x)<300 && abs(recoV[i].y-recoV[i+1].y)<300)
        		cv::line(frame,recoV[i],recoV[i+1],CV_RGB(0,255,0), 2);
        	}
        }
        if(recoA.size()>2 && recoA[1].x>0)
        {
        	for(int i = 0; i < recoA.size()-1 ; i++)
        	{
        		if(abs(recoA[i].x-recoA[i+1].x)<300 && abs(recoA[i].y-recoA[i+1].y)<300)
        		cv::line(frame,recoA[i],recoA[i+1],CV_RGB(255,255,0), 2);
        	}
        }

        antCentroV=centroV;
        antCentroR=centroR;
        antCentroA=centroA;

        imshow("Mask", Mask);
        imshow("bgsMask", bgsMask);
        imshow("Video", frame);

        key = char(cv::waitKey(1));
        gettimeofday(&t2,NULL);
        tf.tv_usec=(t2.tv_usec - t1.tv_usec);
        tf.tv_sec=(t2.tv_sec - t1.tv_sec);
        tfoto.push_back(tf);

        t2.tv_usec=0;
        t1.tv_usec=0;
		t1.tv_sec=0;
		t2.tv_sec=0;
		tf.tv_usec=0;

		}
    // Close the video
    video.release();


    	fs.open(filename0, std::fstream::out);
    	for(int i = 0; i < recoV.size()-1 ; i++)
    	{
    		fs << i << "," << tfoto[i].tv_usec << "," << tfoto[i].tv_sec<< std::endl;
    	}
    	fs.close();


    if(recoV.size()>2 && recoV[1].x>0)
    {
    	fs.open(filename1, std::fstream::out);
    	fs << "X" << "," << "Y" <<  "," << "frame" << std::endl;
    	for(int i = 0; i < recoV.size()-1 ; i++)
    	{
    		fs << recoV[i].x << "," << recoV[i].y << "," << recoVcoord[i] << std::endl;
    	}
    	fs.close();
    }

    if(recoR.size()>2 && recoR[1].x>0)
    {
    	fs.open(filename2, std::fstream::out);
    	fs << "X" << "," << "Y" <<  "," << "frame" <<  std::endl;
    	for(int i = 0; i < recoR.size()-1 ; i++)
    	{
    		fs << recoR[i].x << "," << recoR[i].y << "," << recoRcoord[i] << std::endl;
    	}
    	fs.close();
    }

    if(recoA.size()>2 && recoA[1].x>0)
    {
    	fs.open(filename3, std::fstream::out);
    	fs << "X" << "," << "Y" <<  "," << "frame" <<  std::endl;
    	for(int i = 0; i < recoA.size()-1 ; i++)
    	{
    		fs << recoA[i].x << "," << recoA[i].y << "," << recoAcoord[i] << std::endl;
    	}
    	fs.close();
    }


}
