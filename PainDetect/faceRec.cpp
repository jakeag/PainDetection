// *****************************************************************************
// * Authors: Amanda Saenz and Jake Garcia
// * Date: Nov 2, 2017
// * Purpose: Create a webcam with facial recognition that checks if a person
// *          person is in pain (which are saved in a folder)
// *****************************************************************************

#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <string>
#include <iostream>
#include <stdio.h>
using namespace std;
using namespace cv;

//function declarations
void detectAndDisplay(Mat frame);

//Global variables
String face_cascade_name = "haarcascade_frontalface_alt.xml";
String smile_cascade_name = "haarcascade_smile.xml";
String pain_name = "pain.xml";
CascadeClassifier face_cascade;
CascadeClassifier smile_cascade;
CascadeClassifier pain_cascade;
String window_name = "Capture - Face detection";

int main(void)
{
    VideoCapture capture;
    Mat frame;
    
    //Load the cascades
    if(!face_cascade.load(face_cascade_name)){ printf("--(!)Error loading face cascade\n"); return -1; };
    if(!smile_cascade.load(smile_cascade_name)){ printf("--(!)Error loading smile cascade\n"); return -1; };
    if(!pain_cascade.load(pain_name)){ printf("--(!)Error loading pain cascade\n"); return -1; };
    
    cout << "*********************************************" << endl;
    cout << "*      V I D E O  S T R E A M  M E N U      *" << endl;
    cout << "*********************************************" << endl;
    cout << "* - remember to press and hold the buttons  *" << endl;
    cout << "*                                           *" << endl;
    cout << "* -- b -- gaussian blur                     *" << endl;
    cout << "* -- c -- canny                             *" << endl;
    cout << "* -- g -- gray                              *" << endl;
    cout << "* -- s -- shift colors                      *" << endl;
    cout << "* -- r -- remove colors                     *" << endl;
    cout << "* -- d -- 3D                                *" << endl;
    cout << "* -- f -- flip                              *" << endl;
    cout << "*********************************************" << endl;
    
    //Read the video stream
    capture.open(0); //Open default video stream
    if(!capture.isOpened()) //Check if successful
    {   
        capture.release(); 
        capture.open(-1); //Try another index if default fails
    }
    while(capture.read(frame))
    {
        if(frame.empty())
        {
            printf(" --(!) No captured frame -- Break!");
            break;
        }
        
        //Apply the classifier to the frame
        detectAndDisplay(frame);
        
        int c = waitKey(10);
        if( (char)c == 27 ) { break; } // escape
    }
    return 0;
}

void detectAndDisplay(Mat frame)
{
    string text = "Smile";
    string text2 = "pain";
    std::vector<Rect> faces;
    std::vector<Rect> pain;
    std::vector<Rect> smile;
    Mat frame_gray;
    Mat filter_frame;
    Mat faceROI;
    
    cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
    equalizeHist(frame_gray, frame_gray);
    

   
    //-------- Detect faces ---------
    face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0|CASCADE_SCALE_IMAGE, Size(30, 30));
    
    for(size_t i = 0; i < faces.size(); i++)
    {
        faceROI = frame_gray(faces[i]);
        rectangle(frame, faces[i], CV_RGB(0, 255,0), 1);

         //pain detection
        pain_cascade.detectMultiScale(frame_gray, pain, 1.1, 2, 0|CASCADE_SCALE_IMAGE, Size(250,250));
        for(size_t j1 = 0; j1 < pain.size(); j1++)
        {
            int pos_x1 = std::max(pain[j1].x+10, 0);
            int pos_y1 = std::max(pain[j1].y+10, 0);
            Point center( pain[i].x + pain[i].width/2, pain[i].y + pain[i].height/2 );
            ellipse( frame, center, Size( pain[i].width/2, pain[i].height/2), 0, 0, 360, Scalar( 255, 255,0 ), 4, 8, 0 );
            putText(frame, text2, Point(pos_x1, pos_y1), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0),1, 8, false);
        }
        
        //detect smiles
        smile_cascade.detectMultiScale(faceROI, smile, 1.1, 2, 0|CASCADE_SCALE_IMAGE, Size(70,70));
        for(size_t j = 0; j < smile.size(); j++)
        {
            int pos_x = std::max(smile[j].x+10, 0);
            int pos_y = std::max(smile[j].y+10, 0);
            //Point smile_center( faces[i].x + smile[j].x + smile[j].width/2, faces[i].y + smile[j].y + smile[j].height/2 );
//            int radius = cvRound( (smile[j].width + smile[j].height)*0.25 );
//            circle( frame, smile_center, radius, Scalar( 255, 0, 0 ), 4, 8, 0 );
            putText(frame, text, Point(pos_x, pos_y), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0),1, 8, false);
        }
        
        int choice = waitKey(10); //set choice to the user key input
        
        Mat filter_frame = frame.clone(); //stores camera frames
        switch (choice)
        {
            // -- b -- gaussian blur
            case 98:
            {
                GaussianBlur(filter_frame, filter_frame, Size(7,7), 1.5, 1.5);
                imshow(window_name, filter_frame);
                break;
            }
                
            // -- c -- canny
            case 99:
            {
                Canny(filter_frame, filter_frame, 0, 30, 3);
                imshow(window_name, filter_frame);
                break;
            }
                
            // -- g -- gray filter
            case 103:
            {
                cvtColor(filter_frame, filter_frame, CV_BGR2GRAY);
                imshow(window_name, filter_frame);
                break;
            }
                
            // -- s -- shifting colors
            case 115:
            {
                Mat channels[3];
                split(filter_frame, channels);
                channels[1].copyTo(filter_frame);
                channels[2].copyTo(channels[1]);
                filter_frame.copyTo(channels[2]);
                merge(channels, 3, filter_frame);
                
                channels[0].release();
                channels[1].release();
                channels[2].release();
                imshow(window_name, filter_frame);
                break;
            }
                
            // -- r -- remove colors
            case 114:
            {
                Mat channels[3];
                split(filter_frame, channels);
                channels[2] = Mat(filter_frame.rows, filter_frame.cols, CV_8UC1, Scalar(0));
                merge(channels, 3, filter_frame);
                
                channels[0].release();
                channels[1].release();
                channels[2].release();
                imshow(window_name, filter_frame);
                break;
            }
                
            // -- d -- 3D
            case 100:
            {
                Mat channels[3];
                Mat t;
                Point2f a[3];
                Point2f b[3];
                a[0] = Point2f(0,0);
                a[1] = Point2f(10,0);
                a[2] = Point2f(0,10);
                b[0] = Point2f(-10,0);
                b[1] = Point2f(0,0);
                b[2] = Point2f(-10,10);
                t = getAffineTransform(a, b);
                
                split(filter_frame, channels);
                warpAffine(channels[2], filter_frame, t, channels[2].size());
                filter_frame.copyTo(channels[2]);
                merge(channels, 3, filter_frame);
                
                t.release();
                channels[0].release();
                channels[1].release();
                channels[2].release();
                imshow(window_name, filter_frame);
                break;
            }
                
            // -- f -- flip
            case 102:
            {
                flip(filter_frame, filter_frame, -1);
                imshow(window_name, filter_frame);
                break;
            }
                
            default:
                imshow(window_name, frame);
                break;
        }

     }
}