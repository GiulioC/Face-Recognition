/*****************************************************************************
*   Face Recognition using Eigenfaces or Fisherfaces
******************************************************************************
*   by Shervin Emami, 5th Dec 2012
*   http://www.shervinemami.info/openCV.html
******************************************************************************
*   Ch8 of the book "Mastering OpenCV with Practical Computer Vision Projects"
*   Copyright Packt Publishing 2012.
*   http://www.packtpub.com/cool-projects-with-opencv/book
*****************************************************************************/

//////////////////////////////////////////////////////////////////////////////////////
// WebcamFaceRec.cpp, by Shervin Emami (www.shervinemami.info) on 30th May 2012.
// Face Detection & Face Recognition from a webcam using LBP and Eigenfaces or Fisherfaces.
//////////////////////////////////////////////////////////////////////////////////////
//
// Some parts are based on the tutorial & code by Robin Hewitt (2007) at:
// "http://www.cognotics.com/opencv/servo_2007_series/part_5/index.html"
//
// Some parts are based on the tutorial & code by Shervin Emami (2009) at:
// "http://www.shervinemami.info/faceRecognition.html"
//
// Requires OpenCV v2.4.1 or later (from June 2012), otherwise the FaceRecognizer will not compile or run.
//
//////////////////////////////////////////////////////////////////////////////////////


// The Face Recognition algorithm can be one of these and perhaps more, depending on your version of OpenCV, which must be atleast v2.4.1:
//    "FaceRecognizer.Eigenfaces":  Eigenfaces, also referred to as PCA (Turk and Pentland, 1991).
//    "FaceRecognizer.Fisherfaces": Fisherfaces, also referred to as LDA (Belhumeur et al, 1997).
//    "FaceRecognizer.LBPH":        Local Binary Pattern Histograms (Ahonen et al, 2006).
const char *facerecAlgorithm = "FaceRecognizer.Fisherfaces";
//const char *facerecAlgorithm = "FaceRecognizer.Eigenfaces";

const char *imgFolder = "./immagini/";

int saveManuallyClicked = 0;


// Sets how confident the Face Verification algorithm should be to decide if it is an unknown person or a known person.
// A value roughly around 0.5 seems OK for Eigenfaces or 0.7 for Fisherfaces, but you may want to adjust it for your
// conditions, and if you use a different Face Recognition algorithm.
// Note that a higher threshold value means accepting more faces as known people,
// whereas lower values mean more faces will be classified as "unknown".
const float UNKNOWN_PERSON_THRESHOLD = 0.7f;


// Cascade Classifier file, used for Face Detection.
const char *faceCascadeFilename = "lbpcascade_frontalface.xml";     // LBP face detector.
//const char *faceCascadeFilename = "haarcascade_frontalface_alt_tree.xml";  // Haar face detector.
//const char *eyeCascadeFilename1 = "haarcascade_lefteye_2splits.xml";   // Best eye detector for open-or-closed eyes.
//const char *eyeCascadeFilename2 = "haarcascade_righteye_2splits.xml";   // Best eye detector for open-or-closed eyes.
//const char *eyeCascadeFilename1 = "haarcascade_mcs_lefteye.xml";       // Good eye detector for open-or-closed eyes.
//const char *eyeCascadeFilename2 = "haarcascade_mcs_righteye.xml";       // Good eye detector for open-or-closed eyes.
const char *eyeCascadeFilename1 = "haarcascade_eye.xml";               // Basic eye detector for open eyes only.
const char *eyeCascadeFilename2 = "haarcascade_eye_tree_eyeglasses.xml"; // Basic eye detector for open eyes if they might wear glasses.


// Set the desired face dimensions. Note that "getPreprocessedFace()" will return a square face.
const int faceWidth = 70;
const int faceHeight = faceWidth;

// Try to set the camera resolution. Note that this only works for some cameras on
// some computers and only for some drivers, so don't rely on it to work!
const int DESIRED_CAMERA_WIDTH = 640;
const int DESIRED_CAMERA_HEIGHT = 480;

// Parameters controlling how often to keep new faces when collecting them. Otherwise, the training set could look to similar to each other!
const double CHANGE_IN_IMAGE_FOR_COLLECTION = 0.3;      // How much the facial image should change before collecting a new face photo for training.
const double CHANGE_IN_SECONDS_FOR_COLLECTION = 1.0;       // How much time must pass before collecting a new face photo for training.

const char *windowName = "DATABASE CREATION APPLICATION";   // Name shown in the GUI window.
const int BORDER = 8;  // Border between GUI elements to the edge of the image.

const bool preprocessLeftAndRightSeparately = true;   // Preprocess left & right sides of the face separately, in case there is stronger light on one side.

// Set to true if you want to see many windows created, showing various debug info. Set to 0 otherwise.
bool m_debug = false;


#include <stdio.h>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>

// Include OpenCV's C++ Interface
#include "opencv2/opencv.hpp"

// Include the rest of our code!
#include "detectObject.h"       // Easily detect faces or eyes (using LBP or Haar Cascades).
#include "preprocessFace.h"     // Easily preprocess face images, for face recognition.
#include "recognition.h"     // Train the face recognition system and recognize a person from an image.

#include "ImageUtils.h"      // Shervin's handy OpenCV utility functions.
//#include "C:\Windows\System32\kernel32.dll"

using namespace cv;
using namespace std;


#if !defined VK_ESCAPE
    #define VK_ESCAPE 0x1B      // Escape character (27)
#endif

FILE * (file);
string name, surname;
int currentPerson = 0;
vector <string> identities;


// Running mode for the Webcam-based interactive GUI program.
enum MODES { MODE_STARTUP = 0, MODE_DETECTION, MODE_SAVE_IMAGES, MODE_COLLECT_FACES, MODE_DELETE_ALL, MODE_CREATE_DATABASE, MODE_COLLECT_FACES_MANUALLY };
const char* MODE_NAMES[] = {"Startup", "Detection", "Collect Faces", "Training", "Recognition", "Delete All", "ERROR!"}; //nomi che vengono visualizzati sulla GUI
MODES m_mode = MODE_STARTUP;

int m_selectedPerson = -1;
int m_numPersons = 0;
vector<int> m_latestFaces;

// Position of GUI buttons:
Rect m_rcBtnAdd;
Rect m_rcBtnDel;
Rect m_rcBtnDebug;
Rect m_rcBtnSaveFaces;
Rect m_rcBtnSaveOneFace;
Rect m_rcBtnAddManually;
int m_gui_faces_left = -1;
int m_gui_faces_top = -1;



// C++ conversion functions between integers (or floats) to std::string.
template <typename T> string toString(T t)
{
    ostringstream out;
    out << t;
    return out.str();
}

template <typename T> T fromString(string t)
{
    T out;
    istringstream in(t);
    in >> out;
    return out;
}

// Load the face and 1 or 2 eye detection XML classifiers.
void initDetectors(CascadeClassifier &faceCascade, CascadeClassifier &eyeCascade1, CascadeClassifier &eyeCascade2)
{
    // Load the Face Detection cascade classifier xml file.
    try {   // Surround the OpenCV call by a try/catch block so we can give a useful error message!
        faceCascade.load(faceCascadeFilename);
    } catch (cv::Exception &e) {}
    if ( faceCascade.empty() ) {
        cerr << "ERROR: Could not load Face Detection cascade classifier [" << faceCascadeFilename << "]!" << endl;
        cerr << "Copy the file from your OpenCV data folder (eg: 'C:\\OpenCV\\data\\lbpcascades') into this WebcamFaceRec folder." << endl;
        exit(1);
    }
    cout << "Loaded the Face Detection cascade classifier [" << faceCascadeFilename << "]." << endl;

    // Load the Eye Detection cascade classifier xml file.
    try {   // Surround the OpenCV call by a try/catch block so we can give a useful error message!
        eyeCascade1.load(eyeCascadeFilename1);
    } catch (cv::Exception &e) {}
    if ( eyeCascade1.empty() ) {
        cerr << "ERROR: Could not load 1st Eye Detection cascade classifier [" << eyeCascadeFilename1 << "]!" << endl;
        cerr << "Copy the file from your OpenCV data folder (eg: 'C:\\OpenCV\\data\\haarcascades') into this WebcamFaceRec folder." << endl;
        exit(1);
    }
    cout << "Loaded the 1st Eye Detection cascade classifier [" << eyeCascadeFilename1 << "]." << endl;

    // Load the Eye Detection cascade classifier xml file.
    try {   // Surround the OpenCV call by a try/catch block so we can give a useful error message!
        eyeCascade2.load(eyeCascadeFilename2);
    } catch (cv::Exception &e) {}
    if ( eyeCascade2.empty() ) {
        cerr << "Could not load 2nd Eye Detection cascade classifier [" << eyeCascadeFilename2 << "]." << endl;
        // Dont exit if the 2nd eye detector did not load, because we have the 1st eye detector at least.
        //exit(1);
    }
    else
        cout << "Loaded the 2nd Eye Detection cascade classifier [" << eyeCascadeFilename2 << "]." << endl;
}

// Get access to the webcam.
void initWebcam(VideoCapture &videoCapture, int cameraNumber)
{
    // Get access to the default camera.
    try {   // Surround the OpenCV call by a try/catch block so we can give a useful error message!
        videoCapture.open(cameraNumber);
    } catch (cv::Exception &e) {}
    if ( !videoCapture.isOpened() ) {
        cerr << "ERROR: Could not access the camera!" << endl;
        exit(1);
    }
    cout << "Loaded camera " << cameraNumber << "." << endl;
}


// Draw text into an image. Defaults to top-left-justified text, but you can give negative x coords for right-justified text,
// and/or negative y coords for bottom-justified text.
// Returns the bounding rect around the drawn text.
Rect drawString(Mat img, string text, Point coord, Scalar color, float fontScale = 0.6f, int thickness = 1, int fontFace = FONT_HERSHEY_COMPLEX)
{
    // Get the text size & baseline.
    int baseline=0;
    Size textSize = getTextSize(text, fontFace, fontScale, thickness, &baseline);
    baseline += thickness;

    // Adjust the coords for left/right-justified or top/bottom-justified.
    if (coord.y >= 0) {
        // Coordinates are for the top-left corner of the text from the top-left of the image, so move down by one row.
        coord.y += textSize.height;
    }
    else {
        // Coordinates are for the bottom-left corner of the text from the bottom-left of the image, so come up from the bottom.
        coord.y += img.rows - baseline + 1;
    }
    // Become right-justified if desired.
    if (coord.x < 0) {
        coord.x += img.cols - textSize.width + 1;
    }

    // Get the bounding box around the text.
    Rect boundingRect = Rect(coord.x, coord.y - textSize.height, textSize.width, baseline + textSize.height);

    // Draw anti-aliased text.
    putText(img, text, coord, fontFace, fontScale, color, thickness, CV_AA);

    // Let the user know how big their text is, in case they want to arrange things.
    return boundingRect;
}

// Draw a GUI button into the image, using drawString().
// Can specify a minWidth if you want several buttons to all have the same width.
// Returns the bounding rect around the drawn button, allowing you to position buttons next to each other.
Rect drawButton(Mat img, string text, Point coord, int minWidth = 160)
{
    int B = BORDER;
    Point textCoord = Point(coord.x + B, coord.y + B);
    // Get the bounding box around the text.
    Rect rcText = drawString(img, text, textCoord, CV_RGB(0,0,0));
    // Draw a filled rectangle around the text.
    Rect rcButton = Rect(rcText.x - B, rcText.y - B, rcText.width + 2*B, rcText.height + 2*B);
    // Set a minimum button width.
    if (rcButton.width < minWidth)
        rcButton.width = minWidth;
    // Make a semi-transparent white rectangle.
    Mat matButton = img(rcButton);
    matButton += CV_RGB(90, 90, 90);
    // Draw a non-transparent white border.
    rectangle(img, rcButton, CV_RGB(200,200,200), 1, CV_AA);

    // Draw the actual text that will be displayed, using anti-aliasing.
    drawString(img, text, textCoord, CV_RGB(10,55,20));

    return rcButton;
}

bool isPointInRect(const Point pt, const Rect rc)
{
    if (pt.x >= rc.x && pt.x <= (rc.x + rc.width - 1))
        if (pt.y >= rc.y && pt.y <= (rc.y + rc.height - 1))
            return true;

    return false;
}

// Mouse event handler. Called automatically by OpenCV when the user clicks in the GUI window.
void onMouse(int event, int x, int y, int, void*)
{
    // We only care about left-mouse clicks, not right-mouse clicks or mouse movement.
    if (event != CV_EVENT_LBUTTONDOWN)
        return;

    // Check if the user clicked on one of our GUI buttons.
    Point pt = Point(x,y);
    if (isPointInRect(pt, m_rcBtnAdd)) {
        cout << "User clicked [Add Person] button when numPersons was " << m_numPersons << endl;
        // Check if there is already a person without any collected faces, then use that person instead.
        // This can be checked by seeing if an image exists in their "latest collected face".
        if ((m_numPersons == 0) || (m_latestFaces[m_numPersons-1] >= 0)) {
            // Add a new person.
            m_numPersons++;
            m_latestFaces.push_back(-1); // Allocate space for an extra person.
            cout << "Num Persons: " << m_numPersons << endl;
        }
        // Use the newly added person. Also use the newest person even if that person was empty.
        m_selectedPerson = m_numPersons - 1;
        m_mode = MODE_COLLECT_FACES;
    }
    else if (isPointInRect(pt, m_rcBtnDel)) {
        cout << "User clicked [Delete All] button." << endl;
        m_mode = MODE_DELETE_ALL;
    }
    else if (isPointInRect(pt, m_rcBtnDebug)) {
        cout << "User clicked [Debug] button." << endl;
		m_mode = MODE_SAVE_IMAGES;
        //m_debug = !m_debug;
        //cout << "Debug mode: " << m_debug << endl;
    }
	else if (isPointInRect(pt, m_rcBtnSaveFaces)) {
		cout << "User clicked [Save Faces] button." << endl;
		m_mode = MODE_CREATE_DATABASE;
	}
	else if (isPointInRect(pt, m_rcBtnSaveOneFace)) {
		cout << "User clicked [Add 1 Face] button." << endl;
		//m_mode = MODE_SAVE_ONE_FACE;
		saveManuallyClicked = 1;
	}
	else if (isPointInRect(pt, m_rcBtnAddManually)) {
		cout << "User clicked [Add Person Manually] button when numPersons was " << m_numPersons << endl;
		if ((m_numPersons == 0) || (m_latestFaces[m_numPersons - 1] >= 0)) {
			// Add a new person.
			m_numPersons++;
			m_latestFaces.push_back(-1); // Allocate space for an extra person.
			cout << "Num Persons: " << m_numPersons << endl;
		}
		m_selectedPerson = m_numPersons - 1;
		m_mode = MODE_COLLECT_FACES_MANUALLY;
	}
}


// Main loop that runs forever, until the user hits Escape to quit.
void recognizeAndTrainUsingWebcam(VideoCapture &videoCapture, CascadeClassifier &faceCascade, CascadeClassifier &eyeCascade1, CascadeClassifier &eyeCascade2, int val)
{
    Ptr<FaceRecognizer> model;
    vector<Mat> preprocessedFaces;
    vector<int> faceLabels;
    Mat old_prepreprocessedFace;
    double old_time = 0;

    // Since we have already initialized everything, lets start in Detection mode.
    m_mode = MODE_DETECTION;


    // Run forever, until the user hits Escape to "break" out of this loop.
    while (true) {


		//cout << "Brigthness: " << videoCapture.get(CV_CAP_PROP_BRIGHTNESS) << endl;

        // Grab the next camera frame. Note that you can't modify camera frames.
        Mat cameraFrame;
        videoCapture >> cameraFrame;
        if( cameraFrame.empty() ) {
            cerr << "ERROR: Couldn't grab the next camera frame." << endl;
            exit(1);
        }

        // Get a copy of the camera frame that we can draw onto.
        Mat displayedFrame;
        cameraFrame.copyTo(displayedFrame);

        // Run the face recognition system on the camera image. It will draw some things onto the given image, so make sure it is not read-only memory!
        int identity = -1;

        // Find a face and preprocess it to have a standard size and contrast & brightness.
        Rect faceRect;  // Position of detected face.
        Rect searchedLeftEye, searchedRightEye; // top-left and top-right regions of the face, where eyes were searched.
        Point leftEye, rightEye;    // Position of the detected eyes.
        Mat preprocessedFace = getPreprocessedFace(displayedFrame, faceWidth, faceCascade, eyeCascade1, eyeCascade2, preprocessLeftAndRightSeparately, &faceRect, &leftEye, &rightEye, &searchedLeftEye, &searchedRightEye);
		//file = fopen("file.txt", "w");
		////fprintf(file, "%d", "\n");
		//fprintf(file, "%d ", preprocessedFace);
		//fclose(file);

        bool gotFaceAndEyes = false;
        if (preprocessedFace.data)
            gotFaceAndEyes = true;

        // Draw an anti-aliased rectangle around the detected face.
        if (faceRect.width > 0) {
            rectangle(displayedFrame, faceRect, CV_RGB(255, 255, 0), 2, CV_AA);

            Scalar eyeColor = CV_RGB(0,255,255);
            if (leftEye.x >= 0) {   // Check if the eye was detected
                circle(displayedFrame, Point(faceRect.x + leftEye.x, faceRect.y + leftEye.y), 6, eyeColor, 1, CV_AA);
            }
            if (rightEye.x >= 0) {   // Check if the eye was detected
                circle(displayedFrame, Point(faceRect.x + rightEye.x, faceRect.y + rightEye.y), 6, eyeColor, 1, CV_AA);
            }
        }

        if (m_mode == MODE_DETECTION) {
            // Don't do anything special.
        }
		else if (m_mode == MODE_COLLECT_FACES_MANUALLY){
			if (saveManuallyClicked == 1){
				if (gotFaceAndEyes) {

					// Draw light-blue anti-aliased circles for the 2 eyes.

					// Check if this face looks somewhat different from the previously collected face.
					double imageDiff = 10000000000.0;
					if (old_prepreprocessedFace.data) {
						imageDiff = getSimilarity(preprocessedFace, old_prepreprocessedFace);
					}

					// Also record when it happened.
					double current_time = (double)getTickCount();
					double timeDiff_seconds = (current_time - old_time) / getTickFrequency();

					// Only process the face if it is noticeably different from the previous frame and there has been noticeable time gap.
					if ((imageDiff > CHANGE_IN_IMAGE_FOR_COLLECTION) && (timeDiff_seconds > CHANGE_IN_SECONDS_FOR_COLLECTION)) {

						if (currentPerson != m_numPersons){
							cout << "Name of the subject: ";
							cin >> name;
							cout << "Surname of the subject: ";
							cin >> surname;
							identities.push_back(name + " " + surname);
							currentPerson++;
						}
						cout << name + " " + surname + " " << m_numPersons;
						string newFolderPath = "./immagini/subject_" + toString(m_numPersons) + "/";
						mkdir(newFolderPath.c_str());
						string imgName = toString(val) + ".jpg";
						string imgPath = newFolderPath + imgName;
						imwrite(imgPath, displayedFrame); //displayedFrame oppure preprocessedFace
						ofstream outputFile(newFolderPath + "/identity.txt");
						outputFile << name + " " + surname;
						val++;

						//aggiunge nome e cognome della persona all'array identities
						if (currentPerson != m_numPersons){
							cout << "Nome del soggetto: ";
							cin >> name;
							cout << "Cognome del soggetto: ";
							cin >> surname;
							identities.push_back(name + " " + surname);
							currentPerson++;
						}

						// Also add the mirror image to the training set, so we have more training data, as well as to deal with faces looking to the left or right.
						Mat mirroredFace;
						flip(preprocessedFace, mirroredFace, 1);

						// Add the face images to the list of detected faces.
						preprocessedFaces.push_back(preprocessedFace);
						preprocessedFaces.push_back(mirroredFace);
						faceLabels.push_back(m_selectedPerson);
						faceLabels.push_back(m_selectedPerson);

						// Keep a reference to the latest face of each person.
						m_latestFaces[m_selectedPerson] = preprocessedFaces.size() - 2;  // Point to the non-mirrored face.
						// Show the number of collected faces. But since we also store mirrored faces, just show how many the user thinks they stored.
						cout << "Saved face " << (preprocessedFaces.size() / 2) << " for person " << m_selectedPerson << endl;

						// Make a white flash on the face, so the user knows a photo has been taken.
						Mat displayedFaceRegion = displayedFrame(faceRect);
						displayedFaceRegion += CV_RGB(90, 90, 90);

						// Keep a copy of the processed face, to compare on next iteration.
						old_prepreprocessedFace = preprocessedFace;
						old_time = current_time;
					}
				}
			}
			saveManuallyClicked = 0;
		}
        else if (m_mode == MODE_COLLECT_FACES) {
            // Check if we have detected a face.
            if (gotFaceAndEyes) {

                // Check if this face looks somewhat different from the previously collected face.
                double imageDiff = 10000000000.0;
                if (old_prepreprocessedFace.data) {
                    imageDiff = getSimilarity(preprocessedFace, old_prepreprocessedFace);
                }

                // Also record when it happened.
                double current_time = (double)getTickCount();
                double timeDiff_seconds = (current_time - old_time)/getTickFrequency();

                // Only process the face if it is noticeably different from the previous frame and there has been noticeable time gap.
                if ((imageDiff > CHANGE_IN_IMAGE_FOR_COLLECTION) && (timeDiff_seconds > CHANGE_IN_SECONDS_FOR_COLLECTION)) {

					if (currentPerson != m_numPersons){
							cout << "Nome del soggetto: ";
							cin >> name;
							cout << "Cognome del soggetto: ";
							cin >> surname;
							identities.push_back(name + " " + surname);
							currentPerson++;
					}
					cout << name + " " + surname+" "<<m_numPersons;
					string newFolderPath = "./immagini/subject_" + toString(m_numPersons) + "/";
					mkdir(newFolderPath.c_str());
					string imgName = toString(val) + ".jpg";
					string imgPath = newFolderPath + imgName;
					imwrite(imgPath, displayedFrame); //displayedFrame oppure preprocessedFace
					ofstream outputFile(newFolderPath+"/identity.txt");
					outputFile << name + " " + surname;					
					val++;

                    // Also add the mirror image to the training set, so we have more training data, as well as to deal with faces looking to the left or right.
                    Mat mirroredFace;
                    flip(preprocessedFace, mirroredFace, 1);

                    // Add the face images to the list of detected faces.
                    preprocessedFaces.push_back(preprocessedFace);
                    preprocessedFaces.push_back(mirroredFace);
                    faceLabels.push_back(m_selectedPerson);
                    faceLabels.push_back(m_selectedPerson);

                    // Keep a reference to the latest face of each person.
                    m_latestFaces[m_selectedPerson] = preprocessedFaces.size() - 2;  // Point to the non-mirrored face.
                    // Show the number of collected faces. But since we also store mirrored faces, just show how many the user thinks they stored.
                    cout << "Saved face " << (preprocessedFaces.size()/2) << " for person " << m_selectedPerson << endl;

                    // Make a white flash on the face, so the user knows a photo has been taken.
                    Mat displayedFaceRegion = displayedFrame(faceRect);
                    displayedFaceRegion += CV_RGB(90,90,90);

                    // Keep a copy of the processed face, to compare on next iteration.
                    old_prepreprocessedFace = preprocessedFace;
                    old_time = current_time;
                }
            }
        }
		else if (m_mode == MODE_DELETE_ALL) {
			// Restart everything!
			m_selectedPerson = -1;
			m_numPersons = 0;
			m_latestFaces.clear();
			preprocessedFaces.clear();
			faceLabels.clear();
			old_prepreprocessedFace = Mat();

			// Restart in Detection mode.
			m_mode = MODE_DETECTION;
		}
		else if (m_mode == MODE_CREATE_DATABASE) {
			FileStorage file("faceDB.xml", FileStorage::WRITE);
			file << "faces" << preprocessedFaces;
			file << "faceLabels" << faceLabels;
			file << "identities" << identities;
			cout << "Database created" << endl;
			file.release();
			break;
		}
		else if (m_mode == MODE_SAVE_IMAGES){
			// Check if there is enough data to train from. For Eigenfaces, we can learn just one person if we want, but for Fisherfaces,
			// we need atleast 2 people otherwise it will crash!
			bool haveEnoughData = true;
			if (strcmp(facerecAlgorithm, "FaceRecognizer.Fisherfaces") == 0) {
				if ((m_numPersons < 1) || (m_numPersons == 2 && m_latestFaces[1] < 0)) {
					cout << "Warning: Fisherfaces needs atleast 2 people, otherwise there is nothing to differentiate! Collect more data ..." << endl;
					haveEnoughData = false;
				}
			}
			if (m_numPersons < 1 || preprocessedFaces.size() <= 0 || preprocessedFaces.size() != faceLabels.size()) {
				cout << "Warning: Need some training data before it can be learnt! Collect more data ..." << endl;
				haveEnoughData = false;
			}

			if (haveEnoughData) {
				// Start training from the collected faces using Eigenfaces or a similar algorithm.
				model = learnCollectedFaces(preprocessedFaces, faceLabels, facerecAlgorithm);

				// Show the internal face recognition data, to help debugging.
				showTrainingDebugData(model, faceWidth, faceHeight, facerecAlgorithm);
				m_mode = MODE_DETECTION;
			}
			else {
				// Since there isn't enough training data, go back to the face collection mode!
				cout << "not enough data !";
				m_mode = MODE_DETECTION;
			}
		}
		//else if (m_mode == MODE_SAVE_ONE_FACE) {


		//	// Restart in Detection mode.
		//	m_mode = MODE_DETECTION;
		//}
        else {
            cerr << "ERROR: Invalid run mode " << m_mode << endl;
            exit(1);
        }

        // Show the current mode.
        if (m_mode >= 0) {
            string modeStr = "MODE: " + string(MODE_NAMES[m_mode]);
            drawString(displayedFrame, modeStr, Point(BORDER, -BORDER-2), CV_RGB(0,0,0));       // Black shadow
            drawString(displayedFrame, modeStr, Point(BORDER+1, -BORDER-1), CV_RGB(0,255,0)); // Green text
        }

        // Show the current preprocessed face in the top-center of the display.
        int cx = (displayedFrame.cols - faceWidth) / 2;
        if (preprocessedFace.data) {
            // Get a BGR version of the face, since the output is BGR color.
            Mat srcBGR = Mat(preprocessedFace.size(), CV_8UC3);
            cvtColor(preprocessedFace, srcBGR, CV_GRAY2BGR);
            // Get the destination ROI (and make sure it is within the image!).
            //min(m_gui_faces_top + i * faceHeight, displayedFrame.rows - faceHeight);
            Rect dstRC = Rect(cx, BORDER, faceWidth, faceHeight);
            Mat dstROI = displayedFrame(dstRC);
            // Copy the pixels from src to dst.
            srcBGR.copyTo(dstROI);
        }
        // Draw an anti-aliased border around the face, even if it is not shown.
        rectangle(displayedFrame, Rect(cx-1, BORDER-1, faceWidth+2, faceHeight+2), CV_RGB(200,200,200), 1, CV_AA);

        // Draw the GUI buttons into the main image.
		m_rcBtnAdd = drawButton(displayedFrame, "Add Person A", Point(BORDER, BORDER));
		m_rcBtnAddManually = drawButton(displayedFrame, "Add Person M", Point(m_rcBtnAdd.x, m_rcBtnAdd.y + m_rcBtnAdd.height), m_rcBtnAdd.width);
		m_rcBtnSaveOneFace = drawButton(displayedFrame, "Add 1 Face", Point(m_rcBtnAddManually.x, m_rcBtnAddManually.y + m_rcBtnAddManually.height), m_rcBtnAdd.width);
        //m_rcBtnDel = drawButton(displayedFrame, "Delete All", Point(m_rcBtnAdd.x, m_rcBtnAdd.y + m_rcBtnAdd.height), m_rcBtnAdd.width);
		m_rcBtnDebug = drawButton(displayedFrame, "Save Images", Point(m_rcBtnSaveOneFace.x, m_rcBtnSaveOneFace.y + m_rcBtnSaveOneFace.height), m_rcBtnAdd.width);
		m_rcBtnSaveFaces = drawButton(displayedFrame, "Create DB", Point(m_rcBtnDebug.x, m_rcBtnDebug.y + m_rcBtnDebug.height), m_rcBtnAdd.width);
		//m_rcBtnSaveOneFace = drawButton(displayedFrame, "Add 1 Face", Point(m_rcBtnSaveFaces.x, m_rcBtnSaveFaces.y + m_rcBtnSaveFaces.height), m_rcBtnAdd.width);
		m_rcBtnDel = drawButton(displayedFrame, "Delete All", Point(m_rcBtnSaveFaces.x, m_rcBtnSaveFaces.y + m_rcBtnSaveFaces.height), m_rcBtnAdd.width);
		//m_rcBtnAddManually = drawButton(displayedFrame, "Add Person M", Point(m_rcBtnSaveOneFace.x, m_rcBtnSaveOneFace.y + m_rcBtnSaveOneFace.height), m_rcBtnAdd.width);

        // Show the most recent face for each of the collected people, on the right side of the display.
        m_gui_faces_left = displayedFrame.cols - BORDER - faceWidth;
        m_gui_faces_top = BORDER;
        for (int i=0; i<m_numPersons; i++) {
            int index = m_latestFaces[i];
            if (index >= 0 && index < (int)preprocessedFaces.size()) {
                Mat srcGray = preprocessedFaces[index];
                if (srcGray.data) {
                    // Get a BGR version of the face, since the output is BGR color.
                    Mat srcBGR = Mat(srcGray.size(), CV_8UC3);
                    cvtColor(srcGray, srcBGR, CV_GRAY2BGR);
                    // Get the destination ROI (and make sure it is within the image!).
                    int y = min(m_gui_faces_top + i * faceHeight, displayedFrame.rows - faceHeight);
                    Rect dstRC = Rect(m_gui_faces_left, y, faceWidth, faceHeight);
                    Mat dstROI = displayedFrame(dstRC);
                    // Copy the pixels from src to dst.
                    srcBGR.copyTo(dstROI);
                }
            }
        }

        // Highlight the person being collected, using a red rectangle around their face.
        if (m_mode == MODE_COLLECT_FACES) {
            if (m_selectedPerson >= 0 && m_selectedPerson < m_numPersons) {
                int y = min(m_gui_faces_top + m_selectedPerson * faceHeight, displayedFrame.rows - faceHeight);
                Rect rc = Rect(m_gui_faces_left, y, faceWidth, faceHeight);
                rectangle(displayedFrame, rc, CV_RGB(255,0,0), 3, CV_AA);
            }
        }

        // Highlight the person that has been recognized, using a green rectangle around their face.
        /*if (identity >= 0 && identity < 1000) {
            int y = min(m_gui_faces_top + identity * faceHeight, displayedFrame.rows - faceHeight);
            Rect rc = Rect(m_gui_faces_left, y, faceWidth, faceHeight);
            rectangle(displayedFrame, rc, CV_RGB(0,255,0), 3, CV_AA);
        }*/

        // Show the camera frame on the screen.
        imshow(windowName, displayedFrame);

        
        // IMPORTANT: Wait for atleast 20 milliseconds, so that the image can be displayed on the screen!
        // Also checks if a key was pressed in the GUI window. Note that it should be a "char" to support Linux.
        char keypress = waitKey(20);  // This is needed if you want to see anything!

        if (keypress == VK_ESCAPE) {   // Escape Key
            // Quit the program!
            break;
        }
    }//end while
}


int main(int argc, char *argv[])
{
    CascadeClassifier faceCascade;
    CascadeClassifier eyeCascade1;
    CascadeClassifier eyeCascade2;
    VideoCapture videoCapture;
	int i = 0;

    cout << "WebcamFaceRec, by Shervin Emami (www.shervinemami.info), June 2012." << endl;
    cout << "Realtime face detection + face recognition from a webcam using LBP and Eigenfaces or Fisherfaces." << endl;
    cout << "Compiled with OpenCV version " << CV_VERSION << endl << endl;

    // Load the face and 1 or 2 eye detection XML classifiers.
    initDetectors(faceCascade, eyeCascade1, eyeCascade2);

    cout << endl;
    cout << "Hit 'Escape' in the GUI window to quit." << endl;

    // Allow the user to specify a camera number, since not all computers will be the same camera number.
    int cameraNumber = 0;   // Change this if you want to use a different camera device.
    if (argc > 1) {
        cameraNumber = atoi(argv[1]);
    }

    // Get access to the webcam.
    initWebcam(videoCapture, cameraNumber);

    // Try to set the camera resolution. Note that this only works for some cameras on
    // some computers and only for some drivers, so don't rely on it to work!
    videoCapture.set(CV_CAP_PROP_FRAME_WIDTH, DESIRED_CAMERA_WIDTH);
    videoCapture.set(CV_CAP_PROP_FRAME_HEIGHT, DESIRED_CAMERA_HEIGHT);

    // Create a GUI window for display on the screen.
    namedWindow(windowName); // Resizable window, might not work on Windows.
    // Get OpenCV to automatically call my "onMouse()" function when the user clicks in the GUI window.
    setMouseCallback(windowName, onMouse, 0);

    // Run Face Recogintion interactively from the webcam. This function runs until the user quits.
    recognizeAndTrainUsingWebcam(videoCapture, faceCascade, eyeCascade1, eyeCascade2, i);

    return 0;
}
