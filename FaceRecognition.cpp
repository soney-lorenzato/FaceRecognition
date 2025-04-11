#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <opencv2/objdetect.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect/face.hpp>
#include "opencv2/opencv.hpp"
#include "opencv2/core/types.hpp"
#include <opencv2/dnn.hpp>

#include <chrono>
#include <thread>

#include <map>
#include <vector>
#include <string>


using namespace cv;
using namespace std;


int main(int argc, char** argv)
{
	cv::Ptr<cv::FaceRecognizerSF> _recognizer = cv::FaceRecognizerSF::create(
		"face_recognition_sface_2021dec.onnx", "");

    cv::Ptr<cv::FaceDetectorYN> _detector = cv::FaceDetectorYN::create(
        "face_detection_yunet_2023mar.onnx", "", cv::Size(320, 320), 0.9F, 0.3F, 1);

	cv::Mat image_soney = imread("images\\Soney3.jpg");

	// Detect face
    Mat faces_soney;
    _detector->setInputSize(image_soney.size());
    _detector->detect(image_soney, faces_soney);
    if (faces_soney.rows < 1)
    {
        return -1;
    }

    cv::Mat aligned_soney;
    _recognizer->alignCrop(image_soney, faces_soney.row(0), aligned_soney);

	cv::Mat features_soney;
	_recognizer->feature(aligned_soney, features_soney);
    features_soney = features_soney.clone();

    const int device_id = 0;
    auto cap = cv::VideoCapture(device_id);

    cv::Mat query_frame;

    while (cv::waitKey(1) < 0)
    {
        bool has_frame = cap.read(query_frame);
        if (!has_frame)
        {
            std::cout << "No frames grabbed! Exiting ...\n";
            break;
        }

        Mat faces_camera;
		_detector->setInputSize(query_frame.size());
        _detector->setTopK(10);
        _detector->detect(query_frame, faces_camera);
        if (faces_camera.rows < 1)
        {
            cv::imshow("query_frame", query_frame);
            continue;
        }
        
        for (int i = 0; i < faces_camera.rows; ++i)
        {
            cv::Mat aligned_camera;
            _recognizer->alignCrop(query_frame, faces_camera.row(i), aligned_camera);
            //cv::imshow("aligned_camera", aligned_camera);

            // Extract features from camera
            cv::Mat features_camera;
            _recognizer->feature(aligned_camera, features_camera);
            features_camera = features_camera.clone();

            double inteiro = _recognizer->match(features_soney, features_camera, cv::FaceRecognizerSF::DisType::FR_COSINE);

            double _threshold_cosine = 0.363;
            double _threshold_norml2 = 1.128;
            bool samePerson = false;
            samePerson = (inteiro >= _threshold_cosine);
            //samePerson = (inteiro <= _threshold_norml2);

            printf("samePerson = %d / match = %f\n", samePerson, inteiro);

            // Draw bounding boxes
            int x1 = static_cast<int>(faces_camera.at<float>(i, 0));
            int y1 = static_cast<int>(faces_camera.at<float>(i, 1));
            int w = static_cast<int>(faces_camera.at<float>(i, 2));
            int h = static_cast<int>(faces_camera.at<float>(i, 3));

            cv::rectangle(query_frame,
                cv::Rect(x1, y1, w, h),
                samePerson ?
                    cv::Scalar(0,255,0) :
                    cv::Scalar(0, 0, 255),
                2);

            // Put text
            cv::putText(query_frame, 
                samePerson ? 
                    "Soney" : 
                    "Other person", 
                cv::Point(x1, y1 - 20), 
                cv::FONT_HERSHEY_DUPLEX, 
                1, 
                samePerson ? 
                    cv::Scalar(0, 255, 0) : 
                    cv::Scalar(0, 0, 255),
                2);
        }
        // diaplay camera frame with boxes and texts
        cv::imshow("query_frame", query_frame);
    }
	
    return 0;
}
