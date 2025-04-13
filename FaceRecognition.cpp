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
#include <algorithm>

using namespace cv;
using namespace std;

class FaceFeatures
{
private:
    cv::Mat _feature;
	string _name;
public:
	FaceFeatures(cv::Mat image, string name) : _feature(image), _name(name) {}
	cv::Mat getFeature() { return _feature; }
	string getName() { return _name; }
};



int main(int argc, char** argv)
{
	cv::Ptr<cv::FaceRecognizerSF> _recognizer = cv::FaceRecognizerSF::create(
		"face_recognition_sface_2021dec.onnx", "");

    cv::Ptr<cv::FaceDetectorYN> _detector = cv::FaceDetectorYN::create(
        "face_detection_yunet_2023mar.onnx", "", cv::Size(320, 320), 0.9F, 0.3F, 10);

    list<FaceFeatures> _features = {};

	cv::Mat image_soney     = imread("images\\Soney3.jpg");
    cv::Mat image_maria     = imread("images\\MariaLuiza.jpeg");

    list<tuple<string, cv::Mat>> _knownImages = { 
        {"Soney", image_soney}, 
        {"Maria Luiza", image_maria}
    };

    for (tuple<string, cv::Mat> var : _knownImages)
    {
        string name = std::get<0>(var);
        cv::Mat photo = std::get<1>(var);

        // Detect faces
        Mat faces;
        _detector->setInputSize(photo.size());
        _detector->detect(photo, faces);
        if (faces.rows < 1)
        {
            printf("Person %s not loaded !!!", name.c_str());
            continue;
        }

        cv::Mat aligned_photo;
        _recognizer->alignCrop(photo, faces.row(0), aligned_photo);

        cv::Mat features_photo;
        _recognizer->feature(aligned_photo, features_photo);
        features_photo = features_photo.clone();

        // Save Soney features
        _features.push_back(FaceFeatures(features_photo, name));
    }

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

            // Extract features from camera
            cv::Mat features_camera;
            _recognizer->feature(aligned_camera, features_camera);
            features_camera = features_camera.clone();

			bool foundAPerson = false;
			string foundName = "Other person";

            for (FaceFeatures feat : _features)
            {
                double inteiro = _recognizer->match(feat.getFeature(), features_camera, cv::FaceRecognizerSF::DisType::FR_COSINE);
                double _threshold_cosine = 0.363;
                double _threshold_norml2 = 1.128;
                bool samePerson = false;
                samePerson = (inteiro >= _threshold_cosine);
                //samePerson = (inteiro <= _threshold_norml2);
                
                if (samePerson) {
                    foundAPerson = true;
					foundName = feat.getName();
                    printf("samePerson = %d / match = %f / name = %s\n", samePerson, inteiro, feat.getName().c_str());
                    break;
                }
            }

            // Draw bounding boxes
            int x1 = static_cast<int>(faces_camera.at<float>(i, 0));
            int y1 = static_cast<int>(faces_camera.at<float>(i, 1));
            int w = static_cast<int>(faces_camera.at<float>(i, 2));
            int h = static_cast<int>(faces_camera.at<float>(i, 3));

            cv::rectangle(query_frame,
                cv::Rect(x1, y1, w, h),
                foundAPerson ?
                cv::Scalar(0, 255, 0) :
                cv::Scalar(0, 0, 255),
                2);

            // Put text
            cv::putText(query_frame,
                foundAPerson ?
                foundName.c_str() :
                "Other person",
                cv::Point(x1, y1 - 20),
                cv::FONT_HERSHEY_DUPLEX,
                1,
                foundAPerson ?
                cv::Scalar(0, 255, 0) :
                cv::Scalar(0, 0, 255),
                2);

        }
        // diaplay camera frame with boxes and texts
        cv::imshow("query_frame", query_frame);
    }
	
    return 0;
}
